# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Internal support classes for experiment state, routing, and metrics.

This module is transitional: its logic is moving into the logging primitives
(:mod:`litlogger.primitives` and :mod:`litlogger.media`) and it will be
deleted once the migration completes.
"""

import re
from typing import TYPE_CHECKING, Callable

from lightning_sdk.lightning_cloud.openapi import V1MediaType

from litlogger.media import File, Model, _media_download_fn, _wrap_media_file
from litlogger.primitives import (
    Metadata,
    Metric,
    RestoredFiles,
    _to_v1_media_type,
    model_version_sort_key,
    natural_sort_key,
)
from litlogger.series import Series
from litlogger.session import ExperimentSession
from litlogger.types import MediaType

if TYPE_CHECKING:
    from litlogger.experiment import Experiment


def _session_for(exp: "Experiment") -> ExperimentSession:
    """Return the experiment's session, or assemble one from its parts.

    Transitional glue for the support shims: real experiments carry a session;
    the partially-populated mock experiments used by the legacy unit tests do
    not, so missing infrastructure degrades to ``None`` for writes that never
    touch it.
    """
    session = getattr(exp, "_session", None)
    if isinstance(session, ExperimentSession):
        return session

    metrics_api = getattr(exp, "_metrics_api", None)
    return ExperimentSession(
        client=getattr(metrics_api, "client", None),  # type: ignore[arg-type]
        metrics_api=metrics_api,  # type: ignore[arg-type]
        media_api=getattr(exp, "_media_api", None),  # type: ignore[arg-type]
        artifacts_api=getattr(exp, "_artifacts_api", None),  # type: ignore[arg-type]
        teamspace=getattr(exp, "_teamspace", None),
        experiment=exp,
        queue=getattr(exp, "_metrics_queue", None),  # type: ignore[arg-type]
        stats=getattr(exp, "_stats", None),  # type: ignore[arg-type]
        store_step=bool(getattr(exp, "store_step", True)),
        store_created_at=bool(getattr(exp, "store_created_at", False)),
        last_steps=getattr(exp, "_resumed_steps", None) or {},
        background=getattr(exp, "_manager", None),
    )


class ExperimentSeriesSupport:
    """Helpers for key registration, series creation, and metric batching."""

    @staticmethod
    def register_key_type(exp: "Experiment", key: str, key_type: str) -> None:
        if key in exp._key_types:
            if exp._key_types[key] != key_type:
                raise KeyError(f"Key {key!r} is already used as {exp._key_types[key]}, cannot use as {key_type}")
            return
        exp._key_types[key] = key_type

    @staticmethod
    def ensure_series(exp: "Experiment", key: str) -> Series:
        if key not in exp._series:
            exp._series[key] = Series(exp, key)
        return exp._series[key]

    @staticmethod
    def log_metric_value(exp: "Experiment", key: str, value: float, step: int | None = None) -> None:
        Metric(key, value, step=step).enqueue(_session_for(exp))


class ExperimentStateSupport:
    """Helpers for remote-state reconstruction and metadata refresh."""

    @staticmethod
    def natural_sort_key(value: str | None) -> tuple[object, ...]:
        return natural_sort_key(value)

    @staticmethod
    def model_version_sort_key(version_info: object) -> tuple[object, ...]:
        return model_version_sort_key(version_info)

    @staticmethod
    def remote_model_from_version(exp: "Experiment", key: str, model_key: str, version_info: object) -> Model:
        return Model._from_version(_session_for(exp), key, model_key, version_info)

    @staticmethod
    def resolve_remote_model(exp: "Experiment", key: str) -> Model | Series | None:
        cached = exp._model_lookup_cache.get(key)
        if cached is not None or key in exp._missing_model_keys:
            return cached

        resolved = Model._resolve(_session_for(exp), key)
        if resolved is None:
            exp._missing_model_keys.add(key)
            return None

        if isinstance(resolved, list):
            series = Series(exp, key)
            series._type = "file"
            series._values = list(resolved)
            exp._model_lookup_cache[key] = series
            return series

        exp._model_lookup_cache[key] = resolved
        return resolved

    @staticmethod
    def rebuild_state(exp: "Experiment") -> None:
        """Rebuild state from remote metadata, steps, artifacts, and media."""
        # TODO: add BE support for restoring model states as well
        session = _session_for(exp)

        for name, value in Metadata._current_tags(session).items():
            exp._key_types[name] = "metadata"
            exp._metadata_values[name] = value

        metric_values = Metric._restore_values(session)
        for name in exp._resumed_steps:
            exp._key_types[name] = "metric"
            series = Series(exp, name)
            series._type = "metric"
            if name in metric_values:
                series._values = list(metric_values[name])
            exp._series[name] = series

        ExperimentStateSupport._merge_restored(exp, File._restore_all(session, dict(exp._key_types)))
        ExperimentStateSupport._merge_restored(exp, File._restore_media(session, dict(exp._key_types)))

    @staticmethod
    def _merge_restored(exp: "Experiment", restored: RestoredFiles) -> None:
        """Register one restore pass's results in the experiment's local state."""
        for key, file in restored.statics.items():
            exp._key_types[key] = "static_file"
            exp._static_files[key] = file
        for key, values in restored.series.items():
            exp._key_types[key] = "file_series"
            series = Series(exp, key)
            series._type = "file"
            series._values = values
            exp._series[key] = series

    @staticmethod
    def create_download_fn(exp: "Experiment", key: str) -> Callable[[str], str]:
        def _download(path: str) -> str:
            file = File(path)
            file._bind_remote(
                _session_for(exp),
                remote_path=key,
                cloud_account=getattr(exp._metrics_store, "cluster_id", None),
            )
            return file.save(path)

        return _download

    @staticmethod
    def bind_remote_model(exp: "Experiment", key: str, value: Model, model_name: str) -> None:
        value._bind_remote_model(key=key, model_name=model_name)

    @staticmethod
    def model_experiment_name(exp: "Experiment", key: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", key).strip("-") or "model"

    @staticmethod
    def code_tags(exp: "Experiment") -> dict[str, str]:
        exp._update_metrics_store()
        tags = getattr(exp._metrics_store, "tags", None) or []
        return {tag.name: tag.value for tag in tags if tag.from_code}

    @staticmethod
    def create_media_download_fn(
        exp: "Experiment", storage_path: str, cloud_account: str | None = None
    ) -> Callable[[str], str]:
        return _media_download_fn(_session_for(exp), storage_path, cloud_account)

    @staticmethod
    def wrap_media_file(exp: "Experiment", media_name: str, media_type: V1MediaType) -> File:
        return _wrap_media_file(media_name, media_type)

    @staticmethod
    def update_metrics_store(exp: "Experiment") -> None:
        resp = exp._metrics_api.get_experiment_metrics_by_name(
            exp._teamspace.id,
            name=exp._metrics_store.name,
        )

        if resp is not None:
            exp._metrics_store = resp


class ExperimentIOSupport:
    """Helpers for metadata, artifact, media, and model routing."""

    @staticmethod
    def media_type_to_v1(exp: "Experiment", media_type: MediaType) -> V1MediaType:
        return _to_v1_media_type(media_type)

    @staticmethod
    def upload_media(
        exp: "Experiment",
        name: str,
        file_path: str,
        media_type: MediaType,
        step: int | None = None,
        epoch: int | None = None,
        caption: str | None = None,
    ) -> None:
        exp._media_api.upload_media(
            experiment_id=exp._metrics_store.id,
            teamspace=exp._teamspace,
            file_path=file_path,
            name=name,
            media_type=exp._media_type_to_v1(media_type),
            step=step,
            epoch=epoch,
            caption=caption,
        )
        exp._stats.media_logged += 1

    @staticmethod
    def upload_media_value(
        exp: "Experiment",
        key: str,
        value: File,
        name: str | None = None,
        step: int | None = None,
        epoch: int | None = None,
        caption: str | None = None,
    ) -> None:
        upload_path = value._get_upload_path()
        media_name = name or key
        exp._upload_media(media_name, upload_path, value._media_type, step=step, epoch=epoch, caption=caption)
        value.name = media_name
        value._cleanup()

    @staticmethod
    def upload_model_value(exp: "Experiment", key: str, value: Model) -> None:
        """Upload a model through litmodels and bind the remote wrapper."""
        # TODO: Persist model recovery data via backend-supported experiment
        # bindings so resumed experiments can rebuild these wrappers.
        cloud_account = exp._metrics_store.cluster_id
        model_name = value._log_model(
            experiment_name=exp.name,
            teamspace=exp._teamspace,
            key=exp._model_experiment_name(key),
            experiment=exp,
            cloud_account=cloud_account if isinstance(cloud_account, str) else None,
        )
        exp._stats.models_logged += 1
        exp._bind_remote_model(key, value, model_name)

    @staticmethod
    def log_file_series_value(exp: "Experiment", key: str, value: File, index: int, step: int | None = None) -> None:
        if value._media_type == MediaType.MODEL and not isinstance(value, Model):
            raise TypeError("Model media values must use the Model wrapper.")
        value._log_key = key
        value._series_index = index
        value._series_step = step
        value.log(_session_for(exp))

    @staticmethod
    def set_metadata_value(exp: "Experiment", key: str, value: str) -> None:
        Metadata(key, value).log(_session_for(exp))

    @staticmethod
    def set_static_file(exp: "Experiment", key: str, value: File) -> None:
        if value._media_type == MediaType.MODEL and not isinstance(value, Model):
            raise TypeError("Model media values must use the Model wrapper.")
        value._log_key = key
        value._series_index = None
        value._series_step = None
        value.log(_session_for(exp))
