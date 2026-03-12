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
"""Internal support classes for experiment state, routing, and metrics."""

import contextlib
import re
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from lightning_sdk.lightning_cloud.openapi import V1MediaType

from litlogger.media import File, Image, Model, Text
from litlogger.series import Series
from litlogger.types import MediaType, Metrics, MetricValue, PhaseType

if TYPE_CHECKING:
    from litlogger.experiment import Experiment


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
        if exp._manager.exception is not None:
            raise exp._manager.exception

        created_at = datetime.now() if exp.store_created_at else None
        actual_step = step if exp.store_step else None
        mv = MetricValue(value=value, created_at=created_at, step=actual_step)
        batch: dict[str, Metrics] = {key: Metrics(name=key, values=[mv])}
        exp._metrics_queue.put(batch)
        exp._stats.record_metric(key, value)


class ExperimentStateSupport:
    """Helpers for remote-state reconstruction and metadata refresh."""

    @staticmethod
    def rebuild_state(exp: "Experiment") -> None:
        """Rebuild state from remote metadata, trackers, artifacts, and media.

        TODO: Add backend-supported recovery for model bindings so resumed
        experiments can reconstruct ``Model`` values without storing them in
        frontend-visible metadata tags.
        """
        tags = getattr(exp._metrics_store, "tags", None) or []
        for tag in tags:
            if tag.from_code:
                exp._key_types[tag.name] = "metadata"
                exp._metadata_values[tag.name] = tag.value

        trackers = exp._metrics_api.get_trackers_from_metrics_store(exp._metrics_store)
        if trackers:
            for name in trackers:
                exp._key_types[name] = "metric"

        artifacts = getattr(exp._metrics_store, "artifacts", None) or []
        for artifact in artifacts:
            key = artifact.path if hasattr(artifact, "path") else str(artifact)
            if key not in exp._key_types:
                exp._key_types[key] = "static_file"
                file = File(key)
                file.name = key
                file._download_fn = exp._create_download_fn(key)
                exp._static_files[key] = file

        with contextlib.suppress(AttributeError):
            media_response = exp._media_api.client.lit_logger_service_list_lit_logger_media(
                project_id=exp._teamspace.id,
                metrics_stream_id=exp._metrics_store.id,
            )
            media_items = getattr(media_response, "media", None)
            if not isinstance(media_items, list):
                media_items = []
            series_entries: dict[str, list[tuple[int, File]]] = {}
            for media in media_items:
                name = media.name or media.storage_path or media.id
                storage_path = media.storage_path or name
                wrapped = exp._wrap_media_file(name, media.media_type)
                wrapped.name = name
                wrapped._download_fn = exp._create_media_download_fn(storage_path, media.cluster_id)

                match = re.match(r"^(?P<key>.+)/(?P<index>\d+)$", name)
                if match:
                    key = match.group("key")
                    index = int(match.group("index"))
                    if key in exp._key_types and exp._key_types[key] != "file_series":
                        continue
                    exp._key_types[key] = "file_series"
                    series_entries.setdefault(key, []).append((index, wrapped))
                    continue

                if name not in exp._key_types:
                    exp._key_types[name] = "static_file"
                    exp._static_files[name] = wrapped

            for key, entries in series_entries.items():
                series = Series(exp, key)
                series._type = "file"
                series._values = [value for _, value in sorted(entries, key=lambda item: item[0])]
                exp._series[key] = series

    @staticmethod
    def create_download_fn(exp: "Experiment", key: str) -> Callable[[str], str]:
        def _download(path: str) -> str:
            file = File(path)
            file.bind_remote_artifact(
                teamspace=exp._teamspace,
                experiment_name=exp.name,
                remote_path=key,
                client=exp._artifacts_api.client,
                cloud_account=getattr(exp._metrics_store, "cluster_id", None),
            )
            return file.save(path)

        return _download

    @staticmethod
    def bind_remote_model(exp: "Experiment", key: str, value: Model, model_name: str) -> None:
        value.bind_remote_model(key=key, model_name=model_name)

    @staticmethod
    def model_experiment_name(exp: "Experiment", key: str) -> str:
        safe_key = re.sub(r"[^A-Za-z0-9._-]+", "-", key).strip("-") or "model"
        return f"{exp.name}-{safe_key}"

    @staticmethod
    def code_tags(exp: "Experiment") -> dict[str, str]:
        exp._update_metrics_store()
        tags = getattr(exp._metrics_store, "tags", None) or []
        return {tag.name: tag.value for tag in tags if tag.from_code}

    @staticmethod
    def create_media_download_fn(
        exp: "Experiment", storage_path: str, cloud_account: str | None = None
    ) -> Callable[[str], str]:
        def _download(path: str) -> str:
            exp._teamspace.download_file(storage_path, file_path=path, cloud_account=cloud_account)
            return path

        return _download

    @staticmethod
    def wrap_media_file(exp: "Experiment", media_name: str, media_type: V1MediaType) -> File:
        if media_type == V1MediaType.IMAGE:
            return Image(media_name)
        if media_type == V1MediaType.TEXT:
            text = Text("")
            text.path = media_name
            return text
        return File(media_name)

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
        if media_type == MediaType.IMAGE:
            return V1MediaType.IMAGE
        if media_type == MediaType.TEXT:
            return V1MediaType.TEXT
        raise ValueError(f"Unsupported media type for file upload: {media_type}")

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
        """Upload a model through litmodels and bind the remote wrapper.

        TODO: Persist model recovery data via backend-supported experiment
        bindings so resumed experiments can rebuild these wrappers.
        """
        experiment_name = exp._model_experiment_name(key)
        cloud_account = exp._metrics_store.cluster_id
        model_name = value.log_model(
            experiment_name=experiment_name,
            teamspace=exp._teamspace,
            cloud_account=cloud_account if isinstance(cloud_account, str) else None,
        )
        exp._stats.models_logged += 1
        exp._bind_remote_model(key, value, model_name)

    @staticmethod
    def log_file_series_value(exp: "Experiment", key: str, value: File, index: int, step: int | None = None) -> None:
        if value._media_type == MediaType.MODEL:
            exp._upload_model_value(f"{key}/{index}", value)
            return

        if value._media_type != MediaType.FILE:
            exp._upload_media_value(key, value, name=f"{key}/{index}", step=step)
            return

        remote_path = f"{key}/{index}"
        value.log_artifact(
            teamspace=exp._teamspace,
            metrics_store=exp._metrics_store,
            remote_path=remote_path,
            client=exp._artifacts_api.client,
            experiment_name=exp.name,
        )
        exp._stats.artifacts_logged += 1

    @staticmethod
    def set_metadata_value(exp: "Experiment", key: str, value: str) -> None:
        current_tags = ExperimentStateSupport.code_tags(exp)
        current_tags[key] = value
        exp._metrics_api.update_experiment_metrics(
            teamspace_id=exp._teamspace.id,
            metrics_store_id=exp._metrics_store.id,
            phase=PhaseType.RUNNING,
            metadata=current_tags,
        )

    @staticmethod
    def set_static_file(exp: "Experiment", key: str, value: File) -> None:
        if value._media_type == MediaType.MODEL:
            exp._upload_model_value(key, value)
            return

        if value._media_type != MediaType.FILE:
            exp._upload_media_value(key, value)
            return

        value.log_artifact(
            teamspace=exp._teamspace,
            metrics_store=exp._metrics_store,
            remote_path=key,
            client=exp._artifacts_api.client,
            experiment_name=exp.name,
        )
        exp._stats.artifacts_logged += 1
