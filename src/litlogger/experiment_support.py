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

from litlogger.media import File, Image, Model, Text, Video
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
    def natural_sort_key(value: str | None) -> tuple[object, ...]:
        if not value:
            return ("",)
        parts = re.split(r"(\d+)", value)
        key: list[object] = []
        for part in parts:
            if not part:
                continue
            key.append(int(part) if part.isdigit() else part)
        return tuple(key)

    @staticmethod
    def model_version_sort_key(version_info: object) -> tuple[object, ...]:
        index = getattr(version_info, "index", None)
        if isinstance(index, int):
            return (0, index)

        created_at = getattr(version_info, "created_at", None)
        if isinstance(created_at, datetime):
            return (1, created_at)

        updated_at = getattr(version_info, "updated_at", None)
        if isinstance(updated_at, datetime):
            return (2, updated_at)

        version = getattr(version_info, "version", None)
        return (3, *ExperimentStateSupport.natural_sort_key(version))

    @staticmethod
    def remote_model_from_version(exp: "Experiment", key: str, model_key: str, version_info: object) -> Model:
        metadata = getattr(version_info, "metadata", None) or {}
        kind = "object" if metadata.get("litModels.integration") == "save_model" else "artifact"
        version = getattr(version_info, "version", None)
        registry_name = f"{exp._teamspace.owner.name}/{exp._teamspace.name}/{model_key}"
        if version:
            registry_name += f":{version}"

        model = Model.from_remote(registry_name, kind, version=version)
        model._bind_remote_model(key=key, model_name=registry_name)
        return model

    @staticmethod
    def resolve_remote_model(exp: "Experiment", key: str) -> Model | Series | None:
        cached = exp._model_lookup_cache.get(key)
        if cached is not None or key in exp._missing_model_keys:
            return cached

        model_key = ExperimentStateSupport.model_experiment_name(exp, key)
        try:
            models = exp._teamspace.list_models()
            model_info = next((model for model in models if getattr(model, "name", None) == model_key), None)
            if model_info is None:
                exp._missing_model_keys.add(key)
                return None

            versions = exp._teamspace.list_model_versions(model_key)
            complete_versions = [version for version in versions if getattr(version, "upload_complete", True)]
            if not complete_versions:
                exp._missing_model_keys.add(key)
                return None
            complete_versions.sort(key=ExperimentStateSupport.model_version_sort_key)

            if len(complete_versions) == 1:
                model = ExperimentStateSupport.remote_model_from_version(exp, key, model_key, complete_versions[0])
                exp._model_lookup_cache[key] = model
                return model

            series = Series(exp, key)
            series._type = "file"
            series._values = [
                ExperimentStateSupport.remote_model_from_version(exp, key, model_key, version_info)
                for version_info in complete_versions
            ]
            exp._model_lookup_cache[key] = series
            return series
        except Exception:
            exp._missing_model_keys.add(key)
            return None

    @staticmethod
    def rebuild_state(exp: "Experiment") -> None:
        """Rebuild state from remote metadata, steps, artifacts, and media."""
        # TODO: add BE support for restoring model states as well
        exp._update_metrics_store()
        tags = getattr(exp._metrics_store, "tags", None) or []
        for tag in tags:
            if tag.from_code:
                exp._key_types[tag.name] = "metadata"
                exp._metadata_values[tag.name] = tag.value

        for name in exp._resumed_steps:
            exp._key_types[name] = "metric"

        artifacts = getattr(exp._metrics_store, "artifacts", None) or []
        with contextlib.suppress(AttributeError):
            artifact_response = exp._metrics_api.client.lit_logger_service_list_logger_artifacts(
                project_id=exp._teamspace.id,
                metrics_stream_id=exp._metrics_store.id,
            )
            listed_artifacts = getattr(artifact_response, "logger_artifacts", None)
            if isinstance(listed_artifacts, list):
                artifacts = listed_artifacts
        artifact_series_entries: dict[str, list[tuple[int, File]]] = {}
        for artifact in artifacts:
            name = artifact.path if hasattr(artifact, "path") else str(artifact)
            wrapped = File(name)
            wrapped.name = name
            wrapped._download_fn = exp._create_download_fn(name)

            # Treat names like "reports/3" as the 4th entry of a file series keyed by "reports".
            match = re.match(r"^(?P<key>.+)/(?P<index>\d+)$", name)
            if match:
                key = match.group("key")
                index = int(match.group("index"))
                if key in exp._key_types and exp._key_types[key] != "file_series":
                    continue
                exp._key_types[key] = "file_series"
                artifact_series_entries.setdefault(key, []).append((index, wrapped))
                continue

            if name in exp._key_types:
                continue
            exp._key_types[name] = "static_file"
            exp._static_files[name] = wrapped

        for key, file_entries in artifact_series_entries.items():
            series = Series(exp, key)
            series._type = "file"
            series._values = [value for _, value in sorted(file_entries)]
            exp._series[key] = series

        with contextlib.suppress(AttributeError):
            media_response = exp._media_api.client.lit_logger_service_list_lit_logger_media(
                project_id=exp._teamspace.id,
                metrics_stream_id=exp._metrics_store.id,
            )
            media_items = getattr(media_response, "media", None)
            if not isinstance(media_items, list):
                media_items = []
            series_entries: dict[str, list[tuple[int, File]]] = {}
            direct_media_entries: dict[str, list[tuple[int | None, int, File]]] = {}
            for position, media in enumerate(media_items):
                name = media.name or media.storage_path or media.id
                storage_path = media.storage_path or name
                wrapped = exp._wrap_media_file(name, media.media_type)
                wrapped.name = name
                wrapped._download_fn = exp._create_media_download_fn(storage_path, media.cluster_id)

                # Treat names like "logs/3" as the 4th entry of a media series keyed by "logs".
                match = re.match(r"^(?P<key>.+)/(?P<index>\d+)$", name)
                if match:
                    key = match.group("key")
                    index = int(match.group("index"))
                    if key in exp._key_types and exp._key_types[key] != "file_series":
                        continue
                    exp._key_types[key] = "file_series"
                    series_entries.setdefault(key, []).append((index, wrapped))
                    continue

                direct_media_entries.setdefault(name, []).append((getattr(media, "step", None), position, wrapped))

            for name, media_entries in direct_media_entries.items():
                if name in exp._key_types:
                    continue
                if len(media_entries) == 1 and name not in exp._key_types:
                    exp._key_types[name] = "static_file"
                    exp._static_files[name] = media_entries[0][2]
                    continue

                exp._key_types[name] = "file_series"
                series_values = series_entries.setdefault(name, [])
                for step, position, wrapped in media_entries:
                    sort_index = step if isinstance(step, int) else position
                    series_values.append((sort_index, wrapped))

            for key, file_entries in series_entries.items():
                series = Series(exp, key)
                series._type = "file"
                series._values = [value for _, value in sorted(file_entries)]
                exp._series[key] = series

    @staticmethod
    def create_download_fn(exp: "Experiment", key: str) -> Callable[[str], str]:
        def _download(path: str) -> str:
            file = File(path)
            file._bind_remote_artifact(
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

        if media_type == V1MediaType.VIDEO:
            return Video(media_name)
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
        if media_type == MediaType.VIDEO:
            return V1MediaType.VIDEO
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
        if value._media_type == MediaType.MODEL:
            if not isinstance(value, Model):
                raise TypeError("Model media values must use the Model wrapper.")
            if not value._version_provided:
                value.version = f"v{index + 1}"
            exp._upload_model_value(key, value)
            return

        if value._media_type != MediaType.FILE:
            exp._upload_media_value(key, value, name=key, step=step)
            return

        remote_path = f"{key}/{index}"
        value._log_artifact(
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
            if not isinstance(value, Model):
                raise TypeError("Model media values must use the Model wrapper.")
            exp._upload_model_value(key, value)
            return

        if value._media_type != MediaType.FILE:
            exp._upload_media_value(key, value)
            return

        value._log_artifact(
            teamspace=exp._teamspace,
            metrics_store=exp._metrics_store,
            remote_path=key,
            client=exp._artifacts_api.client,
            experiment_name=exp.name,
        )
        exp._stats.artifacts_logged += 1
