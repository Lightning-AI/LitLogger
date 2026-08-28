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
"""Model logging primitive."""

import os
from typing import TYPE_CHECKING, Any, Callable

from lightning_sdk import Teamspace
from typing_extensions import override

from litlogger.models import download_model, load_model, save_model, upload_model
from litlogger.primitives._utils import model_version_sort_key, sanitize_model_key
from litlogger.primitives.file import File
from litlogger.types import MediaType

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


def _sanitize_version_for_model_name(version: str) -> str:
    """Sanitize version string for use in model names."""
    return version.replace(":", "-")


class Model(File):
    """Represents a model to be logged.

    Can take either a Python model object or a local path to a pre-saved model
    artifact. Uploads are handled through the model registry.

    Args:
        data: Python model object or path to a pre-saved model file/directory.
        name: Optional registry name override for this model. Defaults to the experiment name.
        version: Optional model version. Defaults to ``"latest"``.
        metadata: Optional metadata to associate with the model upload.
        staging_dir: Optional local staging directory for object-based uploads.
        description: Optional human-readable description of the model.
    """

    def __init__(
        self,
        data: Any,
        name: str | None = None,
        version: str | None = None,
        metadata: dict[str, str] | None = None,
        staging_dir: str | None = None,
        description: str = "",
        _kind: str | None = None,
    ) -> None:
        self._data = data
        self.registry_name = name
        self.version = version or "latest"
        self._version_provided = version is not None
        self.metadata = metadata
        self.staging_dir = staging_dir
        self._kind = _kind or ("artifact" if isinstance(data, str) else "object")
        self._model_name: str | None = None
        self._load_fn: Callable[[str | None], Any] | None = None

        if isinstance(data, str):
            super().__init__(data, description=description)
        else:
            super().__init__("", description=description)

    @classmethod
    def from_remote(cls: type["Model"], model_name: str, kind: str, version: str | None = None) -> "Model":
        """Create a remote-bound model wrapper for resumed experiments."""
        data = model_name if kind == "artifact" else object()
        model = cls(data, version=version, _kind=kind)
        if kind == "artifact":
            model.path = model_name
        return model

    @property
    @override
    def _media_type(self) -> MediaType:
        return MediaType.MODEL

    @property
    def _model_kind(self) -> str:
        return self._kind

    def _get_upload_path(self) -> str:
        if isinstance(self._data, str):
            if os.path.isfile(self.path):
                return super()._get_upload_path()
            return self.path
        return super()._get_upload_path()

    def _registry_name(self, experiment_name: str, teamspace: Teamspace) -> str:
        """Resolve the registry name for this model."""
        model_name = f"{teamspace.owner.name}/{teamspace.name}/{experiment_name}"
        if self.version:
            model_name += f":{_sanitize_version_for_model_name(self.version)}"
        return model_name

    def _bind_remote_model(self, *, key: str, model_name: str) -> None:
        """Bind remote model download/load behavior to this wrapper."""
        self.name = key
        self._model_name = model_name

        def _download(path: str) -> str:
            result = download_model(name=model_name, download_dir=path, progress_bar=False)
            return result if isinstance(result, str) else result[0]

        def _load(staging_dir: str | None = None) -> Any:
            return load_model(name=model_name, download_dir=staging_dir or ".")

        self._download_fn = _download
        if self._model_kind == "object":
            self._load_fn = _load
        else:
            self._load_fn = None

    def _log_model(
        self,
        *,
        experiment_name: str,
        teamspace: Teamspace,
        key: str | None = None,
        experiment: Any = None,
        cloud_account: str | None = None,
        verbose: bool = False,
    ) -> str:
        """Upload this model to the registry and return its registry name."""
        model_name = self._registry_name(self.registry_name or key or experiment_name, teamspace)

        try:
            if self._model_kind == "artifact":
                upload_model(
                    name=model_name,
                    model=self._get_upload_path(),
                    verbose=False,
                    progress_bar=verbose,
                    cloud_account=cloud_account,
                    metadata=self.metadata,
                    experiment=experiment,
                )
            else:
                if self.staging_dir is not None:
                    os.makedirs(self.staging_dir, exist_ok=True)
                save_model(
                    name=model_name,
                    model=self._data,
                    staging_dir=self.staging_dir,
                    verbose=False,
                    progress_bar=verbose,
                    cloud_account=cloud_account,
                    metadata=self.metadata,
                    experiment=experiment,
                )
        finally:
            self._cleanup()
        return model_name

    @classmethod
    def _from_version(
        cls: type["Model"], session: "ExperimentSession", key: str, model_key: str, version_info: object
    ) -> "Model":
        """Build a remote-bound model wrapper for one registry version."""
        metadata = getattr(version_info, "metadata", None) or {}
        kind = "object" if metadata.get("litModels.integration") == "save_model" else "artifact"
        version = getattr(version_info, "version", None)
        registry_name = f"{session.teamspace.owner.name}/{session.teamspace.name}/{model_key}"
        if version:
            registry_name += f":{version}"

        model = cls.from_remote(registry_name, kind, version=version)
        model._bind_remote_model(key=key, model_name=registry_name)
        return model

    @classmethod
    def _resolve(cls: type["Model"], session: "ExperimentSession", key: str) -> "Model | list[Model] | None":
        """Look up an experiment key in the model registry (lazy restore).

        Returns a single bound Model, an ordered list of them (one per
        complete version), or None when nothing matches. Lookup failures are
        propagated so callers do not cache transient failures as missing keys.
        """
        model_key = sanitize_model_key(key)
        models = session.teamspace.list_models()
        model_info = next((model for model in models if getattr(model, "name", None) == model_key), None)
        if model_info is None:
            return None

        versions = session.teamspace.list_model_versions(model_key)
        complete_versions = [version for version in versions if getattr(version, "upload_complete", True)]
        if not complete_versions:
            return None
        complete_versions.sort(key=model_version_sort_key)

        if len(complete_versions) == 1:
            return cls._from_version(session, key, model_key, complete_versions[0])
        return [cls._from_version(session, key, model_key, version_info) for version_info in complete_versions]

    @override
    def log(self, session: "ExperimentSession") -> None:
        """Upload this model to the registry now, in the caller's thread.

        Series elements without an explicit version are auto-versioned from
        their position (``v{index + 1}``).
        """
        if self._series_index is not None and not self._version_provided:
            self.version = f"v{self._series_index + 1}"

        key = self._log_key
        cloud_account = getattr(session.metrics_store, "cluster_id", None)
        model_name = self._log_model(
            experiment_name=session.experiment_name,
            teamspace=session.teamspace,
            key=sanitize_model_key(key) if key is not None else None,
            experiment=session.experiment,
            cloud_account=cloud_account if isinstance(cloud_account, str) else None,
        )
        session.stats.models_logged += 1
        self._bind_remote_model(key=key if key is not None else model_name, model_name=model_name)

    def load(self, staging_dir: str | None = None) -> Any:
        """Load a remote model object via the registry helpers."""
        if self._read_barrier is not None:
            self._read_barrier()
        if self._load_fn is None:
            raise RuntimeError("Model has no remote load context. It must be uploaded to an experiment first.")
        return self._load_fn(staging_dir)
