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
"""Media types for litlogger.

These classes represent files and media objects that can be logged to experiments.
File wraps a local path, while other media objects can accept Python objects and render
them to temporary files for upload.
"""

import os
import tempfile
from importlib import import_module
from typing import Any, Callable

from lightning_sdk import Teamspace
from litmodels import download_model, load_model, save_model, upload_model
from typing_extensions import override

from litlogger.api.artifacts_api import ArtifactsApi
from litlogger.api.client import LitRestClient
from litlogger.types import MediaType


def _sanitize_version_for_model_name(version: str) -> str:
    """Sanitize version string for use in model names."""
    return version.replace(":", "-")


class File:
    """Represents a file to be logged to the experiment.

    Args:
        path: Path to the local file.
        description: Optional human-readable description of the file.
    """

    def __init__(self, path: str, description: str = "") -> None:
        self.path = path
        self.name: str = ""
        self.description = description
        self._temp_path: str | None = None
        self._download_fn: Callable[[str], str] | None = None

    def _get_upload_path(self) -> str:
        """Get a stable path for upload.

        Creates a hardlink to a temp location so the original file can be
        safely modified or deleted while a background upload is in progress.
        Falls back to a copy if hardlinking is not supported, or returns the
        original path if the file doesn't exist yet.
        """
        if not self.path or not os.path.exists(self.path):
            return self.path
        try:
            suffix = os.path.splitext(self.path)[1]
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            os.unlink(tmp)
            os.link(self.path, tmp)
            self._temp_path = tmp
            return tmp
        except OSError:
            import shutil

            suffix = os.path.splitext(self.path)[1]
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            shutil.copy2(self.path, tmp)
            self._temp_path = tmp
            return tmp

    def _cleanup(self) -> None:
        """Clean up any temporary files created during upload."""
        if self._temp_path is not None and os.path.exists(self._temp_path):
            try:
                os.unlink(self._temp_path)
            except PermissionError:
                # Windows cannot unlink a file while another handle is still open.
                # Leave the temp path in place so a later cleanup attempt can retry.
                return
            self._temp_path = None

    def save(self, path: str) -> str:
        """Download the remote file to a local path.

        Only works for files that have been uploaded to an experiment.

        Args:
            path: Local path where the file should be saved.

        Returns:
            str: The local path where the file was saved.

        Raises:
            RuntimeError: If the file has no remote download context.
        """
        if self._download_fn is None:
            raise RuntimeError("File has no remote context. It must be uploaded to an experiment first.")
        return self._download_fn(path)

    def _artifact_display_path(self, remote_path: str | None = None) -> str:
        """Resolve the display path used for artifact storage."""
        if remote_path is not None:
            return remote_path.replace("\\", "/")

        try:
            rel_path = os.path.relpath(self.path)
        except ValueError:
            rel_path = None

        if rel_path is not None and not rel_path.startswith(".."):
            return rel_path.replace("\\", "/")
        return os.path.basename(self.path).replace("\\", "/")

    def _bind_remote_artifact(
        self,
        *,
        teamspace: Teamspace,
        experiment_name: str,
        remote_path: str,
        client: LitRestClient | None = None,
        cloud_account: str | None = None,
    ) -> None:
        """Bind remote artifact download behavior to this file wrapper."""
        api = ArtifactsApi(client=client or LitRestClient(max_retries=5))
        full_remote_path = f"experiments/{experiment_name}/{remote_path}"
        self.name = remote_path
        self._download_fn = lambda path: api.download_file(
            teamspace=teamspace,
            remote_path=full_remote_path,
            local_path=path,
            cloud_account=cloud_account,
        )

    def _log_artifact(
        self,
        *,
        teamspace: Teamspace,
        metrics_store: Any,
        experiment_name: str,
        client: LitRestClient | None = None,
        remote_path: str | None = None,
    ) -> str:
        """Upload this file as an experiment artifact and bind remote access."""
        upload_path = self._get_upload_path()
        display_path = self._artifact_display_path(remote_path)
        api = ArtifactsApi(client=client or LitRestClient(max_retries=5))
        api.upload_experiment_file_artifact(
            teamspace=teamspace,
            metrics_store=metrics_store,
            experiment_name=experiment_name,
            file_path=upload_path,
            remote_path=display_path,
        )
        self._cleanup()
        cloud_account = getattr(metrics_store, "cluster_id", None)
        self._bind_remote_artifact(
            teamspace=teamspace,
            experiment_name=experiment_name,
            remote_path=display_path,
            client=api.client,
            cloud_account=cloud_account if isinstance(cloud_account, str) else None,
        )
        return display_path

    @property
    def _media_type(self) -> MediaType:
        return MediaType.FILE

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({self.path!r})"

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, File):
            return NotImplemented
        return type(self) is type(other) and self.path == other.path

    def __hash__(self) -> int:  # noqa: D105
        return hash((type(self), self.path))


class Image(File):
    """Represents an image to be logged.

    Can take a file path (str) or a Python object (PIL Image, numpy array,
    or torch Tensor) and will render it to a temporary file for upload.

    Args:
        data: The image data - a file path string, PIL Image, numpy array, or torch Tensor.
        format: Image format for rendering objects to disk (default: "png").
        description: Optional human-readable description of the image.
    """

    def __init__(self, data: Any, format: str = "png", description: str = "") -> None:  # noqa: A002
        self._data = data
        self._format = format
        if isinstance(data, str):
            super().__init__(data, description=description)
        else:
            super().__init__("", description=description)

    def _get_upload_path(self) -> str:
        if isinstance(self._data, str):
            return super()._get_upload_path()
        return self._render_to_temp()

    def _render_to_temp(self) -> str:
        """Render the image data to a temporary file."""
        suffix = f".{self._format.lower()}"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self._temp_path = path

        data = self._data
        img = None

        # Handle torch.Tensor -> numpy
        try:
            import torch

            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except ImportError:
            pass

        # Handle numpy array
        try:
            np = import_module("numpy")

            if isinstance(data, np.ndarray):
                pil_image = import_module("PIL.Image")

                if data.dtype != np.uint8:
                    data = (data * 255).astype(np.uint8) if data.max() <= 1.0 else data.astype(np.uint8)

                if data.ndim == 2:
                    img = pil_image.fromarray(data)
                elif data.ndim == 3:
                    # Handle CHW -> HWC format
                    if data.shape[0] in (1, 3, 4) and data.shape[2] not in (1, 3, 4):
                        data = data.transpose(1, 2, 0)
                    if data.shape[2] == 1:
                        data = data.squeeze(2)
                    img = pil_image.fromarray(data)
                else:
                    raise ValueError(f"Unsupported array shape for image: {data.shape}")

        except ImportError:
            pass

        # Handle PIL Image
        try:
            pil_image = import_module("PIL.Image")

            if isinstance(data, pil_image.Image):
                img = data

        except ImportError:
            pass

        # if valid image type was passed, save it and return
        if img is not None:
            img.save(self._temp_path)
            return self._temp_path

        raise TypeError(f"Unsupported image type: {type(data).__name__}")

    @property
    @override
    def _media_type(self) -> MediaType:
        return MediaType.IMAGE


class Text(File):
    """Represents text content to be logged.

    Takes a string and writes it to a temporary file for upload.

    Args:
        content: The text string to log.
        description: Optional human-readable description of the text.
    """

    def __init__(self, content: str, description: str = "") -> None:
        self._content = content
        super().__init__("", description=description)

    def _get_upload_path(self) -> str:
        if self.path and os.path.exists(self.path):
            return super()._get_upload_path()
        return self._render_to_temp()

    def _render_to_temp(self) -> str:
        """Write text content to a temporary file."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        self._temp_path = path
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._content)
        self.path = path
        return path

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}('{self.path}')"

    @property
    @override
    def _media_type(self) -> MediaType:
        return MediaType.TEXT


class Model(File):
    """Represents a model to be logged.

    Can take either a Python model object or a local path to a pre-saved model
    artifact. Uploads are handled through litmodels rather than the artifact API.

    Args:
        data: Python model object or path to a pre-saved model file/directory.
        version: Optional model version. Defaults to ``"latest"``.
        metadata: Optional metadata to associate with the model upload.
        staging_dir: Optional local staging directory for object-based uploads.
        description: Optional human-readable description of the model.
    """

    def __init__(
        self,
        data: Any,
        version: str | None = None,
        metadata: dict[str, str] | None = None,
        staging_dir: str | None = None,
        description: str = "",
        _kind: str | None = None,
    ) -> None:
        self._data = data
        self.version = version or "latest"
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
            return self.path
        return super()._get_upload_path()

    def _registry_name(self, experiment_name: str, teamspace: Teamspace) -> str:
        """Resolve the litmodels registry name for this model."""
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
        cloud_account: str | None = None,
        verbose: bool = False,
    ) -> str:
        """Upload this model to litmodels and return its registry name."""
        model_name = self._registry_name(experiment_name, teamspace)

        if self._model_kind == "artifact":
            upload_model(
                name=model_name,
                model=self._get_upload_path(),
                verbose=False,
                progress_bar=verbose,
                cloud_account=cloud_account,
                metadata=self.metadata,
            )
        else:
            save_model(
                name=model_name,
                model=self._data,
                staging_dir=self.staging_dir,
                verbose=False,
                progress_bar=verbose,
                cloud_account=cloud_account,
                metadata=self.metadata,
            )

        self._cleanup()
        return model_name

    def load(self, staging_dir: str | None = None) -> Any:
        """Load a remote model object via litmodels."""
        if self._load_fn is None:
            raise RuntimeError("Model has no remote load context. It must be uploaded to an experiment first.")
        return self._load_fn(staging_dir)
