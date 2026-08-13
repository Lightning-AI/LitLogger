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

import contextlib
import os
import tempfile
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Mapping

from lightning_sdk import Teamspace
from lightning_sdk.lightning_cloud.openapi import V1MediaType
from typing_extensions import override

from litlogger.models import download_model, load_model, save_model, upload_model
from litlogger.primitives import (
    SERIES_NAME_RE,
    RestoredFiles,
    _enqueue_write,
    _to_v1_media_type,
    model_version_sort_key,
    sanitize_model_key,
)
from litlogger.types import MediaType

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


def _wrap_media_file(media_name: str, media_type: V1MediaType) -> "File":
    """Build the media wrapper matching a listed record's wire type."""
    if media_type == V1MediaType.IMAGE:
        return Image(media_name)
    if media_type == V1MediaType.TEXT:
        text = Text("")
        text.path = media_name
        return text
    if media_type == V1MediaType.VIDEO:
        return Video(media_name)
    return File(media_name)


def _artifact_download_fn(session: "ExperimentSession", key: str) -> Callable[[str], str]:
    """Build a lazy artifact download closure for a restored file."""

    def _download(path: str) -> str:
        file = File(path)
        file._bind_remote(
            session,
            remote_path=key,
            cloud_account=getattr(session.metrics_store, "cluster_id", None),
        )
        return file.save(path)

    return _download


def _media_download_fn(
    session: "ExperimentSession", storage_path: str, cloud_account: str | None = None
) -> Callable[[str], str]:
    """Build a lazy media download closure for a restored file."""

    def _download(path: str) -> str:
        session.teamspace.download_file(storage_path, file_path=path, cloud_account=cloud_account)
        return path

    return _download


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
        # Placement stamped by the experiment at dispatch time: the experiment
        # key, and for series elements the index (and step) within the series.
        self._log_key: str | None = None
        self._series_index: int | None = None
        self._series_step: int | None = None

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

    def _remote_path(self) -> str | None:
        """Resolve the artifact remote path from the stamped placement.

        Static values upload under the experiment key; series elements under
        ``{key}/{index}``. Without a stamped key (standalone ``log()``), the
        path is derived from the local file path at write time.
        """
        if self._log_key is None:
            return None
        if self._series_index is None:
            return self._log_key
        return f"{self._log_key}/{self._series_index}"

    def _bind_remote(
        self,
        session: "ExperimentSession",
        *,
        remote_path: str,
        cloud_account: str | None = None,
    ) -> None:
        """Bind remote artifact download behavior to this file wrapper."""
        api = session.artifacts_api
        teamspace = session.teamspace
        full_remote_path = f"experiments/{session.experiment_name}/{remote_path}"
        self.name = remote_path
        self._download_fn = lambda path: api.download_file(
            teamspace=teamspace,
            remote_path=full_remote_path,
            local_path=path,
            cloud_account=cloud_account,
        )

    def _upload_artifact(self, session: "ExperimentSession", remote_path: str | None = None) -> str:
        """Upload this file as an experiment artifact and bind remote access."""
        upload_path = self._get_upload_path()
        display_path = self._artifact_display_path(remote_path)
        session.artifacts_api.upload_experiment_file_artifact(
            teamspace=session.teamspace,
            metrics_store=session.metrics_store,
            experiment_name=session.experiment_name,
            file_path=upload_path,
            remote_path=display_path,
        )
        self._cleanup()
        cloud_account = getattr(session.metrics_store, "cluster_id", None)
        self._bind_remote(
            session,
            remote_path=display_path,
            cloud_account=cloud_account if isinstance(cloud_account, str) else None,
        )
        return display_path

    def log(self, session: "ExperimentSession") -> None:
        """Upload this file as an experiment artifact now, in the caller's thread."""
        self._upload_artifact(session, remote_path=self._remote_path())
        session.stats.artifacts_logged += 1

    def enqueue(self, session: "ExperimentSession") -> None:
        """Hand this file to the background pipeline for asynchronous upload."""
        _enqueue_write(self, session)

    @staticmethod
    def _restore_all(session: "ExperimentSession", existing_key_types: Mapping[str, str]) -> RestoredFiles:
        """Rebuild static files and file series from the artifact listing.

        ``existing_key_types`` is a snapshot of keys claimed by earlier restore
        passes; keys already claimed by a different kind are skipped, and this
        pass tracks its own claims internally.
        """
        restored = RestoredFiles(statics={}, series={})
        claimed: dict[str, str] = dict(existing_key_types)

        artifacts = getattr(session.metrics_store, "artifacts", None) or []
        with contextlib.suppress(AttributeError):
            listed = session.artifacts_api.list_experiment_artifacts(session.teamspace.id, session.metrics_store.id)
            if listed is not None:
                artifacts = listed

        series_entries: dict[str, list[tuple[int, File]]] = {}
        for artifact in artifacts:
            name = artifact.path if hasattr(artifact, "path") else str(artifact)
            wrapped = File(name)
            wrapped.name = name
            wrapped._download_fn = _artifact_download_fn(session, name)

            match = SERIES_NAME_RE.match(name)
            if match:
                key = match.group("key")
                index = int(match.group("index"))
                if key in claimed and claimed[key] != "file_series":
                    continue
                claimed[key] = "file_series"
                series_entries.setdefault(key, []).append((index, wrapped))
                continue

            if name in claimed:
                continue
            claimed[name] = "static_file"
            restored.statics[name] = wrapped

        for key, file_entries in series_entries.items():
            restored.series[key] = [value for _, value in sorted(file_entries)]
        return restored

    @staticmethod
    def _restore_media(session: "ExperimentSession", existing_key_types: Mapping[str, str]) -> RestoredFiles:
        """Rebuild static media and media series from the media listing.

        A name with one direct record is a static file; several records under
        the same name form a series ordered by step (falling back to listing
        position). ``{key}/{index}`` names reconstruct indexed series like the
        artifact pass.
        """
        restored = RestoredFiles(statics={}, series={})
        claimed: dict[str, str] = dict(existing_key_types)

        with contextlib.suppress(AttributeError):
            media_items = session.media_api.list_media(session.teamspace.id, session.metrics_store.id) or []

            series_entries: dict[str, list[tuple[int, File]]] = {}
            direct_entries: dict[str, list[tuple[int | None, int, File]]] = {}
            for position, media in enumerate(media_items):
                name = media.name or media.storage_path or media.id
                storage_path = media.storage_path or name
                wrapped = _wrap_media_file(name, media.media_type)
                wrapped.name = name
                wrapped._download_fn = _media_download_fn(session, storage_path, media.cluster_id)

                match = SERIES_NAME_RE.match(name)
                if match:
                    key = match.group("key")
                    index = int(match.group("index"))
                    if key in claimed and claimed[key] != "file_series":
                        continue
                    claimed[key] = "file_series"
                    series_entries.setdefault(key, []).append((index, wrapped))
                    continue

                direct_entries.setdefault(name, []).append((getattr(media, "step", None), position, wrapped))

            for name, media_entries in direct_entries.items():
                if name in claimed:
                    continue
                if len(media_entries) == 1:
                    claimed[name] = "static_file"
                    restored.statics[name] = media_entries[0][2]
                    continue

                claimed[name] = "file_series"
                series_values = series_entries.setdefault(name, [])
                for step, position, wrapped in media_entries:
                    sort_index = step if isinstance(step, int) else position
                    series_values.append((sort_index, wrapped))

            for key, file_entries in series_entries.items():
                restored.series[key] = [value for _, value in sorted(file_entries)]
        return restored

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


class _MediaFile(File):
    """Base for rendered media (images, videos, text) uploaded through the media API."""

    @override
    def log(self, session: "ExperimentSession") -> None:
        """Upload this media now, in the caller's thread.

        Media uploads share one remote name per key: series elements are
        differentiated by their step, not by an indexed path.
        """
        name = self._log_key if self._log_key is not None else (self.name or self._artifact_display_path(None))
        upload_path = self._get_upload_path()
        session.media_api.upload_media(
            experiment_id=session.metrics_store.id,
            teamspace=session.teamspace,
            file_path=upload_path,
            name=name,
            media_type=_to_v1_media_type(self._media_type),
            step=self._series_step,
        )
        self.name = name
        self._cleanup()
        session.stats.media_logged += 1


class Image(_MediaFile):
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


class Video(_MediaFile):
    DEFAULT_FPS = 24

    """Represents a video to be logged.

    Accepts:
    - a file path string
    - a MoviePy clip
    - a numpy array / torch tensor of frames

    Supported array shapes:
    - (T, H, W)            grayscale frames
    - (T, H, W, C)         frames in HWC format
    - (T, C, H, W)         frames in CHW format, where C is 1/3/4

    Args:
        data: Video data or a path to a video file.
        format: Output container/extension, default "mp4".
        description: Optional description.
        fps: Frames per second used when rendering arrays or clips that do
            not already carry fps metadata.
    """

    def __init__(
        self,
        data: Any,
        format: str = "mp4",  # noqa: A002
        description: str = "",
        fps: float | None = None,
    ) -> None:
        self._data = data
        self._format = format
        self._fps = fps
        self._temp_path: str | None = None

        if isinstance(data, str):
            super().__init__(data, description=description)
        else:
            super().__init__("", description=description)

    def _get_upload_path(self) -> str:
        if isinstance(self._data, str):
            return super()._get_upload_path()
        return self._render_to_temp()

    def _render_to_temp(self) -> str:
        suffix = f".{self._format.lower()}"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self._temp_path = path

        data = self._data

        # torch.Tensor -> numpy
        try:
            import torch

            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except ImportError:
            pass

        # 1) MoviePy clip
        clip = self._maybe_as_moviepy_clip(data)
        if clip is not None:
            fps = self._fps or getattr(clip, "fps", None) or self.DEFAULT_FPS
            self._write_moviepy_clip(clip, path, fps)
            return path

        # 2) numpy array of frames
        try:
            np = import_module("numpy")

            if isinstance(data, np.ndarray):
                clip = self._moviepy_clip_from_array(data, fps=self._fps or self.DEFAULT_FPS)
                self._write_moviepy_clip(clip, path, self._fps or self.DEFAULT_FPS)
                return path

        except ImportError:
            pass

        raise TypeError(f"Unsupported video type: {type(data).__name__}")

    def _maybe_as_moviepy_clip(self, data: Any) -> None | Any:
        """Return data if it looks like a MoviePy clip, else None."""
        try:
            # MoviePy 2.x layout
            video_clip_mod = import_module("moviepy.video.VideoClip")
            video_clip = video_clip_mod.VideoClip
            if isinstance(data, video_clip):
                return data
        except Exception:
            pass

        try:
            # Older common import path
            editor_mod = import_module("moviepy.editor")
            video_clip = editor_mod.VideoClip
            if isinstance(data, video_clip):
                return data
        except Exception:
            pass

        return None

    def _moviepy_clip_from_array(self, data: Any, fps: float) -> Any:
        np = import_module("numpy")

        if data.dtype != np.uint8:
            if np.issubdtype(data.dtype, np.floating):
                # common case: [0,1] floats
                if data.size and data.max() <= 1.0:
                    data = (data * 255).clip(0, 255).astype(np.uint8)
                else:
                    data = data.clip(0, 255).astype(np.uint8)
            else:
                data = data.clip(0, 255).astype(np.uint8)

        if data.ndim not in (3, 4):
            raise ValueError(
                f"Unsupported array shape for video: {data.shape}. Expected (T,H,W), (T,H,W,C), or (T,C,H,W)."
            )

        # (T, H, W) -> grayscale => expand to (T, H, W, 1)
        if data.ndim == 3:
            data = data[..., None]

        # Handle TCHW -> THWC when channel axis is second
        if data.ndim == 4 and data.shape[1] in (1, 3, 4) and data.shape[-1] not in (1, 3, 4):
            data = data.transpose(0, 2, 3, 1)

        if data.shape[-1] == 1:
            # MoviePy ImageSequenceClip works best with 2D or RGB arrays;
            # convert grayscale to RGB for consistency.
            data = np.repeat(data, 3, axis=-1)

        if data.shape[-1] not in (3, 4):
            raise ValueError(f"Unsupported channel count for video frames: {data.shape[-1]}")

        # Drop alpha for now unless you explicitly want to preserve/use masks.
        if data.shape[-1] == 4:
            data = data[..., :3]

        try:
            # MoviePy 2.x
            image_sequence_clip = import_module("moviepy.video.io.ImageSequenceClip").ImageSequenceClip
        except Exception:
            # Older common path
            image_sequence_clip = import_module("moviepy.editor").ImageSequenceClip

        # list(...) avoids some ndarray edge cases in callers and matches
        # common usage for frame sequences.
        return image_sequence_clip(list(data), fps=fps)

    def _write_moviepy_clip(self, clip: Any, path: str, fps: float) -> None:
        # For MP4, libx264 is the usual sensible default.
        kwargs: dict[str, Any] = {
            "fps": fps,
            "logger": None,
        }

        ext = os.path.splitext(path)[1].lower()
        if ext == ".mp4":
            kwargs["codec"] = "libx264"
            # Better for browser progressive playback.
            kwargs["ffmpeg_params"] = ["-movflags", "+faststart"]

        clip.write_videofile(path, **kwargs)

    @property
    @override
    def _media_type(self) -> MediaType:
        return MediaType.VIDEO


class Text(_MediaFile):
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
        complete version), or None when nothing matches — including on any
        lookup error, which callers negative-cache.
        """
        model_key = sanitize_model_key(key)
        try:
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
        except Exception:
            return None

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
        if self._load_fn is None:
            raise RuntimeError("Model has no remote load context. It must be uploaded to an experiment first.")
        return self._load_fn(staging_dir)
