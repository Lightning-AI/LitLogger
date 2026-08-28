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
"""File artifact primitive and restore helpers."""

import copy
import math
import os
import tempfile
from typing import TYPE_CHECKING, Callable, Mapping

from lightning_sdk.lightning_cloud.openapi import V1MediaType

from litlogger.primitives._utils import (
    SERIES_NAME_RE,
    RestoredFiles,
    parse_storage_name,
    series_storage_name,
    static_storage_name,
)
from litlogger.primitives.primitive import WritePlacement, _enqueue_write
from litlogger.types import MediaType

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


def _wrap_media_file(media_name: str, media_type: V1MediaType) -> "File":
    """Build the media wrapper matching a listed record's wire type."""
    from litlogger.primitives.image import Image
    from litlogger.primitives.text import Text
    from litlogger.primitives.video import Video

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


def _media_sort_index(step: object, position: int) -> float:
    """Return a finite media x-coordinate or the stable listing position."""
    if isinstance(step, str | int | float):
        try:
            numeric_step = float(step)
        except ValueError:
            return float(position)
        if math.isfinite(numeric_step):
            return numeric_step
    return float(position)


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
        self._prepared_upload_path: str | None = None
        self._download_fn: Callable[[str], str] | None = None
        self._read_barrier: Callable[[], None] | None = None

    def _get_upload_path(self) -> str:
        """Get a stable path for upload.

        Creates a hardlink to a temp location so the original file can be
        safely modified or deleted while a background upload is in progress.
        Falls back to a copy if hardlinking is not supported, or returns the
        original path if the file doesn't exist yet.
        """
        if self._prepared_upload_path is not None:
            return self._prepared_upload_path
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
            self._temp_path = tmp
            shutil.copy2(self.path, tmp)
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
        self._prepared_upload_path = None

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
        if self._read_barrier is not None:
            self._read_barrier()
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

    def _remote_path(self, placement: WritePlacement | None) -> str | None:
        """Resolve an unambiguous artifact storage path."""
        if placement is None:
            return None
        if placement.index is None:
            return static_storage_name(placement.key)
        return series_storage_name(placement.key, placement.index)

    def _bind_remote(
        self,
        session: "ExperimentSession",
        *,
        remote_path: str,
        display_name: str | None = None,
        cloud_account: str | None = None,
    ) -> None:
        """Bind remote artifact download behavior to this file wrapper."""
        api = session.artifacts_api
        teamspace = session.teamspace
        full_remote_path = f"experiments/{session.experiment_name}/{remote_path}"
        self.name = display_name if display_name is not None else remote_path
        self._download_fn = lambda path: api.download_file(
            teamspace=teamspace,
            remote_path=full_remote_path,
            local_path=path,
            cloud_account=cloud_account,
        )

    def _upload_artifact(
        self,
        session: "ExperimentSession",
        remote_path: str | None = None,
        display_name: str | None = None,
    ) -> str:
        """Upload this file as an experiment artifact and bind remote access."""
        try:
            upload_path = self._get_upload_path()
            display_path = self._artifact_display_path(remote_path)
            session.artifacts_api.upload_experiment_file_artifact(
                teamspace=session.teamspace,
                metrics_store=session.metrics_store,
                experiment_name=session.experiment_name,
                file_path=upload_path,
                remote_path=display_path,
            )
        finally:
            self._cleanup()
        cloud_account = getattr(session.metrics_store, "cluster_id", None)
        self._bind_remote(
            session,
            remote_path=display_path,
            display_name=display_name,
            cloud_account=cloud_account if isinstance(cloud_account, str) else None,
        )
        return display_path

    def log(self, session: "ExperimentSession", placement: WritePlacement | None = None) -> None:
        """Upload this file as an experiment artifact now, in the caller's thread."""
        self._upload_artifact(
            session,
            remote_path=self._remote_path(placement),
            display_name=placement.key if placement is not None else None,
        )
        session.stats.artifacts_logged += 1

    def enqueue(self, session: "ExperimentSession", placement: WritePlacement | None = None) -> None:
        """Snapshot this wrapper and submit an immutable background operation."""
        snapshot = copy.copy(self)
        snapshot._read_barrier = None
        snapshot._download_fn = None
        snapshot._prepared_upload_path = snapshot._get_upload_path()

        def _write(active_session: "ExperimentSession") -> None:
            snapshot.log(active_session, placement)
            self._adopt_remote_state(snapshot)

        try:
            _enqueue_write(_write, session)
        except Exception:
            snapshot._cleanup()
            raise
        self._read_barrier = session.flush

    def _adopt_remote_state(self, completed: "File") -> None:
        """Copy completed remote binding without copying queued placement state."""
        self.name = completed.name
        self._download_fn = completed._download_fn
        self._read_barrier = None

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
        listed = session.artifacts_api.list_experiment_artifacts(session.teamspace.id, session.metrics_store.id)
        if listed is not None:
            artifacts = listed

        series_entries: dict[str, list[tuple[int, File]]] = {}
        for artifact in artifacts:
            storage_name = artifact.path if hasattr(artifact, "path") else str(artifact)
            explicit = parse_storage_name(storage_name)
            display_name = explicit[1] if explicit is not None else storage_name
            wrapped = File(display_name)
            wrapped.name = display_name
            wrapped._download_fn = _artifact_download_fn(session, storage_name)

            if explicit is not None:
                kind, key, index = explicit
                if kind == "series":
                    if index is None:
                        continue
                    if key in claimed and claimed[key] != "file_series":
                        continue
                    claimed[key] = "file_series"
                    series_entries.setdefault(key, []).append((index, wrapped))
                elif key not in claimed:
                    claimed[key] = "static_file"
                    restored.statics[key] = wrapped
                continue

            # Compatibility for artifacts written before explicit placement
            # names were introduced. These names remain inherently ambiguous.
            match = SERIES_NAME_RE.match(storage_name)
            if match:
                key = match.group("key")
                index = int(match.group("index"))
                if key in claimed and claimed[key] != "file_series":
                    continue
                claimed[key] = "file_series"
                series_entries.setdefault(key, []).append((index, wrapped))
                continue

            if storage_name in claimed:
                continue
            claimed[storage_name] = "static_file"
            restored.statics[storage_name] = wrapped

        for key, file_entries in series_entries.items():
            restored.series[key] = [value for _, value in sorted(file_entries, key=lambda item: item[0])]
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

        media_items = session.media_api.list_media(session.teamspace.id, session.metrics_store.id) or []

        series_entries: dict[str, list[tuple[float, File]]] = {}
        direct_entries: dict[str, list[tuple[object, int, File]]] = {}
        for position, media in enumerate(media_items):
            wire_name = media.name or media.storage_path or media.id
            storage_path = media.storage_path or wire_name
            explicit = parse_storage_name(wire_name)
            display_name = explicit[1] if explicit is not None else wire_name
            wrapped = _wrap_media_file(display_name, media.media_type)
            wrapped.name = display_name
            wrapped._download_fn = _media_download_fn(session, storage_path, media.cluster_id)

            if explicit is not None:
                kind, key, index = explicit
                if kind == "static":
                    if key not in claimed:
                        claimed[key] = "static_file"
                        restored.statics[key] = wrapped
                    continue
                if key in claimed and claimed[key] != "file_series":
                    continue
                claimed[key] = "file_series"
                sort_index = _media_sort_index(getattr(media, "step", None), position)
                if index is not None:
                    sort_index = float(index)
                series_entries.setdefault(key, []).append((sort_index, wrapped))
                continue

            # Compatibility for indexed media names from older clients.
            match = SERIES_NAME_RE.match(wire_name)
            if match:
                key = match.group("key")
                index = int(match.group("index"))
                if key in claimed and claimed[key] != "file_series":
                    continue
                claimed[key] = "file_series"
                series_entries.setdefault(key, []).append((index, wrapped))
                continue

            direct_entries.setdefault(wire_name, []).append((getattr(media, "step", None), position, wrapped))

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
                series_values.append((_media_sort_index(step, position), wrapped))

        for key, file_entries in series_entries.items():
            restored.series[key] = [value for _, value in sorted(file_entries, key=lambda item: item[0])]
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
