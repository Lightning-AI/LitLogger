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
"""Legacy experiment helpers and backwards-compatible APIs."""

import mimetypes
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, cast

from lightning_sdk import Teamspace
from typing_extensions import Self

from litlogger.api.media_api import MediaApi
from litlogger.media import File
from litlogger.media import Model as MediaModel
from litlogger.series import Series
from litlogger.types import MediaType

if TYPE_CHECKING:
    from litlogger.printer import Printer, RunStats

_deprecation_warnings_issued: set[str] = set()


def _warn_deprecated(method: str, message: str) -> None:
    """Issue a deprecation warning for a legacy method, at most once per method."""
    if method not in _deprecation_warnings_issued:
        _deprecation_warnings_issued.add(method)
        warnings.warn(message, DeprecationWarning, stacklevel=3)


class _MetadataValue(str):
    """String-like metadata value that rejects time-series operations."""

    __slots__ = ("_key",)
    _key: str

    def __new__(cls, key: str, value: str) -> Self:
        obj = super().__new__(cls, value)
        obj._key = key
        return obj

    def append(self, value: object, step: int | None = None) -> None:
        raise KeyError(f"Key {self._key!r} is already used as metadata. Cannot append metric values.")

    def extend(self, values: object, start_step: int | None = None) -> None:
        raise KeyError(f"Key {self._key!r} is already used as metadata. Cannot append metric values.")


class LegacyExperiment:
    """Legacy method-based API for backwards compatibility.

    Provides deprecated methods like log_metrics(), log_file(), log_model(), etc.
    These are inherited by Experiment so existing code continues to work.
    """

    if TYPE_CHECKING:
        name: str
        _teamspace: Teamspace
        _printer: Printer
        _stats: RunStats
        _media_api: MediaApi
        _metrics_store: Any

        def __getitem__(self, key: str) -> Series | str | File: ...  # noqa: D105

        def __setitem__(self, key: str, value: str | File) -> None: ...  # noqa: D105

        def update(self, data: dict[str, str | float | int | File | list[float | int | File]]) -> None: ...

        def _upload_media(
            self,
            name: str,
            file_path: str,
            media_type: MediaType,
            step: int | None = None,
            epoch: int | None = None,
            caption: str | None = None,
        ) -> None: ...

    def log_metrics(self, metrics: dict[str, float] | None = None, step: int | None = None, **kwargs: float) -> None:
        """Log metrics to the experiment with background uploading.

        .. deprecated::
            Use ``experiment["key"].append(value, step=step)`` instead.

        Metrics are buffered locally and uploaded to the cloud in batches to optimize performance.
        The batch is sent when either 1 second has passed or 1000 values have been logged.

        Args:
            metrics: Dictionary mapping metric names to numeric values. Example: {"loss": 0.5, "accuracy": 0.95}.
            step: Optional step number for this data point (e.g., training step, epoch).
                If None and store_step=True, no step is recorded.
            kwargs: Additional metric values. Can be used to provide metrics more natural.
                Example: loss=0.5, accuracy: 0.95.

        Raises:
            RuntimeError: If the background thread encountered an error.
        """
        _warn_deprecated(
            "log_metrics", 'log_metrics() is deprecated. Use experiment["key"].append(value, step=step) instead.'
        )
        if metrics is None:
            metrics = {}
        metrics.update(kwargs)
        for name, value in metrics.items():
            cast(Series, self[name]).append(float(value), step=step)

    def log_metadata(self, metadata: dict[str, str] | None = None, **kwargs: str) -> None:
        """Add or update metadata tags on the experiment.

        .. deprecated::
            Use ``experiment["key"] = "value"`` instead.

        Merges the provided key-value pairs into the experiment's existing metadata
        and pushes the update to the cloud immediately.

        Args:
            metadata: Dictionary of metadata key-value pairs to add or update.
                Example: {"optimizer": "adam", "lr": "0.001"}.
            **kwargs: Additional metadata as keyword arguments.
                Example: optimizer="adam", lr="0.001".
        """
        _warn_deprecated("log_metadata", 'log_metadata() is deprecated. Use experiment["key"] = "value" instead.')
        self.update({**(metadata or {}), **kwargs})

    def log_metrics_batch(self, metrics: dict[str, list[dict[str, float]]]) -> None:
        """Log a batch of metrics through the background queue.

        .. deprecated::
            Use ``experiment["key"].extend(values, start_step=step)`` instead.

        This method converts the batch format to Metrics objects and pushes them
        through the background queue, which handles batching and chunking to respect
        API limits.

        Args:
            metrics: Dictionary mapping metric names to lists of dicts with "step" and "value" keys.

        Example::

            {
                "loss": [
                    {"step": 0, "value": 1.0},
                    {"step": 1, "value": 0.5},
                ],
                "accuracy": [
                    {"step": 0, "value": 0.6},
                    {"step": 1, "value": 0.8},
                ],
            }

        Raises:
            RuntimeError: If the background thread encountered an error.
        """
        _warn_deprecated(
            "log_metrics_batch",
            'log_metrics_batch() is deprecated. Use experiment.update() or experiment["key"].extend() instead.',
        )
        if not metrics:
            return
        self.update({name: [v["value"] for v in values] for name, values in metrics.items()})

    def log_file(
        self,
        path: str,
        remote_path: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Upload a file artifact to the cloud for this experiment.

        .. deprecated::
            Use ``experiment["key"] = File("path")`` or
            ``experiment["key"].append(File("path"))`` instead.

        The file is uploaded to cloud storage and registered with the experiment,
        making it visible in the artifacts view and accessible via get_file().

        Args:
            path: Path to the local file to upload. Can be absolute or relative.
            remote_path: Path relative to experiment root for storage and display.
                If None, uses the path relative to cwd if under cwd, otherwise basename.
                Example: remote_path="images/0.png" will store and display as "images/0.png".
            verbose: Whether to print a confirmation message after upload. Defaults to True.
        """
        _warn_deprecated("log_file", 'log_file() is deprecated. Use experiment["key"] = File("path") instead.')
        if remote_path is None:
            try:
                rel = os.path.relpath(path)
            except ValueError:
                rel = None
            remote_path = rel if rel is not None and not rel.startswith("..") else os.path.basename(path)
            remote_path = remote_path.replace("\\", "/")
        self[remote_path] = File(path)
        if verbose:
            self._printer.artifact_logged(path, remote_path)

    def log_files(
        self,
        paths: list[str],
        remote_paths: list[str] | None = None,
        max_workers: int = 10,
    ) -> None:
        """Upload multiple file artifacts to the cloud in parallel.

        .. deprecated::
            Use ``experiment["key"] = File("path")`` for each file instead.

        This is more efficient than calling log_file() multiple times when you have
        many files, as it handles them in parallel.

        Args:
            paths: List of paths to local files to upload.
            remote_paths: Optional list of remote paths, one for each file in paths.
                If provided, must have same length as paths.
                If None, each file uses its default remote path (relative to cwd or basename).
            max_workers: Maximum number of concurrent uploads. Defaults to 10.
        """
        _warn_deprecated(
            "log_files", 'log_files() is deprecated. Use experiment["key"] = File("path") for each file instead.'
        )
        if remote_paths is None:
            remote_paths = [None] * len(paths)  # type: ignore[list-item]

        if len(remote_paths) != len(paths):
            raise ValueError(f"remote_paths length ({len(remote_paths)}) must match paths length ({len(paths)})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.log_file, path, remote, verbose=False): path
                for path, remote in zip(paths, remote_paths, strict=False)
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self._printer.artifact_failed(path, str(e))

    def get_file(self, path: str, remote_path: str | None = None, verbose: bool = True) -> str:
        """Download a file artifact from the cloud for this experiment.

        .. deprecated::
            Use ``experiment["key"]`` to access files instead.

        The file is downloaded from cloud storage (previously uploaded via log_file)
        and saved to the specified local path.

        Args:
            path: Path where the file should be saved locally. Parent directories are created if needed.
            remote_path: Path relative to experiment root where the file is stored.
                If None, uses the path relative to cwd if under cwd, otherwise basename.
                Must match the remote_path used during log_file() for correct resolution.
            verbose: Whether to print a confirmation message after download. Defaults to True.

        Returns:
            str: The local path where the file was saved (same as the input path).
        """
        _warn_deprecated("get_file", 'get_file() is deprecated. Use experiment["key"].save(path) instead.')
        if remote_path is None:
            try:
                rel = os.path.relpath(path)
            except ValueError:
                rel = None
            remote_path = rel if rel is not None and not rel.startswith("..") else os.path.basename(path)
            remote_path = remote_path.replace("\\", "/")
        return cast(File, self[remote_path]).save(path)

    def log_model_artifact(self, path: str, verbose: bool = False, version: str | None = None) -> None:
        """Upload a model file or directory to cloud storage using litmodels.

        .. deprecated::
            Use ``experiment["key"] = File("path")`` instead.

        This uploads raw model files (e.g., weights.pt, checkpoint.ckpt) or entire directories
        to the litmodels registry. Use this when you have pre-saved model files.

        For saving model objects directly, use log_model() instead.

        Args:
            path: Path to the local model file or directory to upload.
            verbose: Whether to show progress bar during upload. Defaults to False.
            version: Optional version string for the model.
        """
        _warn_deprecated(
            "log_model_artifact", 'log_model_artifact() is deprecated. Use experiment["key"] = File("path") instead.'
        )
        model_artifact = MediaModel(path, version=version or "latest")
        model_artifact._log_model(
            experiment_name=self.name,
            teamspace=self._teamspace,
            cloud_account=getattr(self._metrics_store, "cluster_id", None),
        )
        self._stats.models_logged += 1
        if verbose:
            self._printer.artifact_logged(path, f"model artifact: {path}")

    def get_model_artifact(self, path: str, verbose: bool = False, version: str | None = None) -> str:
        """Download a model artifact file or directory from cloud storage using litmodels.

        .. deprecated::
            Use ``experiment["key"]`` to access model artifacts instead.

        This downloads raw model files or directories that were previously uploaded
        via log_model_artifact(). The files are saved to the specified local path.

        Args:
            path: Path where the model should be saved locally. Directories are created if needed.
            verbose: Whether to show progress bar during download. Defaults to False.
            version: Optional version string for the model.

        Returns:
            str: The local path where the model was saved (same as the input path).
        """
        _warn_deprecated(
            "get_model_artifact",
            'get_model_artifact() is deprecated. Use experiment["key"] to access model artifacts instead.',
        )
        model_artifact = MediaModel(path, version=version or "latest")
        model_artifact._bind_remote_model(
            key=path,
            model_name=model_artifact._registry_name(self.name, self._teamspace),
        )
        result = model_artifact.save(path)
        if verbose:
            self._printer.artifact_retrieved(path)
        return result

    def log_model(
        self,
        model: Any,
        staging_dir: str | None = None,
        verbose: bool = False,
        version: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Save and upload a model object to cloud storage using litmodels.

        .. deprecated::
            Use ``experiment["key"] = File("path")`` instead.

        This saves a live model object (e.g., PyTorch module, LightningModule) to disk
        using framework-specific serialization, then uploads it to the litmodels registry.

        For uploading pre-saved model files, use log_model_artifact() instead.

        Args:
            model: The model object to save and upload (e.g., torch.nn.Module, LightningModule).
            staging_dir: Optional local directory for staging the model before upload. If None, uses a temp directory.
            verbose: Whether to show progress bar during upload. Defaults to False.
            version: Optional version string for the model.
            metadata: Optional metadata dictionary to store with the model (e.g., hyperparameters, metrics).

        Returns:
            str: Information about the uploaded model (details from litmodels).
        """
        _warn_deprecated("log_model", 'log_model() is deprecated. Use experiment["key"] = File("path") instead.')
        model_obj = MediaModel(model, version=version or self.name, metadata=metadata, staging_dir=staging_dir)
        model_name = model_obj._log_model(
            experiment_name=self.name,
            teamspace=self._teamspace,
            cloud_account=getattr(self._metrics_store, "cluster_id", None),
        )
        self._stats.models_logged += 1
        if verbose:
            self._printer.print_success("Logged model object")
        return model_name

    def get_model(self, staging_dir: str | None = None, verbose: bool = False, version: str | None = None) -> Any:
        """Get a model object using litmodels load_model.

        .. deprecated::
            Use ``experiment["key"]`` to access models instead.

        Args:
            staging_dir: Optional directory where the model will be downloaded.
            verbose: Whether to show progress bar.
            version: Optional version string for the model.

        Returns:
            The loaded model object.
        """
        _warn_deprecated("get_model", 'get_model() is deprecated. Use experiment["key"] to access models instead.')
        model_obj = MediaModel(object(), version=version or "latest", staging_dir=staging_dir)
        model_obj._bind_remote_model(
            key=self.name,
            model_name=model_obj._registry_name(self.name, self._teamspace),
        )
        result = model_obj.load(staging_dir)
        if verbose:
            self._printer.print_success("Retrieved model object")
        return result

    def log_media(
        self,
        name: str,
        path: str,
        kind: MediaType | None = None,
        step: int | None = None,
        epoch: int | None = None,
        caption: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Upload a media file (image, text, etc.) to the experiment.

        .. deprecated::
            Use ``experiment["key"].append(Image(...))`` or
            ``experiment["key"].append(Text(...))`` instead.

        Args:
            name: Name of the media.
            path: Local path to the media file.
            kind: Type of media (MediaType.IMAGE or MediaType.TEXT).
                  If None, attempts to guess from file extension or mime type.
            step: Optional training step.
            epoch: Optional training epoch.
            caption: Optional caption for the media.
            verbose: Whether to print a confirmation message after upload.

        Raises:
            ValueError: If the file type cannot be determined or is not supported.
            FileNotFoundError: If the file does not exist.
        """
        _warn_deprecated(
            "log_media",
            'log_media() is deprecated. Use experiment["key"].append(Image(...)) '
            'or experiment["key"].append(Text(...)) instead.',
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Media file not found: {path}")

        resolved_kind: MediaType | None = kind

        if resolved_kind is None:
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type:
                if mime_type.startswith("image/"):
                    resolved_kind = MediaType.IMAGE
                elif mime_type.startswith("text/"):
                    resolved_kind = MediaType.TEXT

        if resolved_kind is None:
            raise ValueError(f"Unsupported media type for file: {path}")

        self._upload_media(name, path, resolved_kind, step=step, epoch=epoch, caption=caption)
        if verbose:
            self._printer.media_logged(path, step)
