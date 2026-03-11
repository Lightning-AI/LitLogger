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
"""Experiment abstraction for logging metrics and artifacts to Lightning.ai Cloud."""

import atexit
import contextlib
import mimetypes
import os
import signal
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import JoinableQueue
from threading import Event
from time import sleep
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, cast

from lightning_sdk import Teamspace
from lightning_sdk.lightning_cloud.openapi import V1MediaType

from litlogger.api.artifacts_api import ArtifactsApi
from litlogger.api.auth_api import AuthApi
from litlogger.api.media_api import MediaApi
from litlogger.api.metrics_api import MetricsApi
from litlogger.api.utils import _resolve_teamspace, build_experiment_url, get_accessible_url, get_guest_url
from litlogger.artifacts import Artifact, Model, ModelArtifact
from litlogger.background import _BackgroundThread
from litlogger.capture import rerun_and_record
from litlogger.media import File
from litlogger.printer import Printer, RunStats
from litlogger.series import Series
from litlogger.types import MediaType, Metrics, MetricValue, PhaseType

_deprecation_warnings_issued: set[str] = set()

if TYPE_CHECKING:
    from litlogger.printer import Printer, RunStats


def _warn_deprecated(method: str, message: str) -> None:
    """Issue a deprecation warning for a legacy method, at most once per method."""
    if method not in _deprecation_warnings_issued:
        _deprecation_warnings_issued.add(method)
        warnings.warn(message, DeprecationWarning, stacklevel=3)


class _MetadataValue(str):
    """String-like metadata value that rejects time-series operations."""

    def __new__(cls, key: str, value: str) -> "_MetadataValue":
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
        model_artifact = ModelArtifact(
            path=path,
            experiment_name=self.name,
            teamspace=self._teamspace,
            version=version or "latest",
            verbose=verbose,
        )
        model_artifact.log()
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
        model_artifact = ModelArtifact(
            path=path,
            experiment_name=self.name,
            teamspace=self._teamspace,
            version=version or "latest",
            verbose=verbose,
        )
        result = model_artifact.get()
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
        model_obj = Model(
            model=model,
            experiment_name=self.name,
            teamspace=self._teamspace,
            version=version or "latest",
            verbose=verbose,
            metadata=metadata,
            staging_dir=staging_dir,
        )
        model_obj.log()
        self._stats.models_logged += 1
        if verbose:
            self._printer.print_success("Logged model object")
        return model_obj.name

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
        model_obj = Model(
            model=None,  # Not needed for loading
            experiment_name=self.name,
            teamspace=self._teamspace,
            version=version or "latest",
            verbose=verbose,
            staging_dir=staging_dir,
        )
        result = model_obj.get()
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

        media_type = V1MediaType.UNSPECIFIED

        if kind is not None:
            if kind == MediaType.IMAGE:
                media_type = V1MediaType.IMAGE
            elif kind == MediaType.TEXT:
                media_type = V1MediaType.TEXT
        else:
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type:
                if mime_type.startswith("image/"):
                    media_type = V1MediaType.IMAGE
                elif mime_type.startswith("text/"):
                    media_type = V1MediaType.TEXT

        if media_type == V1MediaType.UNSPECIFIED:
            raise ValueError(f"Unsupported media type for file: {path}")

        self._media_api.upload_media(
            experiment_id=self._metrics_store.id,
            teamspace=self._teamspace,
            file_path=path,
            name=name,
            media_type=media_type,
            step=step,
            epoch=epoch,
            caption=caption,
        )
        self._stats.media_logged += 1
        if verbose:
            self._printer.media_logged(path, step)


class Experiment(LegacyExperiment):
    """Core experiment with dict-like API for logging data.

    Supports pythonic dict-like access patterns:
        experiment["key"].append(value)   # time series (metrics or files)
        experiment["key"].extend(values)  # batch time series
        experiment["key"] = "value"       # static metadata
        experiment["key"] = File("path")  # static file artifact
        experiment["key"]                 # fetch data

    Also inherits legacy method-based API from LegacyExperiment for
    backwards compatibility (log_metrics, log_file, log_model, etc.).

    Keys must be unique across metadata, artifacts, and metrics.
    """

    def __init__(
        self,
        name: str,
        log_dir: str = "lightning_logs",
        save_logs: bool = False,
        teamspace: str | Teamspace | None = None,
        light_color: str | None = None,
        dark_color: str | None = None,
        metadata: dict[str, str] | None = None,
        store_step: bool | None = True,
        store_created_at: bool | None = False,
        max_batch_size: int = 1000,
        rate_limiting_interval: int = 1,
        verbose: bool = True,
    ) -> None:
        """Initialize an experiment for logging to the https://lightning.ai platform.

        Args:
            name: A human-friendly name for your experiment.
            log_dir: Local directory where temporary logs/artifacts are stored. Defaults to "lightning_logs".
            save_logs: If True, capture and upload terminal output as a file artifact. Defaults to False.
            teamspace: Teamspace in which to create and display the charts. If None, uses your default teamspace.
            light_color: Hex color of the curve in light mode (overrides the random default). Example: "#FF5733".
            dark_color: Hex color of the curve in dark mode (overrides the random default). Example: "#3498DB".
            metadata: Key-value parameters associated with the experiment (displayed as tags in the UI).
            store_step: Whether to store the provided step for each data point. Defaults to True.
            store_created_at: Whether to store a creation timestamp for each data point. Defaults to False.
            max_batch_size: Number of metric values to batch before uploading. Defaults to 1000.
            rate_limiting_interval: Minimum seconds between uploads. Defaults to 1.
            verbose: If True, print styled console output. Defaults to True.
        """
        self.name = name
        self.save_logs = save_logs
        self._done_event = Event()
        self._finalized = False
        self.store_step = store_step
        self.store_created_at = store_created_at

        # New dict-like API state tracking
        self._key_types: dict[str, str] = {}  # key -> 'metric' | 'file_series' | 'metadata' | 'static_file'
        self._series: dict[str, Series] = {}
        self._metadata_values: dict[str, str] = {}
        self._static_files: dict[str, File] = {}

        # Initialize printer and stats tracking
        self._printer = Printer(verbose=verbose)
        self._stats = RunStats()

        self.terminal_logs_path = os.path.join(log_dir, "logs.txt")
        if self.save_logs and os.environ.get("_IN_PTY_RECORDER") != "1":
            os.makedirs(log_dir, exist_ok=True)
            # Import lazily to avoid import errors on Windows (pty module is Unix-only)
            rerun_and_record(self.terminal_logs_path)
            sys.exit(0)

        self._auth_api = AuthApi()
        is_authenticated = self._auth_api.authenticate()
        if not is_authenticated:
            self._printer.log("No credentials found. Logging in as a guest user.")
            teamspace = None

        self._metrics_api = MetricsApi()
        self._media_api = MediaApi(client=self._metrics_api.client)
        self._artifacts_api = ArtifactsApi()
        self._teamspace = _resolve_teamspace(teamspace)

        # Create metrics stream using API
        self._metrics_store, created = self._metrics_api.get_or_create_experiment_metrics(
            teamspace_id=self._teamspace.id,
            name=self.name,
            metadata=metadata,
            light_color=light_color,
            dark_color=dark_color,
            store_step=bool(store_step),
            store_created_at=bool(store_created_at),
        )

        # Build URLs using API
        if is_authenticated:
            self._url = build_experiment_url(
                owner_name=self._teamspace.owner.name,
                teamspace_name=self._teamspace.name,
                experiment_name=self.name,
            )
        else:
            self._url = get_guest_url(self._auth_api)

        self._accessible_url = get_accessible_url(
            teamspace=self._teamspace,
            owner_name=self._teamspace.owner.name,
            metrics_store=self._metrics_store,
            client=self._metrics_api.client,
        )

        # Initialize metrics management
        self._metrics_queue: JoinableQueue[dict[str, Metrics]] = JoinableQueue()
        self._stop_event = Event()
        self._is_ready_event = Event()
        self._manager = _BackgroundThread(
            teamspace_id=self._teamspace.id,
            metrics_store_id=self._metrics_store.id,
            cloud_account=self._metrics_store.cluster_id,
            metrics_api=self._metrics_api,
            metrics_queue=self._metrics_queue,
            is_ready_event=self._is_ready_event,
            stop_event=self._stop_event,
            done_event=self._done_event,
            log_dir=log_dir,
            store_step=bool(store_step),
            store_created_at=bool(store_created_at),
            rate_limiting_interval=rate_limiting_interval,
            max_batch_size=max_batch_size,
            trackers_init=self._metrics_api.get_trackers_from_metrics_store(self._metrics_store),
        )

        self._manager.start()

        # Wait for background thread to be ready
        while not self._is_ready_event.is_set():
            sleep(0.1)

        # Rebuild state from existing experiment
        if not created:
            self._rebuild_state()

        # Register atexit handler to automatically finalize on exit
        atexit.register(self.finalize)

        # Register signal handlers for graceful shutdown on SIGTERM and SIGINT
        # Note: Windows doesn't support SIGTERM, so we handle it gracefully
        with contextlib.suppress(AttributeError, ValueError):
            signal.signal(signal.SIGTERM, self._signal_handler)
        with contextlib.suppress(AttributeError, ValueError):
            signal.signal(signal.SIGINT, self._signal_handler)

    # ---- Dict-like API ----

    def __getitem__(self, key: str) -> Series:
        """Get a time series, metadata value, or static file by key.

        For time-series keys (metrics or file series), returns a list-like Series object.
        For metadata keys, returns the string value (at runtime).
        For static file keys, returns the File object (at runtime).
        For unknown keys, returns a new empty Series ready for appending.

        The return type is annotated as Series since that is the primary use case
        (``experiment["key"].append(value)``).  Metadata and static-file lookups
        return ``str`` or ``File`` at runtime.

        Args:
            key: The data key.

        Returns:
            Series for time series, str for metadata, File for static files.
        """
        if key in self._key_types:
            kt = self._key_types[key]
            if kt == "metadata":
                return _MetadataValue(key, self._metadata_values[key])  # type: ignore[return-value]
            if kt == "static_file":
                return self._static_files[key]  # type: ignore[return-value]
            # 'metric' or 'file_series'
            if key not in self._series:
                series = Series(self, key)
                if kt == "metric":
                    series._type = "metric"
                elif kt == "file_series":
                    series._type = "file"
                self._series[key] = series
            return self._series[key]
        # New key - return a series proxy for future appends
        if key not in self._series:
            self._series[key] = Series(self, key)
        return self._series[key]

    def __setitem__(self, key: str, value: str | File) -> None:
        """Set a static value (metadata string or file) on the experiment.

        Args:
            key: The data key. Must not already be in use.
            value: A string (metadata) or File (static file artifact).

        Raises:
            KeyError: If the key is already in use.
            TypeError: If value is not a str or File.
        """
        # Check for typed (but not yet registered) series
        if key in self._series and self._series[key]._type is not None:
            raise KeyError(f"Key {key!r} is already used as a time series. Cannot assign static value.")
        # Remove empty (untyped) series proxy if it exists
        self._series.pop(key, None)

        if isinstance(value, File):
            if key in self._key_types and self._key_types[key] != "static_file":
                raise KeyError(
                    f"Key {key!r} is already used as {self._key_types[key]}. Cannot reassign as static_file."
                )
            self._key_types[key] = "static_file"
            self._static_files[key] = value
            self._set_static_file(key, value)
        elif isinstance(value, str):
            if key in self._key_types and self._key_types[key] != "metadata":
                raise KeyError(f"Key {key!r} is already used as {self._key_types[key]}. Cannot reassign as metadata.")
            self._key_types[key] = "metadata"
            self._metadata_values[key] = value
            self._set_metadata_value(key, value)
        else:
            raise TypeError(f"Can only assign str or File, got {type(value).__name__}")

    def update(self, data: dict[str, str | float | int | File | list[float | int | File]]) -> None:
        """Bulk-update the experiment with multiple keys at once.

        Dispatches each value based on its type:
            str        → metadata  (same as experiment["key"] = "value")
            File       → static file artifact  (same as experiment["key"] = file)
            float/int  → append a single metric point  (same as experiment["key"].append(value))
            list       → extend a time series  (same as experiment["key"].extend(values))

        Args:
            data: Dictionary mapping keys to values of mixed types.
        """
        for key, value in data.items():
            if isinstance(value, str | File):
                self[key] = value
            elif isinstance(value, int | float):
                self._ensure_series(key).append(value)
            elif isinstance(value, list):
                self._ensure_series(key).extend(value)
            else:
                raise TypeError(f"Unsupported type for key {key!r}: {type(value).__name__}")

    # ---- Internal helpers ----

    def _register_key_type(self, key: str, key_type: str) -> None:
        """Register a key's type, ensuring uniqueness across all key types.

        Args:
            key: The data key.
            key_type: One of 'metric', 'file_series', 'metadata', 'static_file'.

        Raises:
            KeyError: If the key is already registered with a different type.
        """
        if key in self._key_types:
            if self._key_types[key] != key_type:
                raise KeyError(f"Key {key!r} is already used as {self._key_types[key]}, cannot use as {key_type}")
            return  # Already registered with same type
        self._key_types[key] = key_type

    def _rebuild_state(self) -> None:
        """Rebuild local state from an existing experiment's remote data.

        Called when connecting to an experiment that already exists. Populates
        metadata and metric key types from the metrics store.
        """
        # Rebuild metadata from tags
        tags = getattr(self._metrics_store, "tags", None) or []
        for tag in tags:
            if tag.from_code:
                self._key_types[tag.name] = "metadata"
                self._metadata_values[tag.name] = tag.value

        # Rebuild metric key types from trackers
        trackers = self._metrics_api.get_trackers_from_metrics_store(self._metrics_store)
        if trackers:
            for name in trackers:
                self._key_types[name] = "metric"

        # Rebuild artifact key types from registered artifacts
        artifacts = getattr(self._metrics_store, "artifacts", None) or []
        for artifact in artifacts:
            key = artifact.path if hasattr(artifact, "path") else str(artifact)
            if key not in self._key_types:
                self._key_types[key] = "static_file"
                file = File(key)
                file.name = key
                file._download_fn = self._create_download_fn(key)
                self._static_files[key] = file

    def _create_download_fn(self, key: str) -> Callable[[str], str]:
        """Create a download function for a given artifact key."""

        def _download(path: str) -> str:
            artifact = Artifact(
                path=path,
                experiment_name=self.name,
                teamspace=self._teamspace,
                metrics_store=self._metrics_store,
                client=self._artifacts_api.client,
                remote_path=key,
            )
            return artifact.get()

        return _download

    def _log_metric_value(self, key: str, value: float, step: int | None = None) -> None:
        """Internal: log a single metric value to the background queue.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number (used by legacy API).
        """
        if self._manager.exception is not None:
            raise self._manager.exception

        created_at = datetime.now() if self.store_created_at else None
        actual_step = step if self.store_step else None
        mv = MetricValue(value=value, created_at=created_at, step=actual_step)
        batch: dict[str, Metrics] = {key: Metrics(name=key, values=[mv])}
        self._metrics_queue.put(batch)
        self._stats.record_metric(key, value)

    def _log_file_series_value(self, key: str, value: File, index: int) -> None:
        """Internal: upload a file as part of a time series.

        Args:
            key: Series key.
            value: File to upload.
            index: Index in the series (used for naming).
        """
        upload_path = value._get_upload_path()
        remote_path = f"{key}/{index}"
        artifact = Artifact(
            path=upload_path,
            experiment_name=self.name,
            teamspace=self._teamspace,
            metrics_store=self._metrics_store,
            client=self._artifacts_api.client,
            remote_path=remote_path,
        )
        artifact.log()
        self._stats.artifacts_logged += 1
        value._cleanup()

        # Bind remote name and download capability
        value.name = remote_path
        value._download_fn = self._create_download_fn(remote_path)

    def _set_metadata_value(self, key: str, value: str) -> None:
        """Internal: set a metadata value and push to API.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        current_tags = self.metadata
        current_tags[key] = value
        self._metrics_api.update_experiment_metrics(
            teamspace_id=self._teamspace.id,
            metrics_store_id=self._metrics_store.id,
            phase=PhaseType.RUNNING,
            metadata=current_tags,
        )

    def _set_static_file(self, key: str, value: File) -> None:
        """Internal: upload a static file artifact.

        Args:
            key: Artifact key (used as remote path).
            value: File to upload.
        """
        upload_path = value._get_upload_path()
        artifact = Artifact(
            path=upload_path,
            experiment_name=self.name,
            teamspace=self._teamspace,
            metrics_store=self._metrics_store,
            client=self._artifacts_api.client,
            remote_path=key,
        )
        artifact.log()
        self._stats.artifacts_logged += 1
        value._cleanup()

        # Bind remote name and download capability so self[key].save(path) works
        value.name = key
        value._download_fn = self._create_download_fn(key)

    def _ensure_series(self, key: str) -> Series:
        """Get or create a series proxy for a key.

        Args:
            key: The series key.

        Returns:
            The Series object for this key.
        """
        if key not in self._series:
            self._series[key] = Series(self, key)
        return self._series[key]

    # ---- Properties ----

    @property
    def url(self) -> str:
        """Get the direct URL to view this experiment in the Lightning.ai web interface.

        Returns:
            str: The full URL to the experiment's visualization page.
        """
        return self._url

    @property
    def teamspace(self) -> Teamspace:
        """Get the teamspace for this experiment.

        Returns:
            Teamspace: The teamspace object.
        """
        return self._teamspace

    @property
    def metadata(self) -> dict[str, str]:
        """Get the metadata associated with this experiment from the metrics stream.

        Returns:
            dict[str, str]: The metadata dictionary with key-value pairs from code-defined tags.
        """
        self._update_metrics_store()
        tags = getattr(self._metrics_store, "tags", None) or []
        return {tag.name: tag.value for tag in tags if tag.from_code}

    @property
    def metrics(self) -> dict[str, Series]:
        """Get all metric time series logged to this experiment.

        Returns:
            dict[str, Series]: Mapping of metric names to their series of values.
        """
        return {key: series for key, series in self._series.items() if series._type == "metric"}

    @property
    def artifacts(self) -> dict[str, File | Series]:
        """Get all artifacts (static files and file series) logged to this experiment.

        Returns:
            dict[str, File | Series]: Mapping of artifact keys to File (static)
                or Series (time series of files).
        """
        result: dict[str, File | Series] = {}
        for key, f in self._static_files.items():
            result[key] = f
        for key, series in self._series.items():
            if series._type == "file":
                result[key] = series
        return result

    # ---- Lifecycle ----

    def finalize(self, status: str | None = None, print_summary: bool = True) -> None:
        """Finalize the experiment and upload all remaining metrics.

        This method waits for the background thread to finish uploading all queued metrics,
        and uploads terminal logs if save_logs=True. It's automatically called on exit
        via an atexit handler, but can also be called manually.

        This method is idempotent and can be called multiple times safely.

        Args:
            status: Optional status string for the experiment (currently unused, reserved for future use).
            print_summary: Whether to print the run completion summary. Defaults to True.
        """
        # Return early if already finalized
        if self._finalized:
            return

        # Mark as finalized
        self._finalized = True

        # Wait for the queue to be fully processed
        self._metrics_queue.join()

        # Trigger stop event
        self._stop_event.set()

        # Wait for all the metrics to be uploaded
        while not self._done_event.is_set():
            if self._manager.exception is not None:
                raise self._manager.exception
            sleep(0.1)

        if self.save_logs and os.path.exists(self.terminal_logs_path):
            artifact = Artifact(
                path=self.terminal_logs_path,
                experiment_name=self.name,
                teamspace=self._teamspace,
                metrics_store=self._metrics_store,
                client=self._artifacts_api.client,
                remote_path="console_output.txt",
            )
            artifact.log()

        # Print completion summary with stats
        if print_summary:
            self._printer.experiment_complete(
                name=self.name,
                stats=self._stats,
                url=self._url,
            )

    def print_url(self) -> None:
        """Print the experiment URL and initialization info with styled output."""
        self._printer.experiment_start(
            name=self.name,
            teamspace=self._teamspace.name,
            url=self._url,
            metadata=self.metadata,
        )

    def _update_metrics_store(self) -> None:
        """Re-fetch the metrics store from the API to refresh local state (tags, trackers, etc.)."""
        resp = self._metrics_api.get_experiment_metrics_by_name(
            self._teamspace.id,
            name=self._metrics_store.name,
        )

        if resp is not None:
            self._metrics_store = resp

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle termination signals by calling finalize().

        Args:
            signum: Signal number.
            frame: Current stack frame (unused).
        """
        # Call finalize and then exit with appropriate code
        # For SIGTERM (15) and SIGINT (2), exit with 128 + signal number
        # This follows the convention for signal-induced termination
        self.finalize()
        sys.exit(128 + signum)
