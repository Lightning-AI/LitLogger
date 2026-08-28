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

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import sys
from types import FrameType
from typing import TYPE_CHECKING, Any

from lightning_sdk import Teamspace

from litlogger.capture import rerun_and_record
from litlogger.experiment_legacy import LegacyExperiment, MetadataValue
from litlogger.primitives import (
    File,
    Metadata,
    Metric,
    Model,
    RestoredFiles,
    WritePlacement,
    _to_v1_media_type,
)
from litlogger.series import Series
from litlogger.session import ExperimentSession
from litlogger.types import MediaType

if TYPE_CHECKING:
    import queue

    from litlogger.api.artifacts_api import ArtifactsApi
    from litlogger.api.media_api import MediaApi
    from litlogger.api.metrics_api import MetricsApi
    from litlogger.background import _BackgroundThread
    from litlogger.primitives import QueueItem
    from litlogger.printer import Printer, RunStats


class Experiment(LegacyExperiment):
    """Core experiment with dict-like API for logging data.

    Supports pythonic dict-like access patterns:
        experiment["key"].append(value)   # time series (metrics or files)
        experiment["key"].extend(values)  # batch time series
        experiment["key"] = "value"       # static metadata
        experiment["key"] = File("path")  # static file artifact
        experiment["key"]                 # fetch data

    Also inherits the legacy method-based API used for backwards
    compatibility (log_metrics, log_file, log_model, etc.).

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
        self._finalized = False
        self.store_step = store_step
        self.store_created_at = store_created_at

        # New dict-like API state tracking
        self._key_types: dict[str, str] = {}  # key -> 'metric' | 'file_series' | 'metadata' | 'static_file'
        self._series: dict[str, Series] = {}
        self._metadata_values: dict[str, str] = {}
        self._static_files: dict[str, File] = {}
        self.terminal_logs_path = os.path.join(log_dir, "logs.txt")
        if self.save_logs and os.environ.get("_IN_PTY_RECORDER") != "1":
            os.makedirs(log_dir, exist_ok=True)
            # Import lazily to avoid import errors on Windows (pty module is Unix-only)
            rerun_and_record(self.terminal_logs_path)
            sys.exit(0)

        self._session = ExperimentSession(
            name=name,
            teamspace=teamspace,
            metadata=metadata,
            light_color=light_color,
            dark_color=dark_color,
            store_step=bool(store_step),
            store_created_at=bool(store_created_at),
            rate_limiting_interval=rate_limiting_interval,
            max_batch_size=max_batch_size,
            verbose=verbose,
            experiment=self,
        )

        # Rebuild state from existing experiment
        if not self._session.created:
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
                return MetadataValue(key, self._metadata_values[key])  # type: ignore[return-value]
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
        if isinstance(value, File):
            if key in self._key_types and self._key_types[key] != "static_file":
                raise KeyError(
                    f"Key {key!r} is already used as {self._key_types[key]}. Cannot reassign as static_file."
                )
            self._validate_file_primitive(value)
            value.enqueue(self._session, WritePlacement(key))
            self._series.pop(key, None)
            self._key_types[key] = "static_file"
            self._static_files[key] = value
        elif isinstance(value, str):
            if key in self._key_types and self._key_types[key] != "metadata":
                raise KeyError(f"Key {key!r} is already used as {self._key_types[key]}. Cannot reassign as metadata.")
            Metadata(key, value).enqueue(self._session, WritePlacement(key))
            self._series.pop(key, None)
            self._key_types[key] = "metadata"
            self._metadata_values[key] = value
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

    # ---- Coercion and dispatch ----

    def _validate_file_primitive(self, value: File) -> None:
        """Reject file wrappers that claim model semantics without being Models."""
        if value._media_type == MediaType.MODEL and not isinstance(value, Model):
            raise TypeError("Model media values must use the Model wrapper.")

    def _register_key_type(self, key: str, key_type: str) -> None:
        if key in self._key_types:
            if self._key_types[key] != key_type:
                raise KeyError(f"Key {key!r} is already used as {self._key_types[key]}, cannot use as {key_type}")
            return
        self._key_types[key] = key_type

    def _ensure_series(self, key: str) -> Series:
        if key not in self._series:
            self._series[key] = Series(self, key)
        return self._series[key]

    def _log_metric_value(
        self,
        key: str,
        y: float,
        x: float | None = None,
    ) -> None:
        Metric(key, y, x=x).enqueue(self._session)

    def _log_file_series_value(self, key: str, value: File, index: int, step: float | None = None) -> None:
        self._validate_file_primitive(value)
        value.enqueue(self._session, WritePlacement(key=key, index=index, x=step))

    def _upload_media(
        self,
        name: str,
        file_path: str,
        media_type: MediaType,
        step: float | None = None,
        epoch: int | None = None,
        caption: str | None = None,
    ) -> None:
        # Keyless media upload used by the legacy log_media API: it registers
        # nothing locally, so it stays outside the primitive dispatch.
        self._session.media_api.upload_media(
            experiment_id=self._session.metrics_store.id,
            teamspace=self._session.teamspace,
            file_path=file_path,
            name=name,
            media_type=_to_v1_media_type(media_type),
            step=step,
            epoch=epoch,
            caption=caption,
        )
        self._stats.media_logged += 1

    # ---- Resume orchestration ----

    def resolve_model(self, key: str) -> Model | Series | None:
        """Resolve one model key explicitly through the remote registry.

        Generic ``experiment[key]`` access never performs network I/O. Call
        this method when a model written by an earlier process needs to be
        reconstructed from the registry.
        """
        self._session.flush()
        resolved = Model._resolve(self._session, key)
        if resolved is None:
            return None

        if isinstance(resolved, list):
            series = Series(self, key)
            series._type = "file"
            series._values = list(resolved)
            self._key_types[key] = "file_series"
            self._series[key] = series
            return series

        self._key_types[key] = "static_file"
        self._static_files[key] = resolved
        return resolved

    def _rebuild_state(self) -> None:
        """Rebuild local state from remote metadata, metrics, artifacts, and media."""
        # TODO: add BE support for restoring model states as well
        session = self._session

        for name, value in Metadata._current_tags(session).items():
            self._key_types[name] = "metadata"
            self._metadata_values[name] = value

        metric_values = Metric._restore_values(session)
        for name in self._resumed_steps.keys() | metric_values.keys():
            self._key_types[name] = "metric"
            series = Series(self, name)
            series._type = "metric"
            if name in metric_values:
                series._values = list(metric_values[name])
            self._series[name] = series

        self._merge_restored(File._restore_all(session, dict(self._key_types)))
        self._merge_restored(File._restore_media(session, dict(self._key_types)))

    def _merge_restored(self, restored: RestoredFiles) -> None:
        """Register one restore pass's results in local experiment state."""
        for key, file in restored.statics.items():
            self._key_types[key] = "static_file"
            self._static_files[key] = file
        for key, values in restored.series.items():
            self._key_types[key] = "file_series"
            series = Series(self, key)
            series._type = "file"
            series._values = values
            self._series[key] = series

    # ---- Properties ----

    @property
    def _metrics_api(self) -> MetricsApi:
        return self._session.metrics_api

    @property
    def _media_api(self) -> MediaApi:  # type: ignore[override]
        return self._session.media_api

    @property
    def _artifacts_api(self) -> ArtifactsApi:
        return self._session.artifacts_api

    @property
    def _teamspace(self) -> Teamspace:
        return self._session.teamspace

    @property
    def _metrics_store(self) -> Any:
        return self._session.metrics_store

    @property
    def _metrics_queue(self) -> queue.Queue[QueueItem]:
        return self._session.queue

    @property
    def _resumed_steps(self) -> dict[str, float]:
        return self._session.last_x

    @property
    def _manager(self) -> _BackgroundThread:
        return self._session.background

    @property
    def _stats(self) -> RunStats:  # type: ignore[override]
        return self._session.stats

    @property
    def _printer(self) -> Printer:  # type: ignore[override]
        return self._session.printer

    @property
    def _url(self) -> str:
        return self._session.url

    @property
    def url(self) -> str:
        """Get the direct URL to view this experiment in the Lightning.ai web interface.

        Returns:
            str: The full URL to the experiment's visualization page.
        """
        return self._session.url

    @property
    def teamspace(self) -> Teamspace:
        """Get the teamspace for this experiment.

        Returns:
            Teamspace: The teamspace object.
        """
        return self._session.teamspace

    @property
    def session(self) -> ExperimentSession:
        """The shared infrastructure session primitives log through.

        Returns:
            ExperimentSession: The session created for this experiment.
        """
        return self._session

    @property
    def metadata(self) -> dict[str, str]:
        """Get the metadata associated with this experiment from the metrics stream.

        Returns:
            dict[str, str]: The metadata dictionary with key-value pairs from code-defined tags.
        """
        # Read barrier: queued metadata writes must land before the remote read.
        self._session.flush()
        return Metadata._current_tags(self._session)

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

        self._session.finalize()

        if self.save_logs and os.path.exists(self.terminal_logs_path):
            # Uploaded directly (not registered locally, no stats bump) —
            # console output is bookkeeping, not experiment data.
            File(self.terminal_logs_path)._upload_artifact(self._session, remote_path="console_output.txt")

        # Only a successfully completed finalization is idempotent. A failed
        # attempt must remain retryable and continue surfacing its exception.
        self._finalized = True

        # Print completion summary with stats
        if print_summary:
            self._printer.experiment_complete(
                name=self.name,
                stats=self._stats,
                url=self._session.url,
            )

    def print_url(self) -> None:
        """Print the experiment URL and initialization info with styled output."""
        self._printer.experiment_start(
            name=self.name,
            teamspace=self._session.teamspace.name,
            url=self._session.url,
            metadata=self.metadata,
        )

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

    @property
    def id(self) -> str:
        return self._session.metrics_store_id
