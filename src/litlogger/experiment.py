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
import os
import signal
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import JoinableQueue
from threading import Event
from time import sleep
from types import FrameType
from typing import TYPE_CHECKING, Any, Dict, List, Union

from litlogger.api.artifacts_api import ArtifactsApi
from litlogger.api.auth_api import AuthApi
from litlogger.api.metrics_api import MetricsApi
from litlogger.api.utils import _resolve_teamspace, build_experiment_url, get_accessible_url, get_guest_url
from litlogger.artifacts import Artifact, Model, ModelArtifact
from litlogger.background import _BackgroundThread
from litlogger.capture import rerun_and_record
from litlogger.printer import Printer, RunStats
from litlogger.types import Metrics, MetricValue

if TYPE_CHECKING:
    from lightning_sdk import Teamspace


class Experiment:
    """High-level interface to log, store, and fetch metrics and artifacts of all kinds.

    This class manages the full lifecycle of an experiment on Lightning.ai, including:
    - Creating a metrics stream with automatic buffering and batching
    - Uploading metrics to the cloud in the background
    - Logging and retrieving files, models, and model artifacts
    - Gracefully finalizing the experiment on exit (via atexit handler)

    The experiment can be used directly or via the module-level API (litlogger.init()).
    """

    def __init__(
        self,
        name: str,
        version: str,
        log_dir: str = "lightning_logs",
        save_logs: bool = False,
        teamspace: Union[str, "Teamspace"] | None = None,
        light_color: str | None = None,
        dark_color: str | None = None,
        metadata: Dict[str, str] | None = None,
        store_step: bool | None = True,
        store_created_at: bool | None = False,
        max_batch_size: int = 1000,
        rate_limiting_interval: int = 1,
        verbose: bool = True,
    ) -> None:
        """Initialize an experiment for logging to the https://lightning.ai platform.

        Args:
            name: A human-friendly name for your experiment.
            version: An experiment version identifier (typically a timestamp string).
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
        self.version = version
        self.save_logs = save_logs
        self._done_event = Event()
        self._finalized = False
        self.store_step = store_step
        self.store_created_at = store_created_at
        self._metadata = metadata

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
        self._artifacts_api = ArtifactsApi()
        self._teamspace = _resolve_teamspace(teamspace)

        # Create metrics stream using API
        self._metrics_store = self._metrics_api.create_experiment_metrics(
            teamspace_id=self._teamspace.id,
            name=self.name,
            version=self.version,
            metadata=metadata,
            light_color=light_color,
            dark_color=dark_color,
            store_step=store_step,
            store_created_at=store_created_at,
        )

        # Build URLs using API - use version_number from metrics store for clean URLs
        if is_authenticated:
            self._url = build_experiment_url(
                owner_name=self._teamspace.owner.name,
                teamspace_name=self._teamspace.name,
                experiment_name=self.name,
                version=self._metrics_store.version_number,
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
        self._metrics_queue = JoinableQueue()
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
            version=version,
            store_step=store_step,
            store_created_at=store_created_at,
            rate_limiting_interval=rate_limiting_interval,
            max_batch_size=max_batch_size,
        )

        self._manager.start()

        # Wait for background thread to be ready
        while not self._is_ready_event.is_set():
            sleep(0.1)

        # Register atexit handler to automatically finalize on exit
        atexit.register(self.finalize)

        # Register signal handlers for graceful shutdown on SIGTERM and SIGINT
        # Note: Windows doesn't support SIGTERM, so we handle it gracefully
        with contextlib.suppress(AttributeError, ValueError):
            signal.signal(signal.SIGTERM, self._signal_handler)
        with contextlib.suppress(AttributeError, ValueError):
            signal.signal(signal.SIGINT, self._signal_handler)

    @property
    def url(self) -> str:
        """Get the direct URL to view this experiment in the Lightning.ai web interface.

        Returns:
            str: The full URL to the experiment's visualization page.
        """
        return self._url

    @property
    def teamspace(self) -> "Teamspace":
        """Get the teamspace for this experiment.

        Returns:
            Teamspace: The teamspace object.
        """
        return self._teamspace

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log metrics to the experiment with background uploading.

        Metrics are buffered locally and uploaded to the cloud in batches to optimize performance.
        The batch is sent when either 1 second has passed or 1000 values have been logged.

        Args:
            metrics: Dictionary mapping metric names to numeric values. Example: {"loss": 0.5, "accuracy": 0.95}.
            step: Optional step number for this data point (e.g., training step, epoch).
                If None and store_step=True, no step is recorded.

        Raises:
            RuntimeError: If the background thread encountered an error.
        """
        if self._manager.exception is not None:
            raise self._manager.exception

        batch: Dict[str, Metrics] = {}
        for name, value in metrics.items():
            created_at = None
            if self.store_created_at:
                created_at = datetime.now()

            actual_step = step if self.store_step else None
            metric_value = MetricValue(value=float(value), created_at=created_at, step=actual_step)

            if name not in batch:
                batch[name] = Metrics(name=name, values=[metric_value])
            else:
                batch[name].values.append(metric_value)

            # Track for summary sparklines
            self._stats.record_metric(name, float(value))

        # Push to queue immediately - background thread handles batching and rate limiting
        if batch:
            self._metrics_queue.put(batch)

    def log_metrics_batch(self, metrics: Dict[str, List[Dict[str, float]]]) -> None:
        """Log a batch of metrics through the background queue.

        This method converts the batch format to Metrics objects and pushes them
        through the background queue, which handles batching and chunking to respect
        API limits.

        Args:
            metrics: Dictionary mapping metric names to lists of dicts with "step" and "value" keys.

        Example:
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
        if self._manager.exception is not None:
            raise self._manager.exception

        if not metrics:
            return

        batch: Dict[str, Metrics] = {}
        for name, values in metrics.items():
            metric_values = []
            for v in values:
                created_at = datetime.now() if self.store_created_at else None
                metric_values.append(MetricValue(value=v["value"], step=v.get("step", None), created_at=created_at))
            batch[name] = Metrics(name=name, values=metric_values)

        self._metrics_queue.put(batch)

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
            self.log_file(self.terminal_logs_path, remote_path="console_output.txt", verbose=False)

        # Print completion summary with stats
        if print_summary:
            self._printer.experiment_complete(
                name=self.name,
                stats=self._stats,
                url=self._url,
            )

    def log_file(
        self,
        path: str,
        remote_path: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Upload a file artifact to the cloud for this experiment.

        The file is uploaded to cloud storage and registered with the experiment,
        making it visible in the artifacts view and accessible via get_file().

        Args:
            path: Path to the local file to upload. Can be absolute or relative.
            remote_path: Path relative to experiment root for storage and display.
                If None, uses the path relative to cwd if under cwd, otherwise basename.
                Example: remote_path="images/0.png" will store and display as "images/0.png".
            verbose: Whether to print a confirmation message after upload. Defaults to True.
        """
        artifact = Artifact(
            path=path,
            experiment_name=self.name,
            teamspace=self._teamspace,
            metrics_store=self._metrics_store,
            client=self._artifacts_api.client,
            remote_path=remote_path,
        )
        artifact.log()
        self._stats.artifacts_logged += 1
        if verbose:
            self._printer.artifact_logged(path, remote_path)

    def log_files(
        self,
        paths: list[str],
        remote_paths: List[str] | None = None,
        max_workers: int = 10,
    ) -> None:
        """Upload multiple file artifacts to the cloud in parallel.

        This is more efficient than calling log_file() multiple times when you have
        many files, as it handles them in parallel.

        Args:
            paths: List of paths to local files to upload.
            remote_paths: Optional list of remote paths, one for each file in paths.
                If provided, must have same length as paths.
                If None, each file uses its default remote path (relative to cwd or basename).
            max_workers: Maximum number of concurrent uploads. Defaults to 10.
        """
        if remote_paths is None:
            remote_paths = [None] * len(paths)

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
        artifact = Artifact(
            path=path,
            experiment_name=self.name,
            teamspace=self._teamspace,
            metrics_store=self._metrics_store,
            client=self._artifacts_api.client,
            remote_path=remote_path,
        )
        downloaded_path = artifact.get()
        if verbose:
            self._printer.artifact_retrieved(downloaded_path)
        return downloaded_path

    def log_model_artifact(self, path: str, verbose: bool = False, version: str | None = None) -> None:
        """Upload a model file or directory to cloud storage using litmodels.

        This uploads raw model files (e.g., weights.pt, checkpoint.ckpt) or entire directories
        to the litmodels registry. Use this when you have pre-saved model files.

        For saving model objects directly, use log_model() instead.

        Args:
            path: Path to the local model file or directory to upload.
            verbose: Whether to show progress bar during upload. Defaults to False.
            version: Optional version string for the model. Defaults to the experiment version.
        """
        model_artifact = ModelArtifact(
            path=path,
            experiment_name=self.name,
            teamspace=self._teamspace,
            version=version or self.version,
            verbose=verbose,
        )
        model_artifact.log()
        self._stats.models_logged += 1
        if verbose:
            self._printer.artifact_logged(path, f"model artifact: {path}")

    def get_model_artifact(self, path: str, verbose: bool = False, version: str | None = None) -> str:
        """Download a model artifact file or directory from cloud storage using litmodels.

        This downloads raw model files or directories that were previously uploaded
        via log_model_artifact(). The files are saved to the specified local path.

        Args:
            path: Path where the model should be saved locally. Directories are created if needed.
            verbose: Whether to show progress bar during download. Defaults to False.
            version: Optional version string for the model. Defaults to the experiment version.

        Returns:
            str: The local path where the model was saved (same as the input path).
        """
        model_artifact = ModelArtifact(
            path=path,
            experiment_name=self.name,
            teamspace=self._teamspace,
            version=version or self.version,
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
        metadata: dict | None = None,
    ) -> str:
        """Save and upload a model object to cloud storage using litmodels.

        This saves a live model object (e.g., PyTorch module, LightningModule) to disk
        using framework-specific serialization, then uploads it to the litmodels registry.

        For uploading pre-saved model files, use log_model_artifact() instead.

        Args:
            model: The model object to save and upload (e.g., torch.nn.Module, LightningModule).
            staging_dir: Optional local directory for staging the model before upload. If None, uses a temp directory.
            verbose: Whether to show progress bar during upload. Defaults to False.
            version: Optional version string for the model. Defaults to the experiment version.
            metadata: Optional metadata dictionary to store with the model (e.g., hyperparameters, metrics).

        Returns:
            str: Information about the uploaded model (details from litmodels).
        """
        model_obj = Model(
            model=model,
            experiment_name=self.name,
            teamspace=self._teamspace,
            version=version or self.version,
            verbose=verbose,
            metadata=metadata,
            staging_dir=staging_dir,
        )
        result = model_obj.log()
        self._stats.models_logged += 1
        if verbose:
            self._printer.print_success("Logged model object")
        return result

    def get_model(self, staging_dir: str | None = None, verbose: bool = False, version: str | None = None) -> Any:
        """Get a model object using litmodels load_model.

        Args:
            staging_dir: Optional directory where the model will be downloaded.
            verbose: Whether to show progress bar.
            version: Optional version string for the model.

        Returns:
            The loaded model object.
        """
        model_obj = Model(
            model=None,  # Not needed for loading
            experiment_name=self.name,
            teamspace=self._teamspace,
            version=version or self.version,
            verbose=verbose,
            staging_dir=staging_dir,
        )
        result = model_obj.get()
        if verbose:
            self._printer.print_success("Retrieved model object")
        return result

    def print_url(self) -> None:
        """Print the experiment URL and initialization info with styled output."""
        self._printer.experiment_start(
            name=self.name,
            teamspace=self._teamspace.name,
            url=self._url,
            version=self.version,
            metadata=self._metadata,
        )
