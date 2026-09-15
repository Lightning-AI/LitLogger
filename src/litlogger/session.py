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
"""Session-owned infrastructure and lifecycle for one experiment."""

from __future__ import annotations

import queue
import threading
import weakref
from typing import TYPE_CHECKING, Any

from lightning_sdk import Teamspace

from litlogger.api.artifacts_api import ArtifactsApi
from litlogger.api.auth_api import AuthApi
from litlogger.api.media_api import MediaApi
from litlogger.api.metrics_api import MetricsApi
from litlogger.api.utils import _resolve_teamspace, build_experiment_url, get_accessible_url
from litlogger.background import _BackgroundThread
from litlogger.printer import Printer, RunStats

if TYPE_CHECKING:
    from litlogger.experiment import Experiment
    from litlogger.primitives import QueueItem


class ExperimentSession:
    """Own the remote clients, store, queue, worker, and mutable write state.

    ``Experiment`` constructs exactly one session and delegates infrastructure
    work to it. Primitives receive the same session for synchronous writes,
    background submission, restore, and read barriers.

    Args:
        name: Experiment name.
        teamspace: Teamspace name or object. ``None`` selects the default.
        metadata: Initial code-defined metadata.
        light_color: Optional light-mode chart color.
        dark_color: Optional dark-mode chart color.
        store_step: Whether x-coordinates are persisted through the backend's
            legacy step field.
        store_created_at: Whether metric timestamps are persisted.
        max_batch_size: Maximum metric values per request.
        rate_limiting_interval: Minimum seconds between metric requests.
        verbose: Whether the run printer emits output.
        experiment: Optional owning experiment used only as a weak SDK model
            linkage; infrastructure never reads state back through it.
    """

    def __init__(
        self,
        name: str,
        *,
        teamspace: str | Teamspace | None = None,
        metadata: dict[str, str] | None = None,
        light_color: str | None = None,
        dark_color: str | None = None,
        store_step: bool = True,
        store_created_at: bool = False,
        max_batch_size: int = 1000,
        rate_limiting_interval: int = 1,
        verbose: bool = True,
        experiment: Experiment | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero.")
        if rate_limiting_interval < 0:
            raise ValueError("rate_limiting_interval must be non-negative.")

        self.name = name
        self.store_step = store_step
        self.store_created_at = store_created_at
        self.printer = Printer(verbose=verbose)
        self.stats = RunStats()
        self._experiment_ref = weakref.ref(experiment) if experiment is not None else None

        auth_api = AuthApi()
        auth_api.authenticate()
        self.auth_api: AuthApi | None = auth_api

        self.metrics_api = MetricsApi()
        self.media_api = MediaApi(client=self.metrics_api.client)
        self.artifacts_api = ArtifactsApi(client=self.metrics_api.client)
        self.teamspace = _resolve_teamspace(teamspace)
        self.metrics_store, self.created = self.metrics_api.get_or_create_experiment_metrics(
            teamspace_id=self.teamspace.id,
            name=name,
            metadata=metadata,
            light_color=light_color,
            dark_color=dark_color,
            store_step=store_step,
            store_created_at=store_created_at,
        )
        self.url = build_experiment_url(
            owner_name=self.teamspace.owner.name,
            teamspace_name=self.teamspace.name,
            experiment_name=name,
        )
        self.accessible_url = get_accessible_url(
            teamspace=self.teamspace,
            owner_name=self.teamspace.owner.name,
            metrics_store=self.metrics_store,
            client=self.metrics_api.client,
        )

        self.queue: queue.Queue[QueueItem] = queue.Queue()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.done_event = threading.Event()
        self.last_x = self.metrics_api.get_last_steps(self.teamspace.id, self.metrics_store.id) or {}
        self._coordinate_lock = threading.Lock()
        self._submission_lock = threading.Lock()
        self._accepting = True
        self._failure: Exception | None = None
        self._finalized = False

        self.background = _BackgroundThread(
            session=self,
            rate_limiting_interval=rate_limiting_interval,
            max_batch_size=max_batch_size,
        )
        self.background.start()
        self.ready_event.wait()

    @classmethod
    def _from_components(
        cls: type[ExperimentSession],
        *,
        name: str,
        metrics_api: MetricsApi,
        media_api: MediaApi,
        artifacts_api: ArtifactsApi,
        teamspace: Teamspace,
        metrics_store: Any,
        queue_: queue.Queue[QueueItem],
        stats: RunStats,
        printer: Printer,
        background: _BackgroundThread,
        store_step: bool = True,
        store_created_at: bool = False,
        last_x: dict[str, float] | None = None,
        experiment: Experiment | None = None,
    ) -> ExperimentSession:
        """Build a session around injected components for isolated tests."""
        self = cls.__new__(cls)
        self.name = name
        self.store_step = store_step
        self.store_created_at = store_created_at
        self.printer = printer
        self.stats = stats
        self._experiment_ref = weakref.ref(experiment) if experiment is not None else None
        self.auth_api = None
        self.metrics_api = metrics_api
        self.media_api = media_api
        self.artifacts_api = artifacts_api
        self.teamspace = teamspace
        self.metrics_store = metrics_store
        self.created = False
        self.url = ""
        self.accessible_url = ""
        self.queue = queue_
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.done_event = threading.Event()
        self.last_x = last_x if last_x is not None else {}
        self._coordinate_lock = threading.Lock()
        self._submission_lock = threading.Lock()
        self._accepting = True
        self._failure = None
        self._finalized = False
        self.background = background
        return self

    @property
    def client(self) -> Any:
        """The REST client shared by all API wrappers."""
        return self.metrics_api.client

    @property
    def experiment_link(self) -> Any:
        """Weak owning experiment reference for SDK model linkage."""
        if self._experiment_ref is None:
            return self
        experiment = self._experiment_ref()
        return experiment if experiment is not None else self

    @property
    def experiment_name(self) -> str:
        """Experiment name used for remote paths and model names."""
        return self.name

    @property
    def teamspace_id(self) -> str:
        """Identifier of the owning teamspace."""
        return str(self.teamspace.id)

    @property
    def metrics_store_id(self) -> str:
        """Identifier of the remote metrics stream."""
        return str(self.metrics_store.id)

    @property
    def last_steps(self) -> dict[str, float]:
        """Legacy alias for the per-series last-x mapping."""
        return self.last_x

    @property
    def last_steps_lock(self) -> threading.Lock:
        """Legacy alias for the coordinate lock."""
        return self._coordinate_lock

    def refresh_metrics_store(self) -> None:
        """Refresh the metrics stream in the session-owned mutable slot."""
        refreshed = self.metrics_api.get_experiment_metrics_by_name(
            self.teamspace.id,
            name=self.metrics_store.name,
        )
        if refreshed is not None:
            self.metrics_store = refreshed

    def resolve_x(self, key: str, x: float | None) -> float:
        """Resolve and record an explicit or auto-incremented x-coordinate."""
        with self._coordinate_lock:
            resolved = self.last_x.get(key, -1) + 1 if x is None else x
            self.last_x[key] = resolved
            return resolved

    def submit(self, item: QueueItem) -> None:
        """Atomically reject failed/closed sessions or enqueue one command."""
        with self._submission_lock:
            self.raise_if_background_failed()
            if not self._accepting:
                raise RuntimeError("The experiment session is no longer accepting writes.")
            self.queue.put(item)

    def _record_background_failure(self, exception: Exception) -> None:
        """Close submission and discard commands the failed worker cannot run."""
        with self._submission_lock:
            self._failure = exception
            self._accepting = False
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
                self.queue.task_done()

    def raise_if_background_failed(self) -> None:
        """Raise the background worker's captured exception, if any."""
        failure = self._failure or self.background.exception
        if failure is not None:
            raise failure

    def flush(self) -> None:
        """Flush submitted commands and buffered metrics, then surface failures."""
        from litlogger.primitives import PrimitiveWrite

        self.submit(PrimitiveWrite(lambda active_session: active_session.background.flush_metrics()))
        self.queue.join()
        self.raise_if_background_failed()

    def finalize(self) -> None:
        """Close submission and finish the background worker exactly once."""
        if self._finalized:
            return
        with self._submission_lock:
            self.raise_if_background_failed()
            self._accepting = False
        self.queue.join()
        self.raise_if_background_failed()
        self.stop_event.set()
        self.done_event.wait()
        self.raise_if_background_failed()
        self._finalized = True
