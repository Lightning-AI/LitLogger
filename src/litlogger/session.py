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
"""Shared infrastructure session handed to logging primitives.

The :class:`ExperimentSession` bundles everything a primitive needs to perform
its remote write or restore: the shared REST client, the API wrappers, the
teamspace and experiment identity, the metrics queue, and run statistics.
Primitives receive a session instead of constructing clients or API wrappers
themselves.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multiprocessing import JoinableQueue

    from lightning_sdk import Teamspace

    from litlogger.api.artifacts_api import ArtifactsApi
    from litlogger.api.client import LitRestClient
    from litlogger.api.media_api import MediaApi
    from litlogger.api.metrics_api import MetricsApi
    from litlogger.background import _BackgroundThread
    from litlogger.experiment import Experiment
    from litlogger.printer import RunStats
    from litlogger.types import Metrics


class ExperimentSession:
    """Shared remote-logging context owned by an :class:`~litlogger.experiment.Experiment`.

    The session is created once per experiment, after authentication, teamspace
    resolution, and experiment creation/resume have completed, so it is never
    partially usable. All API wrappers share one :class:`LitRestClient`.

    Args:
        client: The shared REST client used by every API wrapper.
        metrics_api: API wrapper for metrics-stream operations.
        media_api: API wrapper for media uploads and listings.
        artifacts_api: API wrapper for file-artifact transfer.
        teamspace: The resolved teamspace this experiment logs to.
        experiment: The owning experiment. Also handed to the model registry as
            its ``experiment=`` linkage argument; primitives must not use it for
            anything else.
        queue: The joinable queue consumed by the background worker.
        stats: Run statistics updated as writes complete.
        store_step: Whether metric writes persist their step.
        store_created_at: Whether metric writes persist a timestamp.
        last_steps: Per-metric last-step mapping. This is the same object the
            background worker uses for auto-stepping (shared on purpose).
        background: The background worker thread, attached once constructed.
    """

    def __init__(
        self,
        *,
        client: LitRestClient,
        metrics_api: MetricsApi,
        media_api: MediaApi,
        artifacts_api: ArtifactsApi,
        teamspace: Teamspace,
        experiment: Experiment,
        queue: JoinableQueue[dict[str, Metrics]],
        stats: RunStats,
        store_step: bool,
        store_created_at: bool,
        last_steps: dict[str, int],
        background: _BackgroundThread | None = None,
    ) -> None:
        self.client = client
        self.metrics_api = metrics_api
        self.media_api = media_api
        self.artifacts_api = artifacts_api
        self.teamspace = teamspace
        self.experiment = experiment
        self.queue = queue
        self.stats = stats
        self.store_step = store_step
        self.store_created_at = store_created_at
        self.last_steps = last_steps
        # Serializes synchronous auto-stepping; the background worker keeps its
        # own single-threaded access to last_steps.
        self.last_steps_lock = threading.Lock()
        self.background = background

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> ExperimentSession:
        """Build a session from an experiment's already-initialized infrastructure."""
        return cls(
            client=experiment._metrics_api.client,
            metrics_api=experiment._metrics_api,
            media_api=experiment._media_api,
            artifacts_api=experiment._artifacts_api,
            teamspace=experiment._teamspace,
            experiment=experiment,
            queue=experiment._metrics_queue,
            stats=experiment._stats,
            store_step=bool(experiment.store_step),
            store_created_at=bool(experiment.store_created_at),
            last_steps=experiment._resumed_steps,
            background=getattr(experiment, "_manager", None),
        )

    @property
    def experiment_name(self) -> str:
        """The experiment name, doubling as the remote key prefix."""
        return self.experiment.name

    @property
    def teamspace_id(self) -> str:
        """The id of the teamspace this experiment logs to."""
        return str(self.teamspace.id)

    @property
    def metrics_store(self) -> Any:
        """The remote metrics-stream object (``V1MetricsStream``).

        Read through the experiment so refreshes are visible everywhere; the
        experiment attribute is the single mutable slot.
        """
        return self.experiment._metrics_store

    @property
    def metrics_store_id(self) -> str:
        """The id of the remote metrics stream."""
        return str(self.metrics_store.id)

    def refresh_metrics_store(self) -> None:
        """Re-fetch the metrics stream by name, keeping the current one on a miss."""
        resp = self.metrics_api.get_experiment_metrics_by_name(
            self.teamspace.id,
            name=self.metrics_store.name,
        )
        if resp is not None:
            self.experiment._metrics_store = resp

    def raise_if_background_failed(self) -> None:
        """Re-raise an exception captured by the background worker, if any."""
        background = self.background
        if background is not None and background.exception is not None:
            raise background.exception

    def flush(self) -> None:
        """Block until every queued write has been processed, then surface failures."""
        self.queue.join()
        self.raise_if_background_failed()
