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
"""User-facing logging primitives.

Every kind of loggable state is represented by a primitive that owns its
serialization, remote write, and restore behavior. A primitive can write
synchronously (``log``) or hand itself to the experiment's background pipeline
(``enqueue``); both operate against a shared
:class:`~litlogger.session.ExperimentSession` instead of constructing clients
or API wrappers of their own.

The file-like primitives (:class:`~litlogger.media.File` and its subclasses)
live in :mod:`litlogger.media`; this module hosts the contract plus the value
primitives :class:`Metric` and :class:`Metadata`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from litlogger.types import Metrics, MetricValue, PhaseType

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


@runtime_checkable
class Primitive(Protocol):
    """A loggable unit: a synchronous write plus an asynchronous hand-off.

    ``log`` performs the remote write in the caller's thread. ``enqueue`` hands
    the primitive to the experiment's background pipeline, which batches where
    it can (metrics) and otherwise performs the same write off-thread. Callers
    need not distinguish between the two mechanisms beyond that timing choice.
    """

    def log(self, session: ExperimentSession) -> None:
        """Perform this primitive's remote write now, in the caller's thread."""
        ...

    def enqueue(self, session: ExperimentSession) -> None:
        """Hand this primitive to the background pipeline for asynchronous processing."""
        ...


@dataclass
class _QueuedWrite:
    """Envelope for a non-batchable primitive handed to the background worker."""

    primitive: Primitive


def _enqueue_write(primitive: Primitive, session: ExperimentSession) -> None:
    """Queue a primitive for the background worker, surfacing prior failures first."""
    session.raise_if_background_failed()
    # The queue's item type broadens to include _QueuedWrite when the
    # background worker learns to execute queued primitives (async flip).
    session.queue.put(_QueuedWrite(primitive))  # type: ignore[arg-type]


@dataclass
class Metric:
    """A single metric observation for a named series.

    Args:
        key: The metric (series) name.
        value: The observed value.
        step: Optional step for this observation. When omitted, the next step
            in the series' sequence is assigned at write time.
    """

    key: str
    value: float
    step: int | None = None

    def log(self, session: ExperimentSession) -> None:
        """Append this observation synchronously, bypassing the background batcher.

        Auto-stepping mirrors the background worker: a missing step receives
        the next value from the shared per-series sequence.
        """
        created_at = datetime.now() if session.store_created_at else None
        step = self.step if session.store_step else None
        with session.last_steps_lock:
            if step is None:
                step = session.last_steps.get(self.key, -1) + 1
                session.last_steps[self.key] = step
        session.metrics_api.append_experiment_metrics(
            teamspace_id=session.teamspace.id,
            metrics_store_id=session.metrics_store.id,
            metrics=[Metrics(name=self.key, values=[MetricValue(value=self.value, created_at=created_at, step=step)])],
        )
        session.stats.record_metric(self.key, self.value)

    def enqueue(self, session: ExperimentSession) -> None:
        """Queue this observation for the background batcher (the default write path)."""
        session.raise_if_background_failed()

        created_at = datetime.now() if session.store_created_at else None
        actual_step = self.step if session.store_step else None
        mv = MetricValue(value=self.value, created_at=created_at, step=actual_step)
        batch: dict[str, Metrics] = {self.key: Metrics(name=self.key, values=[mv])}
        session.queue.put(batch)
        session.stats.record_metric(self.key, self.value)

    @staticmethod
    def _restore_values(session: ExperimentSession) -> dict[str, list[float]]:
        """Fetch every remote metric series' values for resume."""
        return session.metrics_api.get_metric_values(session.teamspace.id, session.metrics_store.id)


@dataclass
class Metadata:
    """A single metadata entry (code tag) on the experiment.

    Args:
        key: The metadata key.
        value: The metadata value.
    """

    key: str
    value: str

    def log(self, session: ExperimentSession) -> None:
        """Write this entry by read-modify-writing the experiment's full code-tag set."""
        current_tags = self._current_tags(session)
        current_tags[self.key] = self.value
        session.metrics_api.update_experiment_metrics(
            teamspace_id=session.teamspace.id,
            metrics_store_id=session.metrics_store.id,
            phase=PhaseType.RUNNING,
            metadata=current_tags,
        )

    def enqueue(self, session: ExperimentSession) -> None:
        """Queue this entry; the background worker performs the read-modify-write."""
        _enqueue_write(self, session)

    @staticmethod
    def _current_tags(session: ExperimentSession) -> dict[str, str]:
        """Read the experiment's code tags from a freshly refreshed metrics stream."""
        session.refresh_metrics_store()
        tags = getattr(session.metrics_store, "tags", None) or []
        return {tag.name: tag.value for tag in tags if tag.from_code}
