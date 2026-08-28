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
"""Metric logging primitive."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from litlogger.types import Metrics, MetricValue

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


@dataclass
class Metric:
    """A single metric observation for a named series.

    Args:
        key: The metric (series) name.
        y: The observed value.
        step: Legacy x-coordinate for this observation.
        x: Preferred x-coordinate. Mutually exclusive with ``step``. When both
            are omitted, the next coordinate is assigned at write time.
    """

    key: str
    y: float
    step: float | None = None
    x: float | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous coordinates before the metric can be queued."""
        if self.x is not None and self.step is not None:
            raise ValueError("x and step are mutually exclusive.")
        coordinate = self.x if self.x is not None else self.step
        if coordinate is not None and not math.isfinite(coordinate):
            raise ValueError("x must be finite.")

    def _x(self, store_step: bool) -> float | None:
        """Resolve the coordinate that is serialized through the legacy step field."""
        if not store_step:
            return None
        return self.x if self.x is not None else self.step

    def log(self, session: ExperimentSession) -> None:
        """Append this observation synchronously, bypassing the background batcher.

        Auto-stepping mirrors the background worker: a missing step receives
        the next value from the shared per-series sequence.
        """
        created_at = datetime.now() if session.store_created_at else None
        x = self._x(session.store_step)
        with session.last_steps_lock:
            if x is None:
                x = session.last_steps.get(self.key, -1) + 1
            session.last_steps[self.key] = x
        session.metrics_api.append_experiment_metrics(
            teamspace_id=session.teamspace.id,
            metrics_store_id=session.metrics_store.id,
            metrics=[Metrics(name=self.key, values=[MetricValue(value=self.y, created_at=created_at, step=x)])],
        )
        session.stats.record_metric(self.key, self.y)

    def enqueue(self, session: ExperimentSession) -> None:
        """Queue this observation for the background batcher (the default write path)."""
        session.raise_if_background_failed()

        created_at = datetime.now() if session.store_created_at else None
        x = self._x(session.store_step)
        mv = MetricValue(value=self.y, created_at=created_at, step=x)
        batch: dict[str, Metrics] = {self.key: Metrics(name=self.key, values=[mv])}
        session.queue.put(batch)
        session.stats.record_metric(self.key, self.y)

    @staticmethod
    def _restore_values(session: ExperimentSession) -> dict[str, list[float]]:
        """Fetch every remote metric series' values for resume."""
        return session.metrics_api.get_metric_values(session.teamspace.id, session.metrics_store.id)
