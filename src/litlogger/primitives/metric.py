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

from litlogger.primitives.primitive import MetricWrite, WritePlacement
from litlogger.types import Metrics, MetricValue

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


def resolve_x(*, x: float | None = None, step: float | None = None) -> float | None:
    """Normalize the preferred and legacy coordinate arguments."""
    if x is not None and step is not None:
        raise ValueError("x and step are mutually exclusive.")
    coordinate = x if x is not None else step
    if coordinate is not None and not math.isfinite(coordinate):
        raise ValueError("x must be finite.")
    return coordinate


@dataclass(frozen=True, init=False)
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
    x: float | None

    def __init__(
        self,
        key: str,
        y: float,
        step: float | None = None,
        x: float | None = None,
    ) -> None:
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "x", resolve_x(x=x, step=step))

    @property
    def step(self) -> float | None:
        """Legacy alias for the normalized x-coordinate."""
        return self.x

    def log(self, session: ExperimentSession, placement: WritePlacement | None = None) -> None:
        """Append this observation synchronously, bypassing the background batcher.

        Auto-stepping mirrors the background worker: a missing step receives
        the next value from the shared per-series sequence.
        """
        key = placement.key if placement is not None else self.key
        supplied_x = placement.x if placement is not None else self.x
        created_at = datetime.now() if session.store_created_at else None
        x = session.resolve_x(key, supplied_x)
        session.metrics_api.append_experiment_metrics(
            teamspace_id=session.teamspace.id,
            metrics_store_id=session.metrics_store.id,
            metrics=[
                Metrics(
                    name=key,
                    values=[MetricValue(value=self.y, created_at=created_at, x=x if session.store_step else None)],
                )
            ],
        )
        session.stats.record_metric(key, self.y)

    def enqueue(self, session: ExperimentSession, placement: WritePlacement | None = None) -> None:
        """Queue this observation for the background batcher (the default write path)."""
        key = placement.key if placement is not None else self.key
        x = placement.x if placement is not None else self.x
        created_at = datetime.now() if session.store_created_at else None
        session.submit(MetricWrite(key=key, y=self.y, x=x, created_at=created_at))
        session.stats.record_metric(key, self.y)

    @staticmethod
    def _restore_values(session: ExperimentSession) -> dict[str, list[float]]:
        """Fetch every remote metric series' values for resume."""
        return session.metrics_api.get_metric_values(session.teamspace.id, session.metrics_store.id)
