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
"""User-facing data models for metrics and experiments.

These classes provide a clean interface that is independent of the Lightning SDK
implementation details (V1* classes).
"""

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PhaseType(str, Enum):
    """Phase of a metrics store lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class MediaType(str, Enum):
    """Type of media to upload."""

    IMAGE = "image"
    TEXT = "text"
    FILE = "file"
    MODEL = "model"
    VIDEO = "video"


@dataclass(init=False)
class MetricValue:
    """A single metric value with an optional x-coordinate and timestamp.

    Attributes:
        value: The numeric metric value.
        x: Optional numeric x-coordinate. The API serializer maps it to the
            backend's legacy ``step`` field.
        created_at: Optional datetime when this value was created.
    """

    value: float
    x: float | None
    created_at: datetime | None

    def __init__(
        self,
        value: float,
        step: float | None = None,
        created_at: datetime | None = None,
        *,
        x: float | None = None,
    ) -> None:
        if x is not None and step is not None:
            raise ValueError("x and step are mutually exclusive.")
        self.value = value
        self.x = x if x is not None else step
        self.created_at = created_at

    @property
    def step(self) -> float | None:
        """Legacy alias for the x-coordinate."""
        return self.x

    @step.setter
    def step(self, value: float | None) -> None:
        self.x = value


@dataclass
class Metrics:
    """A collection of metric values for a named metric.

    Attributes:
        name: The metric name (e.g., "loss", "accuracy").
        values: List of metric values.
    """

    name: str
    values: list[MetricValue] = field(default_factory=list)


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate statistics for a metric series.

    Attributes:
        name: The metric (series) name.
        count: Number of observations.
        min: Minimum observed value.
        max: Maximum observed value.
        mean: Arithmetic mean of observed values.
        std: Population standard deviation of observed values.
        median: Median of observed values.
        last: Most recently appended value.
        first: First appended value.
    """

    name: str
    count: int
    min: float
    max: float
    mean: float
    std: float
    median: float
    last: float
    first: float

    @classmethod
    def from_values(cls, name: str, values: list[float]) -> MetricSummary:
        """Construct a summary from a name and a non-empty list of values.

        Args:
            name: The metric (series) name.
            values: Non-empty list of observed float values.

        Raises:
            ValueError: If *values* is empty.
        """
        if not values:
            raise ValueError(f"Cannot compute summary for empty values (metric {name!r})")
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return cls(
            name=name,
            count=n,
            min=min(values),
            max=max(values),
            mean=mean,
            std=math.sqrt(variance),
            median=statistics.median(values),
            last=values[-1],
            first=values[0],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation of the summary."""
        return {
            "name": self.name,
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "last": self.last,
            "first": self.first,
        }
