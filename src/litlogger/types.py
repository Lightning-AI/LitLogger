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

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List


class PhaseType(str, Enum):
    """Phase of a metrics store lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class MetricValue:
    """A single metric value with optional step and timestamp.

    Attributes:
        value: The numeric metric value.
        step: Optional step number for this value.
        created_at: Optional datetime when this value was created.
    """

    value: float
    step: int | None = None
    created_at: datetime | None = None


@dataclass
class Metrics:
    """A collection of metric values for a named metric.

    Attributes:
        name: The metric name (e.g., "loss", "accuracy").
        values: List of metric values.
    """

    name: str
    values: List[MetricValue] = field(default_factory=list)


@dataclass
class MetricsTracker:
    """Tracks statistics about a metric series.

    Attributes:
        name: The metric name.
        num_rows: Total number of values logged.
        min_value: Minimum value observed.
        max_value: Maximum value observed.
        min_index: Index where minimum value occurred.
        max_index: Index where maximum value occurred.
        last_value: Most recent value.
        last_index: Index of most recent value.
        started_at: Datetime when tracking started.
        updated_at: Datetime of last update.
        max_user_step: Maximum user-provided step value.
    """

    name: str
    num_rows: int = 0
    min_value: float | None = None
    max_value: float | None = None
    min_index: int | None = None
    max_index: int | None = None
    last_value: float | None = None
    last_index: int | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    max_user_step: int | None = None
