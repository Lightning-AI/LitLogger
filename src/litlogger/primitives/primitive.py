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
"""Primitive contract and immutable commands for the write pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


@dataclass(frozen=True)
class WritePlacement:
    """Immutable location of a primitive within an experiment."""

    key: str
    index: int | None = None
    x: float | None = None

    @property
    def is_series(self) -> bool:
        """Whether this placement identifies one element of a series."""
        return self.index is not None


@runtime_checkable
class Primitive(Protocol):
    """A value that can write synchronously or submit an immutable command."""

    def log(self, session: ExperimentSession, placement: WritePlacement | None = None) -> None:
        """Perform this primitive's remote write immediately."""
        ...

    def enqueue(self, session: ExperimentSession, placement: WritePlacement | None = None) -> None:
        """Submit this primitive to the session's background pipeline."""
        ...


@dataclass(frozen=True)
class MetricWrite:
    """One metric observation awaiting coordinate resolution and batching."""

    key: str
    y: float
    x: float | None
    created_at: datetime | None


@dataclass(frozen=True)
class PrimitiveWrite:
    """A self-contained sequential write operation."""

    operation: Callable[[ExperimentSession], None]

    def execute(self, session: ExperimentSession) -> None:
        """Execute the captured operation against its owning session."""
        self.operation(session)


QueueItem: TypeAlias = MetricWrite | PrimitiveWrite


def _enqueue_write(operation: Callable[[ExperimentSession], None], session: ExperimentSession) -> None:
    """Submit a self-contained primitive operation through the session."""
    session.submit(PrimitiveWrite(operation))
