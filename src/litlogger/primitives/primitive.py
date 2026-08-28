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
"""Contract and queue envelope shared by all logging primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from litlogger.types import Metrics

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


#: Items carried by the experiment queue: metric batches (merged and sent in
#: bulk) or queued primitives (executed one by one by the worker).
QueueItem: TypeAlias = "dict[str, Metrics] | _QueuedWrite"


def _enqueue_write(primitive: Primitive, session: ExperimentSession) -> None:
    """Queue a primitive for the background worker, surfacing prior failures first."""
    session.raise_if_background_failed()
    session.queue.put(_QueuedWrite(primitive))
