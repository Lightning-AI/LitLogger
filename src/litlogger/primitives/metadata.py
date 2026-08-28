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
"""Metadata logging primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from litlogger.primitives.primitive import _enqueue_write
from litlogger.types import PhaseType

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


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
