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
"""Shared base class for rendered media primitives."""

from typing import TYPE_CHECKING

from typing_extensions import override

from litlogger.primitives._utils import _to_v1_media_type, series_storage_name, static_storage_name
from litlogger.primitives.file import File
from litlogger.primitives.primitive import WritePlacement

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


class _MediaFile(File):
    """Base for rendered media (images, videos, text) uploaded through the media API."""

    @override
    def log(self, session: "ExperimentSession", placement: WritePlacement | None = None) -> None:
        """Upload this media now, in the caller's thread.

        Media uploads share one remote name per key: series elements are
        differentiated by their step, not by an indexed path.
        """
        if placement is None:
            name = self.name or self._artifact_display_path(None)
            display_name = name
            x = None
        else:
            name = series_storage_name(placement.key) if placement.is_series else static_storage_name(placement.key)
            display_name = placement.key
            x = placement.x
        try:
            upload_path = self._get_upload_path()
            session.media_api.upload_media(
                experiment_id=session.metrics_store.id,
                teamspace=session.teamspace,
                file_path=upload_path,
                name=name,
                media_type=_to_v1_media_type(self._media_type),
                step=x,
            )
        finally:
            self._cleanup()
        self.name = display_name
        session.stats.media_logged += 1
