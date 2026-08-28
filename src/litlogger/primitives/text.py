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
"""Text logging primitive."""

import os
import tempfile

from typing_extensions import override

from litlogger.primitives._media import _MediaFile
from litlogger.types import MediaType


class Text(_MediaFile):
    """Represents text content to be logged.

    Takes a string and writes it to a temporary file for upload.

    Args:
        content: The text string to log.
        description: Optional human-readable description of the text.
    """

    def __init__(self, content: str, description: str = "") -> None:
        self._content = content
        super().__init__("", description=description)

    def _get_upload_path(self) -> str:
        if self.path and os.path.exists(self.path):
            return super()._get_upload_path()
        return self._render_to_temp()

    def _render_to_temp(self) -> str:
        """Write text content to a temporary file."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        self._temp_path = path
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._content)
        self.path = path
        return path

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}('{self.path}')"

    @property
    @override
    def _media_type(self) -> MediaType:
        return MediaType.TEXT
