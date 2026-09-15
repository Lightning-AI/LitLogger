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
"""Compatibility imports for media primitives.

The concrete implementations live in :mod:`litlogger.primitives`. Imports
from this module remain supported for backwards compatibility.
"""

import warnings

from litlogger.primitives.file import File as File
from litlogger.primitives.file import _wrap_media_file as _wrap_media_file
from litlogger.primitives.image import Image as Image
from litlogger.primitives.model import Model as Model
from litlogger.primitives.model import _sanitize_version_for_model_name as _sanitize_version_for_model_name
from litlogger.primitives.text import Text as Text
from litlogger.primitives.video import Video as Video

warnings.warn(
    "litlogger.media is deprecated; import media primitives from litlogger.primitives instead.",
    DeprecationWarning,
    stacklevel=2,
)
