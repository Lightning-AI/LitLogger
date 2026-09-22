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
"""User-facing logging primitives.

Every kind of loggable state is represented by a primitive that owns its
serialization, remote write, and restore behavior. A primitive can write
synchronously (``log``) or hand itself to the experiment's background pipeline
(``enqueue``); both operate against a shared
:class:`~litlogger.session.ExperimentSession` instead of constructing clients
or API wrappers of their own.

Each concrete primitive lives in its own module in this package. The legacy
:mod:`litlogger.media` module re-exports the file-like primitives for backwards
compatibility.
"""

from litlogger.primitives._utils import SERIES_NAME_RE as SERIES_NAME_RE
from litlogger.primitives._utils import RestoredFiles as RestoredFiles
from litlogger.primitives._utils import _to_v1_media_type as _to_v1_media_type
from litlogger.primitives._utils import model_version_sort_key as model_version_sort_key
from litlogger.primitives._utils import natural_sort_key as natural_sort_key
from litlogger.primitives._utils import sanitize_model_key as sanitize_model_key
from litlogger.primitives.file import File as File
from litlogger.primitives.image import Image as Image
from litlogger.primitives.metadata import Metadata as Metadata
from litlogger.primitives.metric import Metric as Metric
from litlogger.primitives.model import Model as Model
from litlogger.primitives.primitive import MetricWrite as MetricWrite
from litlogger.primitives.primitive import Primitive as Primitive
from litlogger.primitives.primitive import PrimitiveWrite as PrimitiveWrite
from litlogger.primitives.primitive import QueueItem as QueueItem
from litlogger.primitives.primitive import WritePlacement as WritePlacement
from litlogger.primitives.primitive import _enqueue_write as _enqueue_write
from litlogger.primitives.text import Text as Text
from litlogger.primitives.video import Video as Video
