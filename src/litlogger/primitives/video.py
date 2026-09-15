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
"""Video logging primitive."""

import os
import tempfile
from importlib import import_module
from typing import Any

from typing_extensions import override

from litlogger.primitives._media import _MediaFile
from litlogger.types import MediaType


class Video(_MediaFile):
    DEFAULT_FPS = 24

    """Represents a video to be logged.

    Accepts:
    - a file path string
    - a MoviePy clip
    - a numpy array / torch tensor of frames

    Supported array shapes:
    - (T, H, W)            grayscale frames
    - (T, H, W, C)         frames in HWC format
    - (T, C, H, W)         frames in CHW format, where C is 1/3/4

    Args:
        data: Video data or a path to a video file.
        format: Output container/extension, default "mp4".
        description: Optional description.
        fps: Frames per second used when rendering arrays or clips that do
            not already carry fps metadata.
    """

    def __init__(
        self,
        data: Any,
        format: str = "mp4",  # noqa: A002
        description: str = "",
        fps: float | None = None,
    ) -> None:
        self._data = data
        self._format = format
        self._fps = fps
        self._temp_path: str | None = None

        if isinstance(data, str):
            super().__init__(data, description=description)
        else:
            super().__init__("", description=description)

    def _get_upload_path(self) -> str:
        if isinstance(self._data, str):
            return super()._get_upload_path()
        return self._render_to_temp()

    def _render_to_temp(self) -> str:
        suffix = f".{self._format.lower()}"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self._temp_path = path

        data = self._data

        # torch.Tensor -> numpy
        try:
            import torch

            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except ImportError:
            pass

        # 1) MoviePy clip
        clip = self._maybe_as_moviepy_clip(data)
        if clip is not None:
            fps = self._fps or getattr(clip, "fps", None) or self.DEFAULT_FPS
            self._write_moviepy_clip(clip, path, fps)
            return path

        # 2) numpy array of frames
        try:
            np = import_module("numpy")

            if isinstance(data, np.ndarray):
                clip = self._moviepy_clip_from_array(data, fps=self._fps or self.DEFAULT_FPS)
                self._write_moviepy_clip(clip, path, self._fps or self.DEFAULT_FPS)
                return path

        except ImportError:
            pass

        raise TypeError(f"Unsupported video type: {type(data).__name__}")

    def _maybe_as_moviepy_clip(self, data: Any) -> None | Any:
        """Return data if it looks like a MoviePy clip, else None."""
        try:
            # MoviePy 2.x layout
            video_clip_mod = import_module("moviepy.video.VideoClip")
            video_clip = video_clip_mod.VideoClip
            if isinstance(data, video_clip):
                return data
        except Exception:
            pass

        try:
            # Older common import path
            editor_mod = import_module("moviepy.editor")
            video_clip = editor_mod.VideoClip
            if isinstance(data, video_clip):
                return data
        except Exception:
            pass

        return None

    def _moviepy_clip_from_array(self, data: Any, fps: float) -> Any:
        np = import_module("numpy")

        if data.dtype != np.uint8:
            if np.issubdtype(data.dtype, np.floating):
                # common case: [0,1] floats
                if data.size and data.max() <= 1.0:
                    data = (data * 255).clip(0, 255).astype(np.uint8)
                else:
                    data = data.clip(0, 255).astype(np.uint8)
            else:
                data = data.clip(0, 255).astype(np.uint8)

        if data.ndim not in (3, 4):
            raise ValueError(
                f"Unsupported array shape for video: {data.shape}. Expected (T,H,W), (T,H,W,C), or (T,C,H,W)."
            )

        # (T, H, W) -> grayscale => expand to (T, H, W, 1)
        if data.ndim == 3:
            data = data[..., None]

        # Handle TCHW -> THWC when channel axis is second
        if data.ndim == 4 and data.shape[1] in (1, 3, 4) and data.shape[-1] not in (1, 3, 4):
            data = data.transpose(0, 2, 3, 1)

        if data.shape[-1] == 1:
            # MoviePy ImageSequenceClip works best with 2D or RGB arrays;
            # convert grayscale to RGB for consistency.
            data = np.repeat(data, 3, axis=-1)

        if data.shape[-1] not in (3, 4):
            raise ValueError(f"Unsupported channel count for video frames: {data.shape[-1]}")

        # Drop alpha for now unless you explicitly want to preserve/use masks.
        if data.shape[-1] == 4:
            data = data[..., :3]

        try:
            # MoviePy 2.x
            image_sequence_clip = import_module("moviepy.video.io.ImageSequenceClip").ImageSequenceClip
        except Exception:
            # Older common path
            image_sequence_clip = import_module("moviepy.editor").ImageSequenceClip

        # list(...) avoids some ndarray edge cases in callers and matches
        # common usage for frame sequences.
        return image_sequence_clip(list(data), fps=fps)

    def _write_moviepy_clip(self, clip: Any, path: str, fps: float) -> None:
        # For MP4, libx264 is the usual sensible default.
        kwargs: dict[str, Any] = {
            "fps": fps,
            "logger": None,
        }

        ext = os.path.splitext(path)[1].lower()
        if ext == ".mp4":
            kwargs["codec"] = "libx264"
            # Better for browser progressive playback.
            kwargs["ffmpeg_params"] = ["-movflags", "+faststart"]

        clip.write_videofile(path, **kwargs)

    @property
    @override
    def _media_type(self) -> MediaType:
        return MediaType.VIDEO
