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
"""Image logging primitive."""

import os
import tempfile
from importlib import import_module
from typing import Any

from typing_extensions import override

from litlogger.primitives._media import _MediaFile
from litlogger.types import MediaType


class Image(_MediaFile):
    """Represents an image to be logged.

    Can take a file path (str) or a Python object (PIL Image, numpy array,
    or torch Tensor) and will render it to a temporary file for upload.

    Args:
        data: The image data - a file path string, PIL Image, numpy array, or torch Tensor.
        format: Image format for rendering objects to disk (default: "png").
        description: Optional human-readable description of the image.
    """

    def __init__(self, data: Any, format: str = "png", description: str = "") -> None:  # noqa: A002
        self._data = data
        self._format = format
        if isinstance(data, str):
            super().__init__(data, description=description)
        else:
            super().__init__("", description=description)

    def _get_upload_path(self) -> str:
        if isinstance(self._data, str):
            return super()._get_upload_path()
        return self._render_to_temp()

    def _render_to_temp(self) -> str:
        """Render the image data to a temporary file."""
        suffix = f".{self._format.lower()}"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self._temp_path = path

        data = self._data
        img = None

        # Handle torch.Tensor -> numpy
        try:
            import torch

            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except ImportError:
            pass

        # Handle numpy array
        try:
            np = import_module("numpy")

            if isinstance(data, np.ndarray):
                pil_image = import_module("PIL.Image")

                if data.dtype != np.uint8:
                    data = (data * 255).astype(np.uint8) if data.max() <= 1.0 else data.astype(np.uint8)

                if data.ndim == 2:
                    img = pil_image.fromarray(data)
                elif data.ndim == 3:
                    # Handle CHW -> HWC format
                    if data.shape[0] in (1, 3, 4) and data.shape[2] not in (1, 3, 4):
                        data = data.transpose(1, 2, 0)
                    if data.shape[2] == 1:
                        data = data.squeeze(2)
                    img = pil_image.fromarray(data)
                else:
                    raise ValueError(f"Unsupported array shape for image: {data.shape}")

        except ImportError:
            pass

        # Handle PIL Image
        try:
            pil_image = import_module("PIL.Image")

            if isinstance(data, pil_image.Image):
                img = data

        except ImportError:
            pass

        # if valid image type was passed, save it and return
        if img is not None:
            img.save(self._temp_path)
            return self._temp_path

        raise TypeError(f"Unsupported image type: {type(data).__name__}")

    @property
    @override
    def _media_type(self) -> MediaType:
        return MediaType.IMAGE
