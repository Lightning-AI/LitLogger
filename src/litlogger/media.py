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
"""Media types for litlogger.

These classes represent files and media objects that can be logged to experiments.
File wraps a local path, while other media objects can accept Python objects and render
them to temporary files for upload.
"""

import os
import tempfile
from typing import Any, override

from litlogger.types import MediaType


class File:
    """Represents a file to be logged to the experiment.

    Args:
        path: Path to the local file.
        description: Optional human-readable description of the file.
    """

    def __init__(self, path: str, description: str = "") -> None:
        self.path = path
        self.name: str = ""
        self.description = description
        self._temp_path: str | None = None
        self._download_fn: Any | None = None

    def _get_upload_path(self) -> str:
        """Get a stable path for upload.

        Creates a hardlink to a temp location so the original file can be
        safely modified or deleted while a background upload is in progress.
        Falls back to a copy if hardlinking is not supported, or returns the
        original path if the file doesn't exist yet.
        """
        if not self.path or not os.path.exists(self.path):
            return self.path
        try:
            suffix = os.path.splitext(self.path)[1]
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            os.unlink(tmp)
            os.link(self.path, tmp)
            self._temp_path = tmp
            return tmp
        except OSError:
            import shutil

            suffix = os.path.splitext(self.path)[1]
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            shutil.copy2(self.path, tmp)
            self._temp_path = tmp
            return tmp

    def _cleanup(self) -> None:
        """Clean up any temporary files created during upload."""
        if self._temp_path is not None and os.path.exists(self._temp_path):
            os.unlink(self._temp_path)
            self._temp_path = None

    def save(self, path: str) -> str:
        """Download the remote file to a local path.

        Only works for files that have been uploaded to an experiment.

        Args:
            path: Local path where the file should be saved.

        Returns:
            str: The local path where the file was saved.

        Raises:
            RuntimeError: If the file has no remote download context.
        """
        if self._download_fn is None:
            raise RuntimeError("File has no remote context. It must be uploaded to an experiment first.")
        return self._download_fn(path)

    @property
    def _media_type(self) -> MediaType:
        return MediaType.FILE

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({self.path!r})"

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, File):
            return NotImplemented
        return type(self) is type(other) and self.path == other.path

    def __hash__(self) -> int:  # noqa: D105
        return hash((type(self), self.path))


class Image(File):
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
            import numpy as np

            if isinstance(data, np.ndarray):
                from PIL import Image as PILImage

                if data.dtype != np.uint8:
                    data = (data * 255).astype(np.uint8) if data.max() <= 1.0 else data.astype(np.uint8)

                if data.ndim == 2:
                    img = PILImage.fromarray(data)
                elif data.ndim == 3:
                    # Handle CHW -> HWC format
                    if data.shape[0] in (1, 3, 4) and data.shape[2] not in (1, 3, 4):
                        data = data.transpose(1, 2, 0)
                    if data.shape[2] == 1:
                        data = data.squeeze(2)
                    img = PILImage.fromarray(data)
                else:
                    raise ValueError(f"Unsupported array shape for image: {data.shape}")

        except ImportError:
            pass

        # Handle PIL Image
        try:
            from PIL import Image as PILImage

            if isinstance(data, PILImage.Image):
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


class Text(File):
    """Represents text content to be logged.

    Takes a string and writes it to a temporary file for upload.

    Args:
        content: The text string to log.
        description: Optional human-readable description of the text.
    """

    _media_type: str = "text"

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
        with open(path, "w") as f:
            f.write(self._content)
        self.path = path
        return path

    @property
    def _media_type(self) -> MediaType:
        return MediaType.TEXT
