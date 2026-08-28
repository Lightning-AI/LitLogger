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
"""Private helpers shared by file and model primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from lightning_sdk.lightning_cloud.openapi import V1MediaType

from litlogger.types import MediaType

if TYPE_CHECKING:
    from litlogger.primitives.file import File


#: Remote names like ``reports/3`` are the 4th element of a file series keyed
#: by ``reports``; anything else is a static file.
SERIES_NAME_RE = re.compile(r"^(?P<key>.+)/(?P<index>\d+)$")


@dataclass
class RestoredFiles:
    """Result of one bulk file-restore pass: statics plus ordered series values."""

    statics: dict[str, File]
    series: dict[str, list[File]]


def sanitize_model_key(key: str) -> str:
    """Reduce an experiment key to the registry's allowed model-name alphabet."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", key).strip("-") or "model"


def natural_sort_key(value: str | None) -> tuple[tuple[int, int | str], ...]:
    """Sort key treating digit runs numerically (``v2`` before ``v10``)."""
    if not value:
        return ((1, ""),)
    parts = re.split(r"(\d+)", value)
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if not part:
            continue
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(key)


def model_version_sort_key(version_info: object) -> tuple[object, ...]:
    """Order model versions by index, then timestamps, then natural version name."""
    index = getattr(version_info, "index", None)
    if isinstance(index, int):
        return (0, index)

    created_at = getattr(version_info, "created_at", None)
    if isinstance(created_at, datetime):
        return (1, created_at)

    updated_at = getattr(version_info, "updated_at", None)
    if isinstance(updated_at, datetime):
        return (2, updated_at)

    version = getattr(version_info, "version", None)
    return (3, *natural_sort_key(version))


def _to_v1_media_type(media_type: MediaType) -> V1MediaType:
    """Map a user-facing media type to its V1 wire type."""
    if media_type == MediaType.IMAGE:
        return V1MediaType.IMAGE
    if media_type == MediaType.TEXT:
        return V1MediaType.TEXT
    if media_type == MediaType.VIDEO:
        return V1MediaType.VIDEO
    raise ValueError(f"Unsupported media type for file upload: {media_type}")
