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
"""API layer for media upload operations."""

import mimetypes
import os

from blake3 import blake3
from lightning_sdk import Teamspace
from lightning_sdk.lightning_cloud.openapi import (
    LitLoggerServiceCreateLitLoggerMediaBody,
    LitLoggerServiceUpdateLitLoggerMediaBody,
    V1CreateLitLoggerMediaResponse,
    V1LitLoggerMedia,
    V1MediaType,
)

from litlogger.api.client import LitRestClient


def _compute_file_hash(file_path: str) -> str:
    """Compute BLAKE3 hash of a file."""
    file_hasher = blake3(max_threads=blake3.AUTO)
    file_hasher.update_mmap(file_path)
    return file_hasher.hexdigest()


class MediaApi:
    """API layer for uploading media to experiments."""

    def __init__(self, client: LitRestClient | None = None) -> None:
        """Initialize the MediaApi.

        Args:
            client: Optional pre-configured LitRestClient. If None, creates a new one.
        """
        self.client = client or LitRestClient(max_retries=5)

    def upload_media(
        self,
        experiment_id: str,
        teamspace: Teamspace,
        file_path: str,
        name: str,
        media_type: V1MediaType,
        step: int | None = None,
        epoch: int | None = None,
        caption: str | None = None,
    ) -> V1LitLoggerMedia:
        """Upload media to an experiment.

        Args:
            experiment_id: The metrics stream ID (experiment ID).
            teamspace: Teamspace object for the experiment.
            file_path: Local path to the file to upload.
            name: Name for the media.
            media_type: Type of media (image, audio, video, etc.).
            step: Optional training step.
            epoch: Optional training epoch.
            caption: Optional caption for the media.

        Returns:
            The created media object.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        content_hash = _compute_file_hash(file_path)

        mime_type, _ = mimetypes.guess_type(file_path)

        create_body = LitLoggerServiceCreateLitLoggerMediaBody(
            name=name,
            media_type=media_type,
            size_bytes=str(file_size),
            content_hash=content_hash,
            step=step,
            epoch=epoch,
            caption=caption,
            mime_type=mime_type,
        )

        create_response: V1CreateLitLoggerMediaResponse = self.client.lit_logger_service_create_lit_logger_media(
            body=create_body,
            project_id=teamspace.id,
            metrics_stream_id=experiment_id,
        )

        media = create_response.media

        if create_response.already_exists:
            return media

        headers = {"Content-Type": mime_type} if mime_type else None
        teamspace._teamspace_api.upload_file(
            teamspace_id=teamspace.id,
            cloud_account=teamspace.default_cloud_account,
            file_path=file_path,
            remote_path=media.storage_path,
            headers=headers,
            progress_bar=False,
        )

        update_body = LitLoggerServiceUpdateLitLoggerMediaBody(upload_complete=True)

        return self.client.lit_logger_service_update_lit_logger_media(
            body=update_body,
            project_id=teamspace.id,
            metrics_stream_id=experiment_id,
            id=media.id,
        )
