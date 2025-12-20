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
import os
from typing import Any

from lightning_sdk import Teamspace
from lightning_sdk.api.utils import _FileUploader
from lightning_sdk.lightning_cloud.openapi import LitLoggerServiceCreateLoggerArtifactBody

from litlogger.api.client import LitRestClient


class ArtifactsApi:
    """API layer for artifact upload and download operations."""

    def __init__(self, client: LitRestClient | None = None) -> None:
        """Initialize the ArtifactsApi.

        Args:
            client: Optional pre-configured LitRestClient. If None, creates a new one.
        """
        self.client = client or LitRestClient(max_retries=5)

    def upload_experiment_file_artifact(
        self,
        teamspace: Teamspace,
        metrics_store: Any,
        experiment_name: str,
        file_path: str,
        remote_path: str,
    ) -> None:
        """Upload a file as an artifact to the teamspace drive.

        Args:
            teamspace: Teamspace object where the file will be uploaded.
            metrics_store: Metrics store object containing the stream ID.
            experiment_name: Experiment name for organizing artifacts.
            file_path: Local path to the file to upload.
            remote_path: Path relative to experiment root for storage and display.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"file not found: {file_path}")

        # Upload to teamspace drive under experiments folder
        full_remote_path = f"experiments/{experiment_name}/{remote_path}"

        teamspace.upload_file(file_path=file_path, remote_path=full_remote_path, progress_bar=False)

        # Register the artifact with the metrics stream
        self.client.lit_logger_service_create_logger_artifact(
            project_id=teamspace.id,
            metrics_stream_id=metrics_store.id,
            body=LitLoggerServiceCreateLoggerArtifactBody(path=remote_path),
        )

    def download_experiment_file_artifact(
        self,
        teamspace: Teamspace,
        experiment_name: str,
        filename: str,
        local_path: str | None = None,
    ) -> str:
        """Download a file artifact from the teamspace drive.

        Args:
            teamspace: Teamspace object where the file is stored.
            experiment_name: Experiment name where the artifact was uploaded.
            filename: Name of the file to download.
            local_path: Optional local path where the file should be saved.
                       If None, saves to current directory with the same filename.

        Raises:
            FileNotFoundError: If the remote file doesn't exist.
        """
        # Construct the remote path
        remote_path = f"experiments/{experiment_name}/{filename}"

        # Determine local save path
        if local_path is None:
            local_path = filename
        elif os.path.isdir(local_path):
            local_path = os.path.join(local_path, filename)

        # Convert to absolute path for cross-platform compatibility (Windows needs this)
        local_path = os.path.abspath(local_path)

        # Create directory if needed
        local_dir = os.path.dirname(local_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        # Download from teamspace drive
        teamspace.download_file(remote_path=remote_path, file_path=local_path)

        return local_path

    def upload_file(
        self,
        teamspace: Teamspace,
        local_path: str,
        remote_path: str,
    ) -> str:
        """Upload a file to the teamspace drive (generic, not experiment-specific).

        Args:
            teamspace: Teamspace object where the file will be uploaded.
            local_path: Local path to the file to upload.
            remote_path: Remote path in the teamspace drive.

        Returns:
            str: The remote path where the file was uploaded.

        Raises:
            FileNotFoundError: If the local file doesn't exist.
        """
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        # Upload to teamspace drive
        teamspace.upload_file(file_path=local_path, remote_path=remote_path, progress_bar=False)

        return remote_path

    def download_file(
        self,
        teamspace: Teamspace,
        remote_path: str,
        local_path: str,
    ) -> str:
        """Download a file from the teamspace drive (generic, not experiment-specific).

        Args:
            teamspace: Teamspace object where the file is stored.
            remote_path: Remote path in the teamspace drive.
            local_path: Local path where the file should be saved.

        Returns:
            str: The local path where the file was saved.
        """
        # Convert to absolute path for cross-platform compatibility (Windows needs this)
        local_path = os.path.abspath(local_path)

        # Create directory if needed
        local_dir = os.path.dirname(local_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        # Download from teamspace drive
        teamspace.download_file(remote_path=remote_path, file_path=local_path)

        return local_path

    def upload_metrics_binary(
        self,
        teamspace_id: str,
        cloud_account: str,
        file_path: str,
        remote_path: str,
    ) -> None:
        """Upload a metrics binary tar.gz file to the teamspace.

        Args:
            teamspace_id: The teamspace ID.
            cloud_account: Cloud account identifier.
            file_path: Local path to the tar.gz file to upload.
            remote_path: Remote path where the file will be uploaded.
        """
        file_uploader = _FileUploader(
            client=self.client,
            teamspace_id=teamspace_id,
            cloud_account=cloud_account,
            file_path=file_path,
            remote_path=remote_path,
            progress_bar=False,
        )
        file_uploader()
