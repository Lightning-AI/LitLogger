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
"""Artifact and model management for litlogger.

Provides Artifact class for generic file uploads and Model class for model artifacts.
Model class uses litmodels for proper model handling and versioning.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Protocol, Union, runtime_checkable

from litmodels import download_model, load_model, save_model, upload_model

from litlogger.api.artifacts_api import ArtifactsApi
from litlogger.api.client import LitRestClient

if TYPE_CHECKING:
    import torch
    from lightning_sdk import Teamspace

# Re-export for backwards compatibility
__all__ = ["Artifact", "Model", "ModelArtifact", "upload_model", "download_model"]


def _sanitize_version_for_model_name(version: str) -> str:
    """Sanitize version string for use in model names.

    Model names follow the format: owner/teamspace/model:version
    If version contains colons, it will be confused with the name:version separator.
    Replace colons with hyphens to avoid this.
    """
    return version.replace(":", "-")


@runtime_checkable
class GenericArtifact(Protocol):
    """Protocol for artifact classes with log and get methods."""

    def log(self) -> None: ...
    def get(self) -> Any: ...


class Artifact:
    """Helper class for managing a file artifact in a teamspace.

    Represents a single artifact with its local location. The remote path
    is automatically derived from the local path and experiment name.
    Call log() to upload the file and register it with the experiment.

    Args:
        path: Path to the local file.
        experiment_name: Name of the experiment.
        teamspace: Teamspace object where the artifact is stored.
        metrics_store: Metrics store for registering artifacts in the UI.
        client: Optional pre-configured LitRestClient.
        remote_path: Path relative to experiment root for storage and display.
            If None, defaults to the basename of path.
    """

    def __init__(
        self,
        path: str,
        experiment_name: str,
        teamspace: "Teamspace",
        metrics_store: Any,
        client: LitRestClient | None = None,
        remote_path: str | None = None,
    ) -> None:
        self.path = path
        self.experiment_name = experiment_name
        self.teamspace = teamspace
        self.metrics_store = metrics_store
        self._api = ArtifactsApi(client=client or LitRestClient(max_retries=5))

        # Determine display_path:
        # 1. Use provided remote_path if given
        # 2. Try to compute relative path from cwd - use it if file is under cwd
        # 3. Fall back to basename
        if remote_path is not None:
            self.display_path = remote_path
        else:
            try:
                rel_path = os.path.relpath(path)
            except ValueError:
                # On Windows, relpath fails if path and cwd are on different drives
                rel_path = None
            # If relative path doesn't escape cwd (no leading ..), use it
            if rel_path is not None and not rel_path.startswith(".."):
                self.display_path = rel_path
            else:
                self.display_path = os.path.basename(path)
        # Normalize to forward slashes for consistent remote paths across platforms
        self.display_path = self.display_path.replace("\\", "/")
        self.remote_path = f"experiments/{experiment_name}/{self.display_path}"

    def log(self) -> None:
        """Upload the local file to the teamspace drive and register as artifact."""
        self._api.upload_experiment_file_artifact(
            teamspace=self.teamspace,
            metrics_store=self.metrics_store,
            experiment_name=self.experiment_name,
            file_path=self.path,
            remote_path=self.display_path,
        )

    def get(self) -> str:
        """Download the file from the teamspace drive to the local path.

        Returns:
            str: The local path where the file was saved.
        """
        return self._api.download_file(
            teamspace=self.teamspace,
            remote_path=self.remote_path,
            local_path=self.path,
        )


class ModelArtifact:
    """Helper class for managing model artifacts in a teamspace.

    Similar to Artifact but stores models under "models/" instead of "experiments/".
    Uses litmodels helpers if available for better model handling.

    Args:
        path: Path local model directory.
        experiment_name: Name of the experiment (used as the model name).
    """

    def __init__(
        self,
        path: str,
        experiment_name: str,
        teamspace: "Teamspace",
        version: str | None = None,
        verbose: bool = False,
        cloud_account: str | None = None,
        metadata: Dict[str, str] | None = None,
    ) -> None:
        self.path = path
        self.name = f"{teamspace.owner.name}/{teamspace.name}/{experiment_name}"
        if version:
            self.name += f":{_sanitize_version_for_model_name(version)}"
        self.verbose = verbose
        self.cloud_account = cloud_account
        self.metadata = metadata

    def log(self) -> None:
        """Upload the model using litmodels."""
        upload_model(
            name=self.name,
            model=self.path,
            verbose=False,
            progress_bar=self.verbose,
            cloud_account=self.cloud_account,
            metadata=self.metadata,
        )

    def get(self) -> str:
        """Download the model using litmodels.

        Returns:
            str: The local path where the model was saved.
        """
        result = download_model(name=self.name, download_dir=self.path)
        return result if isinstance(result, str) else result[0]


class Model:
    """Helper class for managing models in a teamspace."""

    def __init__(
        self,
        model: Union["torch.nn.Module", Any],
        experiment_name: str,
        teamspace: "Teamspace",
        version: str | None = None,
        verbose: bool = False,
        cloud_account: str | None = None,
        metadata: Dict[str, str] | None = None,
        staging_dir: str | None = None,
    ) -> None:
        self.model = model
        self.staging_dir = staging_dir
        self.name = f"{teamspace.owner.name}/{teamspace.name}/{experiment_name}"
        if version:
            self.name += f":{_sanitize_version_for_model_name(version)}"
        self.verbose = verbose
        self.cloud_account = cloud_account
        self.metadata = metadata

    def log(self) -> None:
        """Upload the model using litmodels."""
        save_model(
            name=self.name,
            model=self.model,
            staging_dir=self.staging_dir,
            verbose=False,
            progress_bar=self.verbose,
            cloud_account=self.cloud_account,
            metadata=self.metadata,
        )

    def get(self) -> Any:
        """Download the model using litmodels.

        Returns:
            The loaded model object.
        """
        return load_model(name=self.name, download_dir=self.staging_dir)
