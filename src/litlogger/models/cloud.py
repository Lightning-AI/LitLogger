"""Cloud-facing model registry helpers."""

from pathlib import Path
from typing import TYPE_CHECKING

from lightning_sdk.lightning_cloud.env import LIGHTNING_CLOUD_URL
from lightning_sdk.models import _extend_model_name_with_teamspace, _parse_org_teamspace_model_version
from lightning_sdk.models import delete_model as sdk_delete_model
from lightning_sdk.models import download_model as sdk_download_model
from lightning_sdk.models import upload_model as sdk_upload_model

if TYPE_CHECKING:
    from lightning_sdk.models import UploadedModelInfo

    from litlogger.experiment import Experiment


_SHOWED_MODEL_LINKS: list[str] = []


def _litlogger_version() -> str:
    from litlogger import __version__

    return __version__


def _print_model_link(name: str, verbose: bool | int) -> None:
    """Print a stable URL to the uploaded model."""
    name = _extend_model_name_with_teamspace(name)
    org_name, teamspace_name, model_name, _ = _parse_org_teamspace_model_version(name)

    url = f"{LIGHTNING_CLOUD_URL}/{org_name}/{teamspace_name}/models/{model_name}"
    msg = f"Model uploaded successfully. Link to the model: '{url}'"
    if int(verbose) > 1:
        print(msg)
    elif url not in _SHOWED_MODEL_LINKS:
        print(msg)
        _SHOWED_MODEL_LINKS.append(url)


def upload_model_files(
    name: str,
    path: str | Path | list[str | Path],
    progress_bar: bool = True,
    cloud_account: str | None = None,
    verbose: bool | int = 1,
    metadata: dict[str, str] | None = None,
    experiment: "Experiment | None" = None,
) -> "UploadedModelInfo":
    """Upload local artifact(s) to Lightning Cloud using the SDK."""
    upload_metadata = dict(metadata or {})
    upload_metadata["litModels"] = _litlogger_version()
    info = sdk_upload_model(
        name=name,
        path=path,
        progress_bar=progress_bar,
        cloud_account=cloud_account,
        metadata=upload_metadata,
        experiment=experiment,
    )
    if verbose:
        _print_model_link(name, verbose)
    return info


def download_model_files(
    name: str,
    download_dir: str | Path = ".",
    progress_bar: bool = True,
) -> str | list[str]:
    """Download artifact(s) for a model version using the SDK."""
    return sdk_download_model(
        name=name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )


def _list_available_teamspaces() -> dict[str, dict]:
    """List teamspaces available to the authenticated user."""
    from lightning_sdk.api import OrgApi, UserApi
    from lightning_sdk.utils import resolve as sdk_resolvers

    org_api = OrgApi()
    user = sdk_resolvers._get_authed_user()
    teamspaces = {}
    for teamspace in UserApi()._get_all_teamspace_memberships(""):
        if teamspace.owner_type == "organization":
            org = org_api._get_org_by_id(teamspace.owner_id)
            teamspaces[f"{org.name}/{teamspace.name}"] = {"name": teamspace.name, "org": org.name}
        elif teamspace.owner_type == "user":
            teamspaces[f"{user.name}/{teamspace.name}"] = {"name": teamspace.name, "user": user}
        else:
            raise RuntimeError(f"Unknown organization type {teamspace.organization_type}")
    return teamspaces


def delete_model_version(name: str, version: str) -> None:
    """Delete a specific model version from the model store."""
    sdk_delete_model(name=f"{name}:{version}")
