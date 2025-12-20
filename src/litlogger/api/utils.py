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
from typing import Any, TypedDict

from lightning_sdk import Teamspace
from lightning_sdk.api.utils import _get_cloud_url
from lightning_sdk.lightning_cloud.openapi import V1OwnerType

from litlogger.api.auth_api import AuthApi
from litlogger.api.client import LitRestClient


def _resolve_teamspace(teamspace: str | Teamspace | None, verbose: bool = False) -> Teamspace:
    if isinstance(teamspace, Teamspace):
        return teamspace
    try:
        return Teamspace(teamspace)
    except Exception as e:
        client = LitRestClient(max_retries=0)
        response = client.projects_service_list_memberships()
        memberships = response.memberships
        if len(memberships) == 0:
            raise ValueError(
                "No valid teamspaces found. Please reach out to lightning.ai team to create a teamspace"
            ) from e

        resolved_teamspace = None
        owner_kwargs = {}
        for membership in memberships:
            if (
                teamspace is not None
                and membership.project_id == teamspace
                or membership.name == teamspace
                or membership.display_name == teamspace
            ):
                resolved_teamspace = membership.name
                owner_kwargs = _resolve_teamspace_owner(client, membership.owner_id, membership.owner_type)
                break

        if resolved_teamspace is None and len(memberships) > 0:
            resolved_teamspace = memberships[0].name
            owner_kwargs = _resolve_teamspace_owner(client, memberships[0].owner_id, memberships[0].owner_type)

        if resolved_teamspace is None:
            raise ValueError(f"Teamspace {teamspace} not found") from None

        if verbose:
            print(f"Defaulting to the teamspace: {resolved_teamspace}")

        return Teamspace(resolved_teamspace, **owner_kwargs)


class OwnerDict(TypedDict):
    user: str | None
    org: str | None


def _resolve_teamspace_owner(client: LitRestClient, owner_id: str, owner_type: V1OwnerType) -> OwnerDict:
    if owner_type == V1OwnerType.USER:
        response = client.user_service_search_users(query=owner_id)
        users = [u for u in response.users if u.id == owner_id]
        if len(users) == 0:
            raise RuntimeError("The owner of the teamspace couldn't be found.")
        return {"user": users[0].username, "org": None}

    return {"org": client.organizations_service_get_organization(id=owner_id).name, "user": None}


def build_experiment_url(
    owner_name: str,
    teamspace_name: str,
    experiment_name: str,
    version: str,
) -> str:
    """Build the direct URL to an experiment.

    Args:
        owner_name: Owner username or org name.
        teamspace_name: Teamspace name.
        experiment_name: Experiment name.
        version: Experiment version.

    Returns:
        str: The experiment URL.
    """
    cloud_url = _get_cloud_url()
    return f"{cloud_url}/{owner_name}/{teamspace_name}/experiments/{experiment_name}%20-%20v{version}"


_MAP_PLUGIN_ID_TO_APP = {"job_run_plugin": "jobs", "distributed_plugin": "mmt", "litdata": "litdata"}


def get_accessible_url(
    teamspace: Teamspace,
    owner_name: str,
    metrics_store: Any,
    client: LitRestClient,
) -> str:
    """Get the accessible URL for viewing metrics.

    Args:
        teamspace: The project/teamspace membership.
        owner_name: Owner name.
        metrics_store: The metrics store object.
        client: REST client for API calls.

    Returns:
        str: The URL where metrics can be viewed.
    """
    cloud_url = _get_cloud_url()
    base_url = f"{cloud_url}/{owner_name}/{teamspace.name}/"

    if metrics_store.cloudspace_id == "":
        return base_url + "experiments"

    cloudspace = client.cloud_space_service_get_cloud_space(project_id=teamspace.id, id=metrics_store.cloudspace_id)
    if metrics_store.work_id != "":
        if metrics_store.plugin_id not in _MAP_PLUGIN_ID_TO_APP:
            raise RuntimeError(
                f"The stream plugin id {metrics_store.plugin_id} wasn't found in {_MAP_PLUGIN_ID_TO_APP}"
            )
        app_id = _MAP_PLUGIN_ID_TO_APP[metrics_store.plugin_id]
        return base_url + f"studios/{cloudspace.name}/app?app_id={app_id}&job_name={metrics_store.job_name}"

    return base_url + f"studios/{cloudspace.name}/lit-logger?app_id=031"


def get_guest_url(auth_api: AuthApi) -> str:
    """Get the guest URL for viewing metrics.

    Returns:
        str: The guest URL.
    """
    cloud_url = _get_cloud_url()
    return f"{cloud_url}/guest/experiments?guestId={auth_api.guest_id}"
