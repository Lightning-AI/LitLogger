# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for API utilities."""

from unittest.mock import MagicMock, patch

import pytest
from lightning_sdk.lightning_cloud.openapi import V1OwnerType
from litlogger.api.utils import _resolve_teamspace, _resolve_teamspace_owner, build_experiment_url, get_accessible_url


def create_mock_teamspace(side_effect=None):
    """Create a Teamspace mock class that works with isinstance checks.

    Args:
        side_effect: Exception or list of exceptions/return values. Each item can be
                     an Exception (to raise) or any other value (to return).

    Returns:
        A class that can be used with patch() and works with isinstance().
    """
    effects = side_effect if isinstance(side_effect, list) else ([side_effect] if side_effect else [])
    call_idx = [0]  # Use list to allow mutation in closure
    calls = []

    class MockTeamspace:
        def __new__(cls, *args, **kwargs):
            calls.append((args, kwargs))
            if call_idx[0] < len(effects):
                effect = effects[call_idx[0]]
                call_idx[0] += 1
                if isinstance(effect, Exception):
                    raise effect
                return effect
            return MagicMock()

    MockTeamspace.calls = calls
    return MockTeamspace


class TestResolveTeamspace:
    """Test the _resolve_teamspace function."""

    def test_resolve_with_valid_teamspace(self):
        """Test resolving with a valid teamspace name that works directly."""
        mock_teamspace = MagicMock()
        MockTeamspace = create_mock_teamspace(side_effect=[mock_teamspace])  # noqa: N806

        with patch("litlogger.api.utils.Teamspace", MockTeamspace):
            result = _resolve_teamspace("my-teamspace")

            # Should try to create Teamspace directly
            assert MockTeamspace.calls == [(("my-teamspace",), {})]
            assert result == mock_teamspace

    def test_resolve_with_none_teamspace_single_membership(self):
        """Test resolving with None teamspace when user has one membership."""
        MockTeamspace = create_mock_teamspace(side_effect=[Exception("Not found"), MagicMock()])  # noqa: N806

        with (
            patch("litlogger.api.utils.Teamspace", MockTeamspace),
            patch("litlogger.api.utils.LitRestClient") as mock_client_class,
            patch("litlogger.api.utils._resolve_teamspace_owner") as mock_resolve_owner,
        ):
            # Setup client
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Setup membership response
            mock_membership = MagicMock()
            mock_membership.name = "default-teamspace"
            mock_membership.owner_id = "owner-123"
            mock_membership.owner_type = V1OwnerType.USER

            mock_response = MagicMock()
            mock_response.memberships = [mock_membership]
            mock_client.projects_service_list_memberships.return_value = mock_response

            # Setup owner resolution
            mock_resolve_owner.return_value = {"user": "my-user", "org": None}

            _resolve_teamspace(None)

            # Should use the first membership
            mock_resolve_owner.assert_called_once_with(mock_client, "owner-123", V1OwnerType.USER)
            # Second call should be with the resolved teamspace name and owner kwargs
            assert len(MockTeamspace.calls) == 2

    def test_resolve_with_matching_project_id(self):
        """Test resolving when teamspace matches a project_id."""
        MockTeamspace = create_mock_teamspace(side_effect=[Exception("Not found"), MagicMock()])  # noqa: N806

        with (
            patch("litlogger.api.utils.Teamspace", MockTeamspace),
            patch("litlogger.api.utils.LitRestClient") as mock_client_class,
            patch("litlogger.api.utils._resolve_teamspace_owner") as mock_resolve_owner,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Setup memberships
            mock_membership1 = MagicMock()
            mock_membership1.project_id = "proj-456"
            mock_membership1.name = "other-teamspace"

            mock_membership2 = MagicMock()
            mock_membership2.project_id = "proj-123"
            mock_membership2.name = "target-teamspace"
            mock_membership2.owner_id = "owner-456"
            mock_membership2.owner_type = V1OwnerType.ORGANIZATION

            mock_response = MagicMock()
            mock_response.memberships = [mock_membership1, mock_membership2]
            mock_client.projects_service_list_memberships.return_value = mock_response

            mock_resolve_owner.return_value = {"user": None, "org": "my-org"}

            _resolve_teamspace("proj-123")

            # Should find the matching project_id
            mock_resolve_owner.assert_called_once_with(mock_client, "owner-456", V1OwnerType.ORGANIZATION)

    def test_resolve_with_matching_name(self):
        """Test resolving when teamspace matches a membership name."""
        MockTeamspace = create_mock_teamspace(side_effect=[Exception("Not found"), MagicMock()])  # noqa: N806

        with (
            patch("litlogger.api.utils.Teamspace", MockTeamspace),
            patch("litlogger.api.utils.LitRestClient") as mock_client_class,
            patch("litlogger.api.utils._resolve_teamspace_owner") as mock_resolve_owner,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Setup memberships
            mock_membership = MagicMock()
            mock_membership.project_id = "proj-123"
            mock_membership.name = "my-teamspace"
            mock_membership.display_name = "My Teamspace"
            mock_membership.owner_id = "owner-123"
            mock_membership.owner_type = V1OwnerType.USER

            mock_response = MagicMock()
            mock_response.memberships = [mock_membership]
            mock_client.projects_service_list_memberships.return_value = mock_response

            mock_resolve_owner.return_value = {"user": "my-user", "org": None}

            _resolve_teamspace("my-teamspace")

            # Should find the matching name
            mock_resolve_owner.assert_called_once()

    def test_resolve_with_matching_display_name(self):
        """Test resolving when teamspace matches a membership display_name."""
        MockTeamspace = create_mock_teamspace(side_effect=[Exception("Not found"), MagicMock()])  # noqa: N806

        with (
            patch("litlogger.api.utils.Teamspace", MockTeamspace),
            patch("litlogger.api.utils.LitRestClient") as mock_client_class,
            patch("litlogger.api.utils._resolve_teamspace_owner") as mock_resolve_owner,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Setup memberships
            mock_membership = MagicMock()
            mock_membership.project_id = "proj-123"
            mock_membership.name = "my-teamspace"
            mock_membership.display_name = "My Beautiful Teamspace"
            mock_membership.owner_id = "owner-123"
            mock_membership.owner_type = V1OwnerType.USER

            mock_response = MagicMock()
            mock_response.memberships = [mock_membership]
            mock_client.projects_service_list_memberships.return_value = mock_response

            mock_resolve_owner.return_value = {"user": "my-user", "org": None}

            _resolve_teamspace("My Beautiful Teamspace")

            # Should find the matching display_name
            mock_resolve_owner.assert_called_once()

    def test_resolve_with_no_memberships(self):
        """Test resolving when user has no teamspaces."""
        MockTeamspace = create_mock_teamspace(side_effect=[Exception("Not found")])  # noqa: N806

        with (
            patch("litlogger.api.utils.Teamspace", MockTeamspace),
            patch("litlogger.api.utils.LitRestClient") as mock_client_class,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.memberships = []
            mock_client.projects_service_list_memberships.return_value = mock_response

            with pytest.raises(ValueError, match="No valid teamspaces found"):
                _resolve_teamspace(None)

    def test_resolve_with_verbose(self):
        """Test resolving with verbose output."""
        MockTeamspace = create_mock_teamspace(side_effect=[Exception("Not found"), MagicMock()])  # noqa: N806

        with (
            patch("litlogger.api.utils.Teamspace", MockTeamspace),
            patch("litlogger.api.utils.LitRestClient") as mock_client_class,
            patch("litlogger.api.utils._resolve_teamspace_owner") as mock_resolve_owner,
            patch("builtins.print") as mock_print,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_membership = MagicMock()
            mock_membership.name = "default-teamspace"
            mock_membership.owner_id = "owner-123"
            mock_membership.owner_type = V1OwnerType.USER

            mock_response = MagicMock()
            mock_response.memberships = [mock_membership]
            mock_client.projects_service_list_memberships.return_value = mock_response

            mock_resolve_owner.return_value = {"user": "my-user", "org": None}

            _resolve_teamspace(None, verbose=True)

            # Should print the default teamspace message
            mock_print.assert_called_once()
            assert "default-teamspace" in str(mock_print.call_args)


class TestResolveTeamspaceOwner:
    """Test the _resolve_teamspace_owner function."""

    def test_resolve_user_owner(self):
        """Test resolving a user owner."""
        mock_client = MagicMock()

        # Setup user response
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.username = "my-username"

        mock_response = MagicMock()
        mock_response.users = [mock_user]
        mock_client.user_service_search_users.return_value = mock_response

        result = _resolve_teamspace_owner(mock_client, "user-123", V1OwnerType.USER)

        assert result == {"user": "my-username", "org": None}
        mock_client.user_service_search_users.assert_called_once_with(query="user-123")

    def test_resolve_user_owner_not_found(self):
        """Test resolving a user owner that doesn't exist."""
        mock_client = MagicMock()

        # Setup empty user response
        mock_response = MagicMock()
        mock_response.users = []
        mock_client.user_service_search_users.return_value = mock_response

        with pytest.raises(RuntimeError, match="owner of the teamspace couldn't be found"):
            _resolve_teamspace_owner(mock_client, "user-123", V1OwnerType.USER)

    def test_resolve_user_owner_wrong_id(self):
        """Test resolving when user search returns different user."""
        mock_client = MagicMock()

        # Setup user response with different ID
        mock_user = MagicMock()
        mock_user.id = "user-456"  # Different from the requested ID
        mock_user.username = "other-user"

        mock_response = MagicMock()
        mock_response.users = [mock_user]
        mock_client.user_service_search_users.return_value = mock_response

        with pytest.raises(RuntimeError, match="owner of the teamspace couldn't be found"):
            _resolve_teamspace_owner(mock_client, "user-123", V1OwnerType.USER)

    def test_resolve_org_owner(self):
        """Test resolving an organization owner."""
        mock_client = MagicMock()

        # Setup org response
        mock_org = MagicMock()
        mock_org.name = "my-organization"
        mock_client.organizations_service_get_organization.return_value = mock_org

        result = _resolve_teamspace_owner(mock_client, "org-123", V1OwnerType.ORGANIZATION)

        assert result == {"org": "my-organization", "user": None}
        mock_client.organizations_service_get_organization.assert_called_once_with(id="org-123")


class TestBuildExperimentUrl:
    """Test the build_experiment_url function."""

    def test_build_experiment_url(self):
        """Test building experiment URL."""
        with patch("litlogger.api.utils._get_cloud_url") as mock_cloud_url:
            mock_cloud_url.return_value = "https://lightning.ai"

            url = build_experiment_url(
                owner_name="my-org",
                teamspace_name="my-teamspace",
                experiment_name="my-experiment",
                version="1",
            )

            # The URL should be constructed from the parameters
            assert url == "https://lightning.ai/my-org/my-teamspace/experiments/my-experiment%20-%20v1"


class TestGetAccessibleUrl:
    """Test the get_accessible_url function."""

    def test_get_accessible_url_no_cloudspace(self):
        """Test getting accessible URL when there's no cloudspace."""
        mock_client = MagicMock()
        mock_teamspace = MagicMock()
        mock_teamspace.name = "my-teamspace"
        mock_teamspace.id = "ts-123"
        mock_metrics_store = MagicMock()
        mock_metrics_store.cloudspace_id = ""

        with patch("litlogger.api.utils._get_cloud_url") as mock_cloud_url:
            mock_cloud_url.return_value = "https://lightning.ai"

            url = get_accessible_url(
                teamspace=mock_teamspace,
                owner_name="my-org",
                metrics_store=mock_metrics_store,
                client=mock_client,
            )

            assert url == "https://lightning.ai/my-org/my-teamspace/experiments"

    def test_get_accessible_url_with_cloudspace_no_work(self):
        """Test getting accessible URL with cloudspace but no work."""
        mock_client = MagicMock()
        mock_cloudspace = MagicMock()
        mock_cloudspace.name = "my-studio"
        mock_client.cloud_space_service_get_cloud_space.return_value = mock_cloudspace

        mock_teamspace = MagicMock()
        mock_teamspace.name = "my-teamspace"
        mock_teamspace.id = "ts-123"

        mock_metrics_store = MagicMock()
        mock_metrics_store.cloudspace_id = "cs-456"
        mock_metrics_store.work_id = ""

        with patch("litlogger.api.utils._get_cloud_url") as mock_cloud_url:
            mock_cloud_url.return_value = "https://lightning.ai"

            url = get_accessible_url(
                teamspace=mock_teamspace,
                owner_name="my-org",
                metrics_store=mock_metrics_store,
                client=mock_client,
            )

            assert url == "https://lightning.ai/my-org/my-teamspace/studios/my-studio/lit-logger?app_id=031"
            mock_client.cloud_space_service_get_cloud_space.assert_called_once_with(project_id="ts-123", id="cs-456")

    def test_get_accessible_url_with_work(self):
        """Test getting accessible URL with cloudspace and work."""
        mock_client = MagicMock()
        mock_cloudspace = MagicMock()
        mock_cloudspace.name = "my-studio"
        mock_client.cloud_space_service_get_cloud_space.return_value = mock_cloudspace

        mock_teamspace = MagicMock()
        mock_teamspace.name = "my-teamspace"
        mock_teamspace.id = "ts-123"

        mock_metrics_store = MagicMock()
        mock_metrics_store.cloudspace_id = "cs-456"
        mock_metrics_store.work_id = "work-789"
        mock_metrics_store.plugin_id = "job_run_plugin"
        mock_metrics_store.job_name = "my-job"

        with patch("litlogger.api.utils._get_cloud_url") as mock_cloud_url:
            mock_cloud_url.return_value = "https://lightning.ai"

            url = get_accessible_url(
                teamspace=mock_teamspace,
                owner_name="my-org",
                metrics_store=mock_metrics_store,
                client=mock_client,
            )

            expected = "https://lightning.ai/my-org/my-teamspace/studios/my-studio/app?app_id=jobs&job_name=my-job"
            assert url == expected

    def test_get_accessible_url_with_unknown_plugin(self):
        """Test getting accessible URL with unknown plugin raises error."""
        mock_client = MagicMock()
        mock_cloudspace = MagicMock()
        mock_cloudspace.name = "my-studio"
        mock_client.cloud_space_service_get_cloud_space.return_value = mock_cloudspace

        mock_teamspace = MagicMock()
        mock_teamspace.name = "my-teamspace"
        mock_teamspace.id = "ts-123"

        mock_metrics_store = MagicMock()
        mock_metrics_store.cloudspace_id = "cs-456"
        mock_metrics_store.work_id = "work-789"
        mock_metrics_store.plugin_id = "unknown_plugin"

        with patch("litlogger.api.utils._get_cloud_url") as mock_cloud_url:
            mock_cloud_url.return_value = "https://lightning.ai"

            with pytest.raises(RuntimeError, match="plugin id unknown_plugin wasn't found"):
                get_accessible_url(
                    teamspace=mock_teamspace,
                    owner_name="my-org",
                    metrics_store=mock_metrics_store,
                    client=mock_client,
                )
