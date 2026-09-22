# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for the artifacts API."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from litlogger.api.artifacts_api import ArtifactsApi


class TestArtifactsApi:
    """Test the ArtifactsApi class."""

    def test_init_with_client(self):
        """Test initialization with a custom client."""
        mock_client = MagicMock()
        api = ArtifactsApi(client=mock_client)
        assert api.client is mock_client

    def test_init_without_client(self):
        """Test initialization creates a default client."""
        with patch("litlogger.api.artifacts_api.LitRestClient") as mock_client_class:
            api = ArtifactsApi()
            mock_client_class.assert_called_once_with(max_retries=5)
            assert api.client == mock_client_class.return_value

    def test_upload_file_success(self):
        """Test successful file upload."""
        mock_teamspace = MagicMock()
        mock_teamspace.default_cloud_account = "acc-default"
        mock_client = MagicMock()
        api = ArtifactsApi(client=mock_client)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            remote_path = api.upload_file(
                teamspace=mock_teamspace,
                local_path=temp_file,
                remote_path="experiments/test/file.txt",
            )

            assert remote_path == "experiments/test/file.txt"
            mock_teamspace.upload_file.assert_called_once_with(
                file_path=temp_file,
                remote_path="experiments/test/file.txt",
                progress_bar=False,
                cloud_account="acc-default",
            )
        finally:
            os.unlink(temp_file)

    def test_upload_file_not_found(self):
        """Test upload with non-existent file."""
        mock_teamspace = MagicMock()
        api = ArtifactsApi()

        with pytest.raises(FileNotFoundError, match="File not found"):
            api.upload_file(
                teamspace=mock_teamspace,
                local_path="/nonexistent/file.txt",
                remote_path="experiments/test/file.txt",
            )

    def test_download_file_success(self):
        """Test successful file download."""
        mock_teamspace = MagicMock()
        mock_teamspace.default_cloud_account = "acc-default"
        api = ArtifactsApi()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "downloaded.txt")

            result = api.download_file(
                teamspace=mock_teamspace,
                remote_path="experiments/test/file.txt",
                local_path=local_path,
            )

            assert result == local_path
            mock_teamspace.download_file.assert_called_once_with(
                remote_path="experiments/test/file.txt",
                file_path=local_path,
                cloud_account="acc-default",
            )

    def test_download_file_creates_directory(self):
        """Test that download creates parent directory if needed."""
        mock_teamspace = MagicMock()
        api = ArtifactsApi()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "subdir", "downloaded.txt")

            api.download_file(
                teamspace=mock_teamspace,
                remote_path="experiments/test/file.txt",
                local_path=local_path,
            )

            # Check that the directory was created
            assert os.path.exists(os.path.dirname(local_path))

    def test_upload_experiment_file_artifact(self):
        """Test experiment-specific file upload with metrics stream registration."""
        mock_teamspace = MagicMock()
        mock_teamspace.name = "my-teamspace"
        mock_teamspace.id = "ts-123"
        mock_metrics_store = MagicMock()
        mock_metrics_store.id = "ms-456"
        mock_metrics_store.cluster_id = "acc-456"
        mock_client = MagicMock()

        api = ArtifactsApi(client=mock_client)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("experiment data")
            temp_file = f.name

        try:
            api.upload_experiment_file_artifact(
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
                experiment_name="my-experiment",
                file_path=temp_file,
                remote_path="data.txt",
            )

            # Check that file was uploaded to correct remote path
            mock_teamspace.upload_file.assert_called_once_with(
                file_path=temp_file,
                remote_path="experiments/my-experiment/data.txt",
                progress_bar=False,
                cloud_account="acc-456",
            )

            # Check that artifact was registered with metrics stream
            mock_client.lit_logger_service_create_logger_artifact.assert_called_once()
            call_args = mock_client.lit_logger_service_create_logger_artifact.call_args
            assert call_args[1]["project_id"] == "ts-123"
            assert call_args[1]["metrics_stream_id"] == "ms-456"
        finally:
            os.unlink(temp_file)

    def test_download_experiment_file_artifact(self):
        """Test experiment-specific file download."""
        mock_teamspace = MagicMock()
        mock_teamspace.default_cloud_account = "acc-default"
        api = ArtifactsApi()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "result.txt")

            downloaded = api.download_experiment_file_artifact(
                teamspace=mock_teamspace,
                experiment_name="my-experiment",
                filename="result.txt",
                local_path=local_path,
            )

            assert downloaded == local_path
            mock_teamspace.download_file.assert_called_once()
            call_args = mock_teamspace.download_file.call_args
            assert "experiments/my-experiment/result.txt" in call_args[1]["remote_path"]
            assert call_args[1]["cloud_account"] == "acc-default"

    def test_upload_metrics_binary(self):
        """Test uploading metrics binary tar.gz resolves the teamspace and delegates to upload_file."""
        mock_teamspace = MagicMock()
        api = ArtifactsApi()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tar.gz") as f:
            f.write(b"compressed metrics data")
            temp_file = f.name

        try:
            with patch("litlogger.api.artifacts_api._resolve_teamspace", return_value=mock_teamspace) as mock_resolve:
                api.upload_metrics_binary(
                    teamspace="my-teamspace",
                    cloud_account="acc-456",
                    file_path=temp_file,
                    remote_path="/litlogger/stream-789.tar.gz",
                )

            # A teamspace name is resolved to a Teamspace, then uploaded via its public method
            mock_resolve.assert_called_once_with("my-teamspace")
            mock_teamspace.upload_file.assert_called_once_with(
                file_path=temp_file,
                remote_path="/litlogger/stream-789.tar.gz",
                progress_bar=False,
                cloud_account="acc-456",
            )
        finally:
            os.unlink(temp_file)


class TestListExperimentArtifacts:
    """Test ArtifactsApi.list_experiment_artifacts."""

    def test_returns_listed_artifacts(self):
        """The logger_artifacts list is returned as-is."""
        art = MagicMock()
        art.path = "results.csv"
        mock_client = MagicMock()
        mock_client.lit_logger_service_list_logger_artifacts.return_value.logger_artifacts = [art]
        api = ArtifactsApi(client=mock_client)

        result = api.list_experiment_artifacts("ts-1", "ms-1")

        mock_client.lit_logger_service_list_logger_artifacts.assert_called_once_with(
            project_id="ts-1",
            metrics_stream_id="ms-1",
        )
        assert result == [art]

    def test_missing_endpoint_returns_none(self):
        """Older SDK clients without the endpoint yield None (caller falls back)."""
        mock_client = MagicMock()
        mock_client.lit_logger_service_list_logger_artifacts.side_effect = AttributeError("no method")
        api = ArtifactsApi(client=mock_client)

        assert api.list_experiment_artifacts("ts-1", "ms-1") is None

    def test_non_list_response_returns_none(self):
        """A response without a list-shaped logger_artifacts yields None."""
        mock_client = MagicMock()
        mock_client.lit_logger_service_list_logger_artifacts.return_value.logger_artifacts = None
        api = ArtifactsApi(client=mock_client)

        assert api.list_experiment_artifacts("ts-1", "ms-1") is None
