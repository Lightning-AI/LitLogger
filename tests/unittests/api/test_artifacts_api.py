# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for artifacts API and artifact classes."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from litlogger.api.artifacts_api import ArtifactsApi
from litlogger.artifacts import Artifact, Model


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

    def test_upload_metrics_binary(self):
        """Test uploading metrics binary tar.gz file."""
        mock_client = MagicMock()
        api = ArtifactsApi(client=mock_client)

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tar.gz") as f:
            f.write(b"compressed metrics data")
            temp_file = f.name

        try:
            with patch("litlogger.api.artifacts_api._FileUploader") as mock_uploader:
                api.upload_metrics_binary(
                    teamspace_id="ts-123",
                    cloud_account="acc-456",
                    file_path=temp_file,
                    remote_path="/litlogger/stream-789.tar.gz",
                )

                # Check that FileUploader was called with correct parameters
                mock_uploader.assert_called_once_with(
                    client=mock_client,
                    teamspace_id="ts-123",
                    cloud_account="acc-456",
                    file_path=temp_file,
                    remote_path="/litlogger/stream-789.tar.gz",
                    progress_bar=False,
                )

                # Check that the uploader was called (executed)
                mock_uploader.return_value.assert_called_once()
        finally:
            os.unlink(temp_file)


class TestArtifact:
    """Test the Artifact class."""

    def test_init(self):
        """Test Artifact initialization."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()
        mock_client = MagicMock()

        artifact = Artifact(
            path="model.pth",
            experiment_name="exp1",
            teamspace=mock_teamspace,
            metrics_store=mock_metrics_store,
            client=mock_client,
        )

        assert artifact.path == "model.pth"
        assert artifact.experiment_name == "exp1"
        assert artifact.metrics_store is mock_metrics_store
        assert artifact.display_path == "model.pth"
        assert artifact.remote_path == "experiments/exp1/model.pth"

    def test_init_preserves_directory_structure(self):
        """Test that directory structure is preserved in remote path."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()

        artifact = Artifact(
            path="subdir/checkpoint.pth",
            experiment_name="exp1",
            teamspace=mock_teamspace,
            metrics_store=mock_metrics_store,
        )

        assert artifact.display_path == "subdir/checkpoint.pth"
        assert artifact.remote_path == "experiments/exp1/subdir/checkpoint.pth"

    def test_log(self):
        """Test artifact log (upload)."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()
        mock_api = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("artifact data")
            temp_file = f.name

        try:
            artifact = Artifact(
                path=temp_file,
                experiment_name="exp1",
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
            )
            artifact._api = mock_api

            result = artifact.log()

            assert result is None  # log() should not return anything
            mock_api.upload_experiment_file_artifact.assert_called_once_with(
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
                experiment_name="exp1",
                file_path=temp_file,
                remote_path=artifact.display_path,
            )
        finally:
            os.unlink(temp_file)

    def test_get(self):
        """Test artifact get (download)."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()
        mock_api = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "artifact.pth")

            artifact = Artifact(
                path=local_path,
                experiment_name="exp1",
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
            )
            artifact._api = mock_api
            mock_api.download_file.return_value = local_path

            result = artifact.get()

            assert result == local_path
            mock_api.download_file.assert_called_once_with(
                teamspace=mock_teamspace,
                remote_path=artifact.remote_path,
                local_path=local_path,
            )


class TestModelArtifact:
    """Test the ModelArtifact class for logging model files using litmodels."""

    def test_init(self):
        """Test ModelArtifact initialization."""
        from litlogger.artifacts import ModelArtifact

        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model_artifact = ModelArtifact(
            path="checkpoint.pth", experiment_name="exp1", teamspace=mock_teamspace, version="v1.0"
        )

        assert model_artifact.path == "checkpoint.pth"
        assert model_artifact.name == "owner/teamspace/exp1:v1.0"

    def test_init_without_version(self):
        """Test ModelArtifact initialization without version."""
        from litlogger.artifacts import ModelArtifact

        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model_artifact = ModelArtifact(path="checkpoint.pth", experiment_name="exp1", teamspace=mock_teamspace)

        assert model_artifact.name == "owner/teamspace/exp1"

    @patch("litlogger.artifacts.upload_model")
    def test_log(self, mock_upload):
        """Test ModelArtifact log method."""
        from litlogger.artifacts import ModelArtifact

        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model_artifact = ModelArtifact(path="model.pt", experiment_name="exp1", teamspace=mock_teamspace, verbose=True)

        model_artifact.log()

        mock_upload.assert_called_once_with(
            name="owner/teamspace/exp1",
            model="model.pt",
            verbose=False,
            progress_bar=True,
            cloud_account=None,
            metadata=None,
        )

    @patch("litlogger.artifacts.download_model")
    def test_get(self, mock_download):
        """Test ModelArtifact get method."""
        from litlogger.artifacts import ModelArtifact

        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model_artifact = ModelArtifact(path="/tmp/models", experiment_name="exp1", teamspace=mock_teamspace)

        model_artifact.get()

        mock_download.assert_called_once_with(name="owner/teamspace/exp1", download_dir="/tmp/models")


class TestModel:
    """Test the Model class for logging model objects using litmodels."""

    def test_init(self):
        """Test Model initialization."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"
        mock_model_obj = MagicMock()

        model = Model(model=mock_model_obj, experiment_name="exp1", teamspace=mock_teamspace, version="v1.0")

        assert model.model == mock_model_obj
        assert model.name == "owner/teamspace/exp1:v1.0"

    @patch("litlogger.artifacts.save_model")
    def test_log(self, mock_save):
        """Test Model log method."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"
        mock_model_obj = MagicMock()

        model = Model(
            model=mock_model_obj, experiment_name="exp1", teamspace=mock_teamspace, staging_dir="/tmp/staging"
        )

        model.log()

        mock_save.assert_called_once_with(
            name="owner/teamspace/exp1",
            model=mock_model_obj,
            staging_dir="/tmp/staging",
            verbose=False,
            progress_bar=False,
            cloud_account=None,
            metadata=None,
        )

    @patch("litlogger.artifacts.load_model")
    def test_get(self, mock_load):
        """Test Model get method."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"
        mock_model_obj = MagicMock()

        model = Model(
            model=mock_model_obj, experiment_name="exp1", teamspace=mock_teamspace, staging_dir="/tmp/staging"
        )

        model.get()

        mock_load.assert_called_once_with(name="owner/teamspace/exp1", download_dir="/tmp/staging")


class TestLitModelsIntegration:
    """Test integration with litmodels package."""

    def test_litmodels_import(self):
        """Test that litmodels functions can be imported."""
        from litlogger.artifacts import download_model, upload_model

        # These should be the litmodels functions
        assert callable(upload_model)
        assert callable(download_model)


class TestGenericArtifactProtocol:
    """Test that all artifact classes implement the GenericArtifact protocol."""

    def test_artifact_implements_protocol(self):
        """Test that Artifact implements GenericArtifact protocol."""
        from litlogger.artifacts import Artifact, GenericArtifact

        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()
        artifact = Artifact(
            path="test.txt",
            experiment_name="exp",
            teamspace=mock_teamspace,
            metrics_store=mock_metrics_store,
        )

        assert isinstance(artifact, GenericArtifact)
        assert hasattr(artifact, "log")
        assert hasattr(artifact, "get")
        assert callable(artifact.log)
        assert callable(artifact.get)

    def test_model_artifact_implements_protocol(self):
        """Test that ModelArtifact implements GenericArtifact protocol."""
        from litlogger.artifacts import GenericArtifact, ModelArtifact

        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model_artifact = ModelArtifact(path="model.pt", experiment_name="exp", teamspace=mock_teamspace)

        assert isinstance(model_artifact, GenericArtifact)
        assert hasattr(model_artifact, "log")
        assert hasattr(model_artifact, "get")
        assert callable(model_artifact.log)
        assert callable(model_artifact.get)

    def test_model_implements_protocol(self):
        """Test that Model implements GenericArtifact protocol."""
        from litlogger.artifacts import GenericArtifact

        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"
        mock_model = MagicMock()

        model = Model(model=mock_model, experiment_name="exp", teamspace=mock_teamspace)

        assert isinstance(model, GenericArtifact)
        assert hasattr(model, "log")
        assert hasattr(model, "get")
        assert callable(model.log)
        assert callable(model.get)

    def test_protocol_structural_subtyping(self):
        """Test that the protocol uses structural subtyping."""
        from litlogger.artifacts import GenericArtifact

        # Create a class that implements the protocol without inheriting
        class CustomArtifact:
            def log(self) -> str:
                return "logged"

            def get(self) -> str:
                return "retrieved"

        custom = CustomArtifact()
        # Should be recognized as implementing the protocol
        assert isinstance(custom, GenericArtifact)
