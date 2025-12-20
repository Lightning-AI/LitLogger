# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for Artifact and Model classes."""

import os
import tempfile
from unittest.mock import MagicMock, patch

from litlogger.artifacts import Artifact, GenericArtifact, Model, ModelArtifact, _sanitize_version_for_model_name


class TestSanitizeVersionForModelName:
    """Test the version sanitization function."""

    def test_sanitize_version_no_colons(self):
        """Test version without colons is unchanged."""
        assert _sanitize_version_for_model_name("v1.0") == "v1.0"
        assert _sanitize_version_for_model_name("2024-01-15") == "2024-01-15"

    def test_sanitize_version_with_colons(self):
        """Test version with colons replaces them with hyphens."""
        assert _sanitize_version_for_model_name("12:30:45") == "12-30-45"
        assert _sanitize_version_for_model_name("2024-01-15:12:30:45") == "2024-01-15-12-30-45"

    def test_sanitize_version_multiple_colons(self):
        """Test version with multiple colons replaces all."""
        assert _sanitize_version_for_model_name("a:b:c:d") == "a-b-c-d"

    def test_sanitize_version_empty_string(self):
        """Test empty string remains empty."""
        assert _sanitize_version_for_model_name("") == ""


class TestArtifact:
    """Test the Artifact class."""

    def test_init_basic(self):
        """Test Artifact initialization with basic parameters."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()

        with (
            patch("litlogger.artifacts.ArtifactsApi") as mock_api_class,
            patch("litlogger.artifacts.LitRestClient") as mock_client_class,
        ):
            artifact = Artifact(
                path="model.pth",
                experiment_name="exp1",
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
            )

            assert artifact.path == "model.pth"
            assert artifact.experiment_name == "exp1"
            assert artifact.teamspace is mock_teamspace
            assert artifact.metrics_store is mock_metrics_store
            assert artifact.display_path == "model.pth"
            assert artifact.remote_path == "experiments/exp1/model.pth"

            # Check that default client was created
            mock_client_class.assert_called_once_with(max_retries=5)

            # Check that API was initialized with client
            mock_api_class.assert_called_once_with(client=mock_client_class.return_value)

    def test_init_with_custom_client(self):
        """Test Artifact initialization with custom client."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()
        mock_client = MagicMock()

        with patch("litlogger.artifacts.ArtifactsApi") as mock_api_class:
            Artifact(
                path="checkpoint.pth",
                experiment_name="exp2",
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
                client=mock_client,
            )

            # Check that API was initialized with the custom client
            mock_api_class.assert_called_once_with(client=mock_client)

    def test_init_preserves_directory_structure(self):
        """Test that directory structure is preserved in remote path."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()

        with patch("litlogger.artifacts.ArtifactsApi"), patch("litlogger.artifacts.LitRestClient"):
            artifact = Artifact(
                path="checkpoints/epoch_10/model.pth",
                experiment_name="exp1",
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
            )

            assert artifact.display_path == "checkpoints/epoch_10/model.pth"
            assert artifact.remote_path == "experiments/exp1/checkpoints/epoch_10/model.pth"

    def test_init_with_custom_remote_path(self):
        """Test Artifact initialization with custom remote_path."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()

        with patch("litlogger.artifacts.ArtifactsApi"), patch("litlogger.artifacts.LitRestClient"):
            artifact = Artifact(
                path="/tmp/some/deep/path/file.txt",
                experiment_name="exp1",
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
                remote_path="custom/path/file.txt",
            )

            assert artifact.display_path == "custom/path/file.txt"
            assert artifact.remote_path == "experiments/exp1/custom/path/file.txt"

    def test_init_with_absolute_path_outside_cwd(self):
        """Test Artifact with absolute path outside cwd uses basename."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()

        with patch("litlogger.artifacts.ArtifactsApi"), patch("litlogger.artifacts.LitRestClient"):
            # Use a path that is definitely outside cwd
            artifact = Artifact(
                path="/tmp/random/deep/nested/file.txt",
                experiment_name="exp1",
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
            )

            # Should use basename since the path is outside cwd
            assert artifact.display_path == "file.txt"
            assert artifact.remote_path == "experiments/exp1/file.txt"

    def test_init_with_relative_path_under_cwd(self):
        """Test Artifact with relative path under cwd preserves structure."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()

        with patch("litlogger.artifacts.ArtifactsApi"), patch("litlogger.artifacts.LitRestClient"):
            artifact = Artifact(
                path="data/outputs/result.json",
                experiment_name="exp1",
                teamspace=mock_teamspace,
                metrics_store=mock_metrics_store,
            )

            assert artifact.display_path == "data/outputs/result.json"
            assert artifact.remote_path == "experiments/exp1/data/outputs/result.json"

    def test_upload(self):
        """Test artifact upload."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pth") as f:
            f.write("model weights")
            temp_file = f.name

        try:
            with (
                patch("litlogger.artifacts.ArtifactsApi") as mock_api_class,
                patch("litlogger.artifacts.LitRestClient"),
            ):
                mock_api = MagicMock()
                mock_api_class.return_value = mock_api

                artifact = Artifact(
                    path=temp_file,
                    experiment_name="exp1",
                    teamspace=mock_teamspace,
                    metrics_store=mock_metrics_store,
                )

                artifact.log()

                mock_api.upload_experiment_file_artifact.assert_called_once_with(
                    teamspace=mock_teamspace,
                    metrics_store=mock_metrics_store,
                    experiment_name="exp1",
                    file_path=temp_file,
                    remote_path=artifact.display_path,
                )
        finally:
            os.unlink(temp_file)

    def test_download(self):
        """Test artifact download."""
        mock_teamspace = MagicMock()
        mock_metrics_store = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "downloaded.pth")

            with (
                patch("litlogger.artifacts.ArtifactsApi") as mock_api_class,
                patch("litlogger.artifacts.LitRestClient"),
            ):
                mock_api = MagicMock()
                mock_api_class.return_value = mock_api
                mock_api.download_file.return_value = local_path

                artifact = Artifact(
                    path=local_path,
                    experiment_name="exp1",
                    teamspace=mock_teamspace,
                    metrics_store=mock_metrics_store,
                )

                result = artifact.get()

                assert result == local_path
                mock_api.download_file.assert_called_once_with(
                    teamspace=mock_teamspace,
                    remote_path=artifact.remote_path,
                    local_path=local_path,
                )


class TestModelArtifact:
    """Test the ModelArtifact class."""

    def test_init_basic(self):
        """Test ModelArtifact initialization."""
        mock_teamspace = MagicMock()

        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model = ModelArtifact(
            path="checkpoint.pth",
            experiment_name="exp1",
            teamspace=mock_teamspace,
            version="v1",
        )

        assert model.name == "owner/teamspace/exp1:v1"

    def test_init_without_version(self):
        """Test ModelArtifact initialization without version."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model = ModelArtifact(
            path="checkpoint.pth",
            experiment_name="exp1",
            teamspace=mock_teamspace,
        )

        assert model.name == "owner/teamspace/exp1"

    def test_init_version_with_colons_sanitized(self):
        """Test that version with colons is sanitized."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model = ModelArtifact(
            path="checkpoint.pth",
            experiment_name="exp1",
            teamspace=mock_teamspace,
            version="2024-01-15:12:30:45",
        )

        # Colons should be replaced with hyphens
        assert model.name == "owner/teamspace/exp1:2024-01-15-12-30-45"

    def test_model_artifact_inherits_from_generic_artifact(self):
        """Test that Model inherits from Artifact."""
        assert issubclass(ModelArtifact, GenericArtifact)

    def test_model_artifact_log(self):
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pth") as f:
            f.write("model weights")
            temp_file = f.name

        try:
            with (
                patch("litlogger.artifacts.upload_model") as mock_upload_model,
            ):
                model = ModelArtifact(
                    path=temp_file,
                    experiment_name="exp1",
                    teamspace=mock_teamspace,
                    version="v1",
                    verbose=True,
                    cloud_account="test-account",
                    metadata={"key": "value"},
                )

                model.log()

                mock_upload_model.assert_called_once_with(
                    name="owner/teamspace/exp1:v1",
                    model=temp_file,
                    verbose=False,
                    progress_bar=True,
                    cloud_account="test-account",
                    metadata={"key": "value"},
                )
        finally:
            os.unlink(temp_file)

    def test_model_artifact_get(self):
        """Test that ModelArtifact can download using download_model."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "model.pth")

            with (
                patch("litlogger.artifacts.download_model") as mock_download_model,
            ):
                mock_download_model.return_value = local_path

                model = ModelArtifact(
                    path=tmpdir,
                    experiment_name="exp1",
                    teamspace=mock_teamspace,
                    version="v1",
                    verbose=True,
                )

                result = model.get()

                assert result == local_path
                mock_download_model.assert_called_once_with(
                    name="owner/teamspace/exp1:v1",
                    download_dir=tmpdir,
                )


class TestModel:
    """Test the Model class."""

    def test_init_basic(self):
        """Test Model initialization."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model = Model(
            model="model",
            experiment_name="exp1",
            teamspace=mock_teamspace,
        )

        assert model.name == "owner/teamspace/exp1"
        assert model.model == "model"
        assert model.staging_dir is None

    def test_init_with_version(self):
        """Test Model initialization with version."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model = Model(
            model="model",
            experiment_name="exp1",
            teamspace=mock_teamspace,
            version="v1.0",
        )

        assert model.name == "owner/teamspace/exp1:v1.0"

    def test_init_version_with_colons_sanitized(self):
        """Test that version with colons is sanitized."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"

        model = Model(
            model="model",
            experiment_name="exp1",
            teamspace=mock_teamspace,
            version="2024-01-15:12:30:45",
        )

        # Colons should be replaced with hyphens
        assert model.name == "owner/teamspace/exp1:2024-01-15-12-30-45"

    def test_model_log(self):
        """Test Model log."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"
        mock_model_obj = MagicMock()

        with patch("litlogger.artifacts.save_model") as mock_save_model:
            model = Model(
                model=mock_model_obj,
                experiment_name="exp1",
                teamspace=mock_teamspace,
                version="v1",
                verbose=True,
                cloud_account="test-account",
                metadata={"key": "value"},
                staging_dir="/tmp/staging",
            )

            model.log()

            mock_save_model.assert_called_once_with(
                name="owner/teamspace/exp1:v1",
                model=mock_model_obj,
                staging_dir="/tmp/staging",
                verbose=False,
                progress_bar=True,
                cloud_account="test-account",
                metadata={"key": "value"},
            )

    def test_model_get(self):
        """Test Model get."""
        mock_teamspace = MagicMock()
        mock_teamspace.owner.name = "owner"
        mock_teamspace.name = "teamspace"
        mock_model_obj = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir, patch("litlogger.artifacts.load_model") as mock_load_model:
            mock_load_model.return_value = mock_model_obj

            model = Model(
                model="model",
                experiment_name="exp1",
                teamspace=mock_teamspace,
                version="v1",
                verbose=True,
                staging_dir=tmpdir,
            )

            result = model.get()

            assert result is mock_model_obj
            mock_load_model.assert_called_once_with(
                name="owner/teamspace/exp1:v1",
                download_dir=tmpdir,
            )
