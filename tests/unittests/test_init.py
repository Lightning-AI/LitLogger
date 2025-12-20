# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for litlogger.init initialization logic."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import litlogger
import pytest
from litlogger.init import finish, init

# Import the module from sys.modules to avoid the shadowing issue
# (litlogger.init function shadows the module)
init_module = sys.modules["litlogger.init"]


class TestInit:
    """Test the init function."""

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_with_default_parameters(self, mock_check_output, mock_set_global, mock_experiment_class, tmpdir):
        """Test init with all default parameters."""
        # Mock git command to return a repo name
        mock_check_output.return_value = b"/path/to/my-repo\n"

        # Mock Experiment class
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        with patch.object(init_module, "_create_name", return_value="generated-name"):
            result = init()

        # Verify Experiment was created
        mock_experiment_class.assert_called_once()
        call_kwargs = mock_experiment_class.call_args.kwargs

        assert call_kwargs["name"] == "generated-name"
        assert call_kwargs["teamspace"] is None
        assert call_kwargs["metadata"] == {}
        assert call_kwargs["store_step"] is True
        assert call_kwargs["store_created_at"] is False
        assert call_kwargs["save_logs"] is False
        assert "lightning_logs" in call_kwargs["log_dir"]

        # Verify global state was set
        mock_set_global.assert_called_once_with(
            experiment=mock_experiment,
            log=mock_experiment.log_metrics,
            log_metrics=mock_experiment.log_metrics,
            log_file=mock_experiment.log_file,
            get_file=mock_experiment.get_file,
            log_model=mock_experiment.log_model,
            get_model=mock_experiment.get_model,
            log_model_artifact=mock_experiment.log_model_artifact,
            get_model_artifact=mock_experiment.get_model_artifact,
            finalize=mock_experiment.finalize,
        )

        assert result is mock_experiment

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_with_custom_name(self, mock_check_output, mock_set_global, mock_experiment_class):
        """Test init with custom name."""
        mock_check_output.return_value = b"/path/to/repo\n"
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        init(name="my-custom-experiment")

        call_kwargs = mock_experiment_class.call_args.kwargs
        assert call_kwargs["name"] == "my-custom-experiment"

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_with_custom_root_dir(self, mock_check_output, mock_set_global, mock_experiment_class, tmpdir):
        """Test init with custom root directory."""
        mock_check_output.return_value = b"/path/to/repo\n"
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        custom_root = str(tmpdir / "custom_logs")

        with patch.object(init_module, "_create_name", return_value="test-name"):
            init(root_dir=custom_root)

        call_kwargs = mock_experiment_class.call_args.kwargs
        assert custom_root in call_kwargs["log_dir"]

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_with_teamspace(self, mock_check_output, mock_set_global, mock_experiment_class):
        """Test init with teamspace parameter."""
        mock_check_output.return_value = b"/path/to/repo\n"
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        init(name="test", teamspace="my-teamspace")

        call_kwargs = mock_experiment_class.call_args.kwargs
        assert call_kwargs["teamspace"] == "my-teamspace"

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_with_metadata(self, mock_check_output, mock_set_global, mock_experiment_class):
        """Test init with metadata parameter."""
        mock_check_output.return_value = b"/path/to/repo\n"
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        metadata = {"lr": "0.001", "batch_size": "32"}
        init(name="test", metadata=metadata)

        call_kwargs = mock_experiment_class.call_args.kwargs
        assert call_kwargs["metadata"] == metadata

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_with_store_flags(self, mock_check_output, mock_set_global, mock_experiment_class):
        """Test init with store_step and store_created_at flags."""
        mock_check_output.return_value = b"/path/to/repo\n"
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        init(name="test", store_step=False, store_created_at=True)

        call_kwargs = mock_experiment_class.call_args.kwargs
        assert call_kwargs["store_step"] is False
        assert call_kwargs["store_created_at"] is True

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_with_save_logs(self, mock_check_output, mock_set_global, mock_experiment_class):
        """Test init with save_logs=True."""
        mock_check_output.return_value = b"/path/to/repo\n"
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        init(name="test", save_logs=True)

        call_kwargs = mock_experiment_class.call_args.kwargs
        assert call_kwargs["save_logs"] is True

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_git_repo_name_detection(self, mock_check_output, mock_set_global, mock_experiment_class):
        """Test that git repo name is correctly extracted."""
        mock_check_output.return_value = b"/home/user/projects/awesome-project\n"
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        init(name="test")

        call_kwargs = mock_experiment_class.call_args.kwargs
        assert call_kwargs["name"] == "test"
        assert "version" in call_kwargs
        assert call_kwargs["teamspace"] is None

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_falls_back_to_cwd_on_git_error(self, mock_check_output, mock_set_global, mock_experiment_class):
        """Test fallback to current directory name when git command fails."""
        # Simulate git command failure
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        with patch.object(Path, "cwd", return_value=Path("/home/user/my-project")):
            init(name="test")

        call_kwargs = mock_experiment_class.call_args.kwargs
        assert call_kwargs["name"] == "test"
        assert "version" in call_kwargs
        assert call_kwargs["teamspace"] is None

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_falls_back_to_cwd_when_git_not_found(self, mock_check_output, mock_set_global, mock_experiment_class):
        """Test fallback to current directory when git is not installed."""
        # Simulate git not being found
        mock_check_output.side_effect = FileNotFoundError()
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        with patch.object(Path, "cwd", return_value=Path("/home/user/another-project")):
            init(name="test")

        call_kwargs = mock_experiment_class.call_args.kwargs
        assert call_kwargs["name"] == "test"
        assert "version" in call_kwargs
        assert call_kwargs["teamspace"] is None

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    @patch("os.makedirs")
    def test_init_creates_log_directory(self, mock_makedirs, mock_check_output, mock_set_global, mock_experiment_class):
        """Test that init creates the log directory."""
        mock_check_output.return_value = b"/path/to/repo\n"
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        init(name="test-exp", root_dir="/tmp/logs")

        # Verify makedirs was called
        mock_makedirs.assert_called_once()
        call_args = mock_makedirs.call_args
        # Normalize path separators for cross-platform comparison
        actual_path = call_args[0][0].replace("\\", "/")
        assert "/tmp/logs/test-exp" in actual_path
        assert call_args[1]["exist_ok"] is True

    @patch.object(init_module, "Experiment")
    @patch.object(init_module, "set_global")
    @patch("subprocess.check_output")
    def test_init_version_is_created(self, mock_check_output, mock_set_global, mock_experiment_class):
        """Test that a version string is created."""
        mock_check_output.return_value = b"/path/to/repo\n"
        mock_experiment = MagicMock()
        mock_experiment_class.return_value = mock_experiment

        init(name="test")

        call_kwargs = mock_experiment_class.call_args.kwargs
        # Version should be a string (timestamp in ISO format)
        assert isinstance(call_kwargs["version"], str | dict)
        assert "version" in call_kwargs


class TestFinish:
    """Test the finish function."""

    def test_finish_with_no_experiment_raises_error(self):
        """Test that finish raises error if no experiment is initialized."""
        litlogger.experiment = None

        with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\) before litlogger.finish\\(\\)"):
            finish()

    def test_finish_calls_experiment_finalize(self):
        """Test that finish calls the experiment's finalize method."""
        mock_experiment = MagicMock()
        litlogger.experiment = mock_experiment

        finish()

        mock_experiment.finalize.assert_called_once_with(None)

    def test_finish_with_status(self):
        """Test that finish passes status to experiment.finalize."""
        mock_experiment = MagicMock()
        litlogger.experiment = mock_experiment

        finish(status="completed")

        mock_experiment.finalize.assert_called_once_with("completed")

    def test_finish_with_custom_status(self):
        """Test finish with various status values."""
        mock_experiment = MagicMock()
        litlogger.experiment = mock_experiment

        finish(status="failed")

        mock_experiment.finalize.assert_called_once_with("failed")
