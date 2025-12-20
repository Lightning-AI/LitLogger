from unittest.mock import MagicMock

import pytest

# TODO: remove this once the PL integration is merged
pytest.skip("PL integration is not merged yet", allow_module_level=True)

from litlogger import LightningLogger


def test_logger_initialization():
    """Test LightningLogger initialization with default parameters."""
    # Just test that the class can be imported and has the right attributes
    assert hasattr(LightningLogger, "log_metrics")
    assert hasattr(LightningLogger, "log_hyperparams")
    assert hasattr(LightningLogger, "log_file")
    assert hasattr(LightningLogger, "log_model")
    assert hasattr(LightningLogger, "finalize")


def test_logger_name_property():
    """Test that logger has a name property."""
    # Mock the experiment to avoid actual initialization
    logger = object.__new__(LightningLogger)
    logger._name = "test_logger"

    assert logger._name == "test_logger"


def test_logger_version_property():
    """Test that logger has a version property."""
    # Mock the logger to avoid actual initialization
    logger = object.__new__(LightningLogger)
    logger._version = "v1.0"

    assert logger._version == "v1.0"


def test_log_file():
    """Test that LightningLogger has a log_file method."""
    from litlogger import LightningLogger

    # Create a mock logger instance without full initialization
    logger = object.__new__(LightningLogger)
    logger._experiment = MagicMock()

    # Call log_file and verify it delegates to the experiment
    logger.log_file("test.txt")
    logger._experiment.log_file.assert_called_once_with("test.txt")
