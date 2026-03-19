"""Integration tests for the deprecated litlogger.LightningLogger wrapper."""

import pytest
from litlogger import LightningLogger

from tests.integrations.lightning_logger_cases import (
    run_custom_metadata,
    run_file_logging,
    run_full_training_integration,
    run_hyperparameter_logging,
    run_save_logs_subprocess,
)

pytestmark = [
    pytest.mark.cloud(),
    pytest.mark.filterwarnings("ignore:litlogger.LightningLogger is deprecated:FutureWarning"),
    pytest.mark.filterwarnings("ignore:`isinstance\\(treespec, LeafSpec\\)` is deprecated:FutureWarning"),
]


def test_full_training_integration(tmpdir):
    run_full_training_integration(LightningLogger, name_prefix="deprecated-wrapper-training", tmpdir=tmpdir)


def test_file_logging(tmpdir):
    run_file_logging(LightningLogger, name_prefix="deprecated-wrapper-files", tmpdir=tmpdir)


def test_hyperparameter_logging():
    run_hyperparameter_logging(LightningLogger, name_prefix="deprecated-wrapper-hparams")


def test_custom_metadata():
    run_custom_metadata(LightningLogger, name_prefix="deprecated-wrapper-metadata")


def test_log_file_creation_with_save_logs(tmpdir):
    run_save_logs_subprocess(
        tmpdir=tmpdir,
        logger_kind="deprecated-wrapper",
        name_prefix="deprecated-wrapper-save-logs",
    )
