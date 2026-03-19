"""Integration tests for lightning.pytorch.loggers.LitLogger."""

import pytest
from lightning.pytorch.loggers import LitLogger

from tests.integrations.lightning_logger_cases import (
    run_console_output,
    run_custom_metadata,
    run_file_and_model_logging,
    run_full_training_integration,
    run_hyperparameter_logging,
    run_save_logs_subprocess,
    run_system_info,
)

pytestmark = pytest.mark.cloud()


def test_full_training_integration(tmpdir):
    run_full_training_integration(LitLogger, name_prefix="pytorch-litlogger-training", tmpdir=tmpdir)


def test_console_output():
    run_console_output(LitLogger, name_prefix="pytorch-litlogger-console")


def test_system_info():
    run_system_info(LitLogger, name_prefix="pytorch-litlogger-system")


def test_file_and_model_logging(tmpdir):
    run_file_and_model_logging(LitLogger, name_prefix="pytorch-litlogger-artifacts", tmpdir=tmpdir)


def test_hyperparameter_logging():
    run_hyperparameter_logging(LitLogger, name_prefix="pytorch-litlogger-hparams")


def test_custom_metadata():
    run_custom_metadata(LitLogger, name_prefix="pytorch-litlogger-metadata")


def test_log_file_creation_with_save_logs(tmpdir):
    run_save_logs_subprocess(
        tmpdir=tmpdir,
        logger_kind="pytorch-litlogger",
        name_prefix="pytorch-litlogger-save-logs",
    )
