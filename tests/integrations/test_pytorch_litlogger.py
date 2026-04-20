"""Integration tests for lightning.pytorch.loggers.LitLogger."""

import pytest
from lightning.pytorch.loggers import LitLogger

from tests.integrations.lightning_logger_cases import (
    run_end_to_end_smoke,
)

pytestmark = [
    pytest.mark.cloud(),
    pytest.mark.filterwarnings("ignore:`isinstance\\(treespec, LeafSpec\\)` is deprecated:FutureWarning"),
]


def test_end_to_end_smoke(tmpdir):
    run_end_to_end_smoke(LitLogger, name_prefix="pytorch-litlogger", tmpdir=tmpdir)
