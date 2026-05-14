"""Integration tests for the deprecated litlogger.LightningLogger wrapper."""

import pytest
from litlogger import LightningLogger

from tests.integrations.lightning_logger_cases import (
    run_end_to_end_smoke,
)

pytestmark = [
    pytest.mark.cloud(),
    pytest.mark.filterwarnings("ignore:litlogger.LightningLogger is deprecated:FutureWarning"),
    pytest.mark.filterwarnings("ignore:`isinstance\\(treespec, LeafSpec\\)` is deprecated:FutureWarning"),
]


def test_end_to_end_smoke(tmpdir):
    run_end_to_end_smoke(LightningLogger, name_prefix="deprecated-wrapper", tmpdir=tmpdir)
