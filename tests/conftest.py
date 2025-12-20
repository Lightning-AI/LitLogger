import os
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import pytest


@contextmanager
def enable_guest_mode():
    cloud_url = os.environ.get("LIGHTNING_CLOUD_URL", "https://lightning.ai")

    # patch out all environment variables to make sure guest mode is enabled
    with patch.dict(os.environ, {"LIGHTNING_CLOUD_URL": cloud_url}, clear=True):
        yield


@pytest.fixture(autouse=True)
def _guest_mode_env():
    is_guest_mode = bool(os.environ.get("TEST_GUEST_MODE", ""))
    ctx = enable_guest_mode if is_guest_mode else nullcontext

    with ctx():
        yield
