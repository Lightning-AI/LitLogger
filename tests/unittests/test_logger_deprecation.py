import pytest


def test_lightning_logger_warns_on_init():
    try:
        from litlogger import LightningLogger
    except ImportError:
        pytest.skip("LightningLogger is unavailable without Lightning dependencies")

    with pytest.warns(DeprecationWarning, match="litlogger.LightningLogger is deprecated"):
        LightningLogger(name="deprecated-test")
