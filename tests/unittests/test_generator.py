import random

from litlogger.generator import _create_name


def test_generator():
    random.seed(42)
    assert _create_name() == "deaf-amber-vrpo"
