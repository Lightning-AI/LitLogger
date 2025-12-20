import contextlib
from unittest.mock import patch

import pytest
import urllib3
from litlogger.api.client import _retry_wrapper


@pytest.mark.parametrize("tries", [0, 1, 2, 3])
def test_retry_wrapper(tries):
    counter = 0

    def my_func(*_, **__):
        nonlocal counter
        counter += 1
        if counter > 3:
            return "success"
        raise urllib3.exceptions.HTTPError("error")

    # Mock time.sleep to make the test instant
    with patch("litlogger.api.client.time.sleep", return_value=None):
        # Call the function
        wrapped_func = _retry_wrapper(None, my_func, max_retries=tries)

        # Check that the function raises an exception and nothing for case 5
        with pytest.raises(RuntimeError) if tries < 3 else contextlib.nullcontext():
            wrapped_func(my_func)

    # Check that the function was called the expected number of times
    assert counter == tries + 1
