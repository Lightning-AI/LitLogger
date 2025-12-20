# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Retry-enabled REST client for Lightning Cloud.

Provides a thin wrapper around GridRestClient that decorates API calls with an
exponential backoff strategy for transient network/server errors.
"""

import time
from functools import wraps
from logging import Logger
from typing import Any, Callable

import urllib3
from lightning_sdk.lightning_cloud.openapi.rest import ApiException
from lightning_sdk.lightning_cloud.rest_client import GridRestClient, _get_next_backoff_time, create_swagger_client

logger = Logger(__name__)


def _should_retry(ex: BaseException) -> bool:
    """Return True if the exception is transient and the request should be retried."""
    if isinstance(ex, urllib3.exceptions.HTTPError):
        return True

    if "not found" in str(ex):
        return False

    if str(ex.status).startswith("4") and ex.status not in (400, 401, 404):
        return True

    return str(ex.status).startswith("5")


def _retry_wrapper(self: Any, func: Callable, max_retries: int = -1) -> Callable:
    """Returns the function decorated by a wrapper that retries the call several times if a connection error occurs.

    The retries follow an exponential backoff.

    """

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        consecutive_errors = 0

        while True:
            try:
                return func(self, *args, **kwargs)
            except (ApiException, urllib3.exceptions.HTTPError) as ex:
                if not _should_retry(ex):
                    raise ex

                msg = f"error: {ex!s}" if isinstance(ex, urllib3.exceptions.HTTPError) else f"response: {ex.status}"

                if consecutive_errors == max_retries:
                    raise RuntimeError(f"The {func.__name__} request failed to reach the server, {msg}.") from ex

                consecutive_errors += 1
                backoff_time = _get_next_backoff_time(consecutive_errors)
                logger.warning(
                    f"The {func.__name__} request failed to reach the server, {msg}."
                    f" Retrying after {backoff_time} seconds."
                )

                time.sleep(backoff_time)

    return wrapped


class LitRestClient(GridRestClient):
    """The LitRestClient is a wrapper around the GridRestClient.

    It wraps all methods to monitor connection exceptions and employs a retry strategy.

    Args:
        max_retries: Maximum number of attempts where each delay between retries is exponential.
            If set to -1, it will retry forever, in contrast if set 0, it runs it only once.

    """

    def __init__(self, max_retries: int = -1) -> None:
        super().__init__(api_client=create_swagger_client())
        if max_retries == 0:
            return
        for base_class in GridRestClient.__mro__:
            for name, attribute in base_class.__dict__.items():
                if callable(attribute) and attribute.__name__ != "__init__":
                    setattr(
                        self,
                        name,
                        _retry_wrapper(self, attribute, max_retries=max_retries),
                    )
