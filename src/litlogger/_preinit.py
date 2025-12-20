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
"""Pre-initialization wrappers for litlogger to provide helpful error messages."""

from typing import Any, Callable


class PreInitObject:
    """Object that raises an error if accessed before litlogger.init() is called."""

    def __init__(self, name: str, destination: Any | None = None) -> None:
        self._name = name

        if destination is not None:
            self.__doc__ = destination.__doc__

    def __getitem__(self, key: str) -> None:
        raise RuntimeError(f"You must call litlogger.init() before {self._name}[{key!r}]")

    def __setitem__(self, key: str, value: Any) -> Any:
        raise RuntimeError(f"You must call litlogger.init() before {self._name}[{key!r}]")

    def __setattr__(self, key: str, value: Any) -> Any:
        if not key.startswith("_"):
            raise RuntimeError(f"You must call litlogger.init() before {self._name}.{key}")
        return object.__setattr__(self, key, value)

    def __getattr__(self, key: str) -> Any:
        if not key.startswith("_"):
            raise RuntimeError(f"You must call litlogger.init() before {self._name}.{key}")
        raise AttributeError


def pre_init_callable(name: str, destination: Any | None = None) -> Callable:
    """Create a callable that raises an error if called before litlogger.init()."""

    def preinit_wrapper(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"You must call litlogger.init() before {name}()")

    preinit_wrapper.__name__ = str(name)
    if destination:
        preinit_wrapper.__wrapped__ = destination  # type: ignore
        preinit_wrapper.__doc__ = destination.__doc__
    return preinit_wrapper
