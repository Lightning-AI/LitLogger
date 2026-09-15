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
"""Series class for time-series data in litlogger experiments."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, overload

from litlogger.primitives import File
from litlogger.primitives.metric import resolve_x

if TYPE_CHECKING:
    from litlogger.experiment import Experiment


class Series:
    """A list-like proxy for time-series data that supports append and extend.

    Tracks either metric values (floats) or file series (File objects).
    The type is determined by the first value appended and cannot be mixed.
    """

    def __init__(self, experiment: Experiment, key: str) -> None:
        self._experiment = experiment
        self._key = key
        self._type: str | None = None  # 'metric' or 'file'
        self._values: list[Any] = []

    def append(
        self,
        y: float | File,
        step: float | None = None,
        x: float | None = None,
    ) -> None:
        """Append a value to the time series.

        Args:
            y: A float/int for metric series, or a File for file series.
            step: Legacy x-coordinate for this data point.
            x: Preferred x-coordinate. Mutually exclusive with ``step``.

        Raises:
            TypeError: If y's type doesn't match the existing series type or is unsupported.
            ValueError: If both ``x`` and ``step`` are provided.
        """
        effective_x = resolve_x(x=x, step=step)
        if isinstance(y, File):
            if self._type is not None and self._type != "file":
                raise TypeError(f"Key {self._key!r} is a metric series, cannot append File")
            new_series = self._type is None
            if new_series:
                self._experiment._register_key_type(self._key, "file_series")
                self._type = "file"
            index = len(self._values)
            file_x = index if effective_x is None else effective_x
            try:
                self._experiment._log_file_series_value(self._key, y, index, step=file_x)
            except Exception:
                if new_series:
                    self._type = None
                    if self._experiment._key_types.get(self._key) == "file_series":
                        self._experiment._key_types.pop(self._key)
                raise
            self._values.append(y)
        elif isinstance(y, int | float):
            if self._type is not None and self._type != "metric":
                raise TypeError(f"Key {self._key!r} is a file series, cannot append numeric value")
            float_val = float(y)
            if math.isnan(float_val) or math.isinf(float_val):
                # FIXME: Remove this when NaN is handled correctly
                warnings.warn(
                    f"Metric '{self._key}' = {float_val} is not a finite number. Skipping.",
                    stacklevel=2,
                )
                return
            new_series = self._type is None
            if new_series:
                self._experiment._register_key_type(self._key, "metric")
                self._type = "metric"
            try:
                self._experiment._log_metric_value(self._key, float_val, x=effective_x)
            except Exception:
                if new_series:
                    self._type = None
                    if self._experiment._key_types.get(self._key) == "metric":
                        self._experiment._key_types.pop(self._key)
                raise
            self._values.append(float_val)
        else:
            raise TypeError(f"Can only append float/int or File, got {type(y).__name__}")

    def extend(
        self,
        values: list[float | int | File],
        start_step: float | None = None,
        start_x: float | None = None,
    ) -> None:
        """Extend the time series with multiple values.

        Args:
            values: List of values to append.
            start_step: Legacy starting x-coordinate.
            start_x: Preferred starting x-coordinate. Mutually exclusive with
                ``start_step``. Each subsequent value increments it by one.

        Raises:
            ValueError: If both ``start_x`` and ``start_step`` are provided.
        """
        coordinate = resolve_x(x=start_x, step=start_step)
        for i, v in enumerate(values):
            x = coordinate + i if coordinate is not None else None
            self.append(v, x=x)

    def __iter__(self) -> Any:  # noqa: D105
        return iter(self._values)

    def __len__(self) -> int:  # noqa: D105
        return len(self._values)

    @overload
    def __getitem__(self, index: int) -> float | File: ...

    @overload
    def __getitem__(self, index: slice) -> list[float | File]: ...

    def __getitem__(self, index: int | slice) -> float | File | list[float | File]:  # noqa: D105
        return self._values[index]

    def __repr__(self) -> str:  # noqa: D105
        return repr(self._values)

    def __eq__(self, other: object) -> bool:
        """Check if this series is equal to another Series or list."""
        if isinstance(other, list):
            return self._values == other
        if isinstance(other, Series):
            return self._values == other._values
        return NotImplemented
