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

from typing import TYPE_CHECKING, Any, overload

from litlogger.media import File

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

    def append(self, value: float | File, step: int | None = None) -> None:
        """Append a value to the time series.

        Args:
            value: A float/int for metric series, or a File for file series.
            step: Optional step number for this data point (e.g., training step, epoch).

        Raises:
            TypeError: If value type doesn't match existing series type or is unsupported.
        """
        if isinstance(value, File):
            if self._type is not None and self._type != "file":
                raise TypeError(f"Key {self._key!r} is a metric series, cannot append File")
            if self._type is None:
                self._type = "file"
                self._experiment._register_key_type(self._key, "file_series")
            self._values.append(value)
            self._experiment._log_file_series_value(self._key, value, len(self._values) - 1, step=step)
        elif isinstance(value, int | float):
            if self._type is not None and self._type != "metric":
                raise TypeError(f"Key {self._key!r} is a file series, cannot append numeric value")
            if self._type is None:
                self._type = "metric"
                self._experiment._register_key_type(self._key, "metric")
            float_val = float(value)
            self._values.append(float_val)
            self._experiment._log_metric_value(self._key, float_val, step=step)
        else:
            raise TypeError(f"Can only append float/int or File, got {type(value).__name__}")

    def extend(self, values: list[float | int | File], start_step: int | None = None) -> None:
        """Extend the time series with multiple values.

        Args:
            values: List of values to append.
            start_step: Optional starting step number. Each subsequent value gets
                start_step, start_step+1, start_step+2, etc.
        """
        for i, v in enumerate(values):
            step = start_step + i if start_step is not None else None
            self.append(v, step=step)

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
