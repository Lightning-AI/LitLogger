# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for the Series class."""

from unittest.mock import MagicMock

import pytest
from litlogger.experiment import Experiment
from litlogger.media import File
from litlogger.series import Series


class TestSeriesAppend:
    """Test Series.append behavior."""

    def test_append_float(self):
        """Test appending a float to a series."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.append(0.5)

        assert len(series) == 1
        assert series[0] == 0.5
        assert series._type == "metric"
        exp._register_key_type.assert_called_once_with("loss", "metric")
        exp._log_metric_value.assert_called_once_with("loss", 0.5, step=None)

    def test_append_int(self):
        """Test appending an int to a series (converted to float)."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "count")
        series.append(42)

        assert series[0] == 42.0
        assert isinstance(series[0], float)

    def test_append_file(self):
        """Test appending a File to a series."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        f = File("test.txt")
        series = Series(exp, "files")
        series.append(f)

        assert len(series) == 1
        assert series[0] is f
        assert series._type == "file"
        exp._register_key_type.assert_called_once_with("files", "file_series")
        exp._log_file_series_value.assert_called_once_with("files", f, 0, step=None)

    def test_append_with_step(self):
        """Test appending a metric with an explicit step."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.append(0.5, step=10)

        exp._log_metric_value.assert_called_once_with("loss", 0.5, step=10)

    def test_append_file_passes_step_through(self):
        """Test that file-like series preserve the provided step."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        f = File("test.txt")
        series = Series(exp, "files")
        series.append(f, step=5)

        assert len(series) == 1
        exp._log_file_series_value.assert_called_once_with("files", f, 0, step=5)

    def test_append_invalid_type(self):
        """Test that appending unsupported type raises TypeError."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "bad")

        with pytest.raises(TypeError, match="Can only append"):
            series.append("string")  # type: ignore[arg-type]

    def test_append_multiple_floats(self):
        """Test appending multiple floats accumulates values."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.append(0.5, step=0)
        series.append(0.3, step=1)
        series.append(0.1, step=2)

        assert len(series) == 3
        assert list(series) == [0.5, 0.3, 0.1]
        assert exp._log_metric_value.call_count == 3

    def test_append_multiple_files(self):
        """Test appending multiple files accumulates with correct indices."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        f1 = File("a.txt")
        f2 = File("b.txt")
        series = Series(exp, "files")
        series.append(f1)
        series.append(f2)

        assert len(series) == 2
        calls = exp._log_file_series_value.call_args_list
        assert calls[0] == (("files", f1, 0), {"step": None})
        assert calls[1] == (("files", f2, 1), {"step": None})


class TestSeriesTypeSafety:
    """Test that Series enforces type consistency."""

    def test_cannot_mix_metric_then_file(self):
        """Test that mixing float and File in same series raises TypeError."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "mixed")
        series.append(0.5)

        with pytest.raises(TypeError, match="metric series"):
            series.append(File("test.txt"))

    def test_cannot_mix_file_then_metric(self):
        """Test that mixing File and float in same series raises TypeError."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "mixed")
        series.append(File("test.txt"))

        with pytest.raises(TypeError, match="file series"):
            series.append(0.5)

    def test_type_set_on_first_append(self):
        """Test that _type is None before first append and set after."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "key")

        assert series._type is None
        series.append(1.0)
        assert series._type == "metric"

    def test_register_key_type_called_once(self):
        """Test that _register_key_type is only called on first append."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.append(1.0)
        series.append(2.0)
        series.append(3.0)

        exp._register_key_type.assert_called_once_with("loss", "metric")


class TestSeriesExtend:
    """Test Series.extend behavior."""

    def test_extend(self):
        """Test extending a series with multiple values."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.extend([0.5, 0.3, 0.1])

        assert len(series) == 3
        assert list(series) == [0.5, 0.3, 0.1]

    def test_extend_with_start_step(self):
        """Test extending with start_step assigns incrementing steps."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.extend([0.5, 0.3, 0.1], start_step=100)

        calls = exp._log_metric_value.call_args_list
        assert len(calls) == 3
        assert calls[0] == (("loss", 0.5), {"step": 100})
        assert calls[1] == (("loss", 0.3), {"step": 101})
        assert calls[2] == (("loss", 0.1), {"step": 102})

    def test_extend_without_start_step(self):
        """Test extending without start_step passes step=None."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.extend([1.0, 2.0])

        calls = exp._log_metric_value.call_args_list
        assert calls[0] == (("loss", 1.0), {"step": None})
        assert calls[1] == (("loss", 2.0), {"step": None})

    def test_extend_empty_list(self):
        """Test extending with empty list is a no-op."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.extend([])

        assert len(series) == 0
        exp._log_metric_value.assert_not_called()

    def test_extend_files(self):
        """Test extending with File objects."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        f1 = File("a.png")
        f2 = File("b.png")
        series = Series(exp, "images")
        series.extend([f1, f2])

        assert len(series) == 2
        assert series._type == "file"


class TestSeriesContainerProtocol:
    """Test Series list-like container behavior."""

    def test_len(self):
        """Test len() on a series."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        assert len(series) == 0
        series.append(1.0)
        assert len(series) == 1

    def test_iteration(self):
        """Test iterating over a series."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.extend([1.0, 2.0, 3.0])

        values = list(series)
        assert values == [1.0, 2.0, 3.0]

    def test_indexing(self):
        """Test integer indexing."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.extend([1.0, 2.0, 3.0])

        assert series[0] == 1.0
        assert series[-1] == 3.0

    def test_slicing(self):
        """Test slicing a series."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.extend([1.0, 2.0, 3.0, 4.0])

        assert series[1:3] == [2.0, 3.0]

    def test_index_out_of_range(self):
        """Test that out-of-range index raises IndexError."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")

        with pytest.raises(IndexError):
            _ = series[0]

    def test_equality_with_list(self):
        """Test series equality comparison with a list."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.extend([1.0, 2.0, 3.0])

        assert series == [1.0, 2.0, 3.0]
        assert series != [1.0, 2.0]

    def test_equality_with_series(self):
        """Test series equality comparison with another Series."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        s1 = Series(exp, "a")
        s2 = Series(exp, "b")
        s1.extend([1.0, 2.0])
        s2.extend([1.0, 2.0])

        assert s1 == s2

    def test_inequality_with_unrelated_type(self):
        """Test that comparing with unrelated type returns NotImplemented."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")

        assert series.__eq__("not a series") is NotImplemented

    def test_repr(self):
        """Test repr shows internal values."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        series = Series(exp, "loss")
        series.extend([1.0, 2.0])

        assert repr(series) == "[1.0, 2.0]"
