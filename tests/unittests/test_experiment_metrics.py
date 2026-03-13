# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for adding and retrieving metrics via the experiment dict-like API."""

import sys
from unittest.mock import MagicMock

import pytest
from litlogger.experiment import Experiment
from litlogger.media import File
from litlogger.series import Series

experiment_module = sys.modules["litlogger.experiment"]


def _make_exp(**overrides):
    """Create a MagicMock wired for the dict-like experiment API."""
    exp = MagicMock(spec=Experiment)
    exp._series = {}
    exp._key_types = {}
    exp._metadata_values = {}
    exp._static_files = {}
    exp._manager = MagicMock()
    exp._manager.exception = None
    exp.store_step = True
    exp.store_created_at = False
    exp._metrics_queue = MagicMock()
    exp._stats = MagicMock()

    # Wire dunder methods on the type
    type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)
    type(exp).__setitem__ = lambda self, key, value: Experiment.__setitem__(self, key, value)

    # Wire regular methods
    exp.update = lambda data: Experiment.update(exp, data)
    exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
    exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
    exp._log_metric_value = lambda key, value, step=None: Experiment._log_metric_value(exp, key, value, step=step)

    for k, v in overrides.items():
        setattr(exp, k, v)
    return exp


# ---------------------------------------------------------------------------
# Adding metrics
# ---------------------------------------------------------------------------


class TestAddMetricAppend:
    """Test experiment['key'].append(value) for metrics."""

    def test_append_single_value(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        assert len(exp["loss"]) == 1
        assert exp["loss"][0] == 0.5
        assert exp._key_types["loss"] == "metric"

    def test_append_multiple_values(self):
        exp = _make_exp()
        exp["loss"].append(1.0, step=0)
        exp["loss"].append(0.5, step=1)
        exp["loss"].append(0.1, step=2)

        assert list(exp["loss"]) == [1.0, 0.5, 0.1]

    def test_append_int_converted_to_float(self):
        exp = _make_exp()
        exp["count"].append(42)

        assert exp["count"][0] == 42.0
        assert isinstance(exp["count"][0], float)

    def test_append_pushes_to_queue(self):
        exp = _make_exp()
        exp["loss"].append(0.5, step=3)

        batch = exp._metrics_queue.put.call_args[0][0]
        assert "loss" in batch
        assert batch["loss"].values[0].value == 0.5
        assert batch["loss"].values[0].step == 3

    def test_append_respects_store_step_false(self):
        exp = _make_exp(store_step=False)
        exp["loss"].append(0.5, step=99)

        batch = exp._metrics_queue.put.call_args[0][0]
        assert batch["loss"].values[0].step is None

    def test_append_records_stats(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        exp._stats.record_metric.assert_called_once_with("loss", 0.5)

    def test_append_raises_on_background_exception(self):
        exp = _make_exp()
        exp._manager.exception = RuntimeError("bg error")

        with pytest.raises(RuntimeError, match="bg error"):
            exp["loss"].append(0.5)


class TestAddMetricExtend:
    """Test experiment['key'].extend(values) for metrics."""

    def test_extend_list(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.1])

        assert list(exp["loss"]) == [1.0, 0.5, 0.1]
        assert exp._metrics_queue.put.call_count == 3

    def test_extend_with_start_step(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5], start_step=10)

        calls = exp._metrics_queue.put.call_args_list
        assert calls[0][0][0]["loss"].values[0].step == 10
        assert calls[1][0][0]["loss"].values[0].step == 11

    def test_extend_empty_list(self):
        exp = _make_exp()
        exp["loss"].extend([])

        assert len(exp["loss"]) == 0
        exp._metrics_queue.put.assert_not_called()


class TestAddMetricUpdate:
    """Test experiment.update() for metrics."""

    def test_update_single_metric(self):
        exp = _make_exp()
        exp.update({"loss": 0.5})

        assert exp["loss"][0] == 0.5
        assert exp._key_types["loss"] == "metric"

    def test_update_multiple_metrics(self):
        exp = _make_exp()
        exp.update({"loss": 0.5, "acc": 0.9})

        assert exp["loss"][0] == 0.5
        assert exp["acc"][0] == 0.9

    def test_update_list_extends_series(self):
        exp = _make_exp()
        exp.update({"loss": [1.0, 0.5, 0.1]})

        assert list(exp["loss"]) == [1.0, 0.5, 0.1]

    def test_update_mixed_types(self):
        """Update can mix metrics, metadata, and files in one call."""
        exp = _make_exp()
        exp._set_metadata_value = MagicMock()
        exp._set_static_file = MagicMock()

        exp.update(
            {
                "loss": 0.5,
                "tag": "v1",
                "config": File("config.yaml"),
                "scores": [0.8, 0.9],
            }
        )

        assert exp["loss"][0] == 0.5
        assert exp._metadata_values["tag"] == "v1"
        assert exp._key_types["config"] == "static_file"
        assert list(exp["scores"]) == [0.8, 0.9]


# ---------------------------------------------------------------------------
# Retrieving metrics
# ---------------------------------------------------------------------------


class TestRetrieveMetricByKey:
    """Test experiment['key'] retrieval for metrics."""

    def test_getitem_returns_series(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        result = exp["loss"]
        assert isinstance(result, Series)
        assert result[0] == 0.5

    def test_getitem_new_key_returns_empty_series(self):
        exp = _make_exp()
        result = exp["new_key"]

        assert isinstance(result, Series)
        assert len(result) == 0

    def test_getitem_same_key_returns_same_series(self):
        exp = _make_exp()
        s1 = exp["loss"]
        s2 = exp["loss"]
        assert s1 is s2

    def test_series_indexing(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 2.0, 3.0])

        assert exp["loss"][0] == 1.0
        assert exp["loss"][-1] == 3.0
        assert exp["loss"][1:3] == [2.0, 3.0]

    def test_series_iteration(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 2.0, 3.0])

        assert list(exp["loss"]) == [1.0, 2.0, 3.0]

    def test_series_len(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 2.0])

        assert len(exp["loss"]) == 2

    def test_series_equality_with_list(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 2.0])

        assert exp["loss"] == [1.0, 2.0]


class TestRetrieveMetricsProperty:
    """Test experiment.metrics property."""

    def test_metrics_returns_metric_series_only(self):
        exp = _make_exp()
        exp._set_metadata_value = MagicMock()
        exp._set_static_file = MagicMock()

        exp["loss"].append(0.5)
        exp["acc"].append(0.9)
        exp["tag"] = "v1"

        result = Experiment.metrics.fget(exp)
        assert "loss" in result
        assert "acc" in result
        assert "tag" not in result

    def test_metrics_empty(self):
        exp = _make_exp()
        result = Experiment.metrics.fget(exp)
        assert result == {}


# ---------------------------------------------------------------------------
# Type conflicts
# ---------------------------------------------------------------------------


class TestMetricTypeConflicts:
    """Test that metric keys conflict properly with other types."""

    def test_cannot_assign_string_to_metric_key(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        with pytest.raises(KeyError, match="time series"):
            exp["loss"] = "oops"

    def test_cannot_assign_file_to_metric_key(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        with pytest.raises(KeyError, match="time series"):
            exp["loss"] = File("data.txt")

    def test_cannot_append_file_to_metric_series(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        with pytest.raises(TypeError, match="metric series"):
            exp["loss"].append(File("data.txt"))

    def test_cannot_append_metric_to_file_series(self):
        exp = _make_exp()
        exp._log_file_series_value = MagicMock()
        exp["images"].append(File("img.png"))

        with pytest.raises(TypeError, match="file series"):
            exp["images"].append(0.5)
