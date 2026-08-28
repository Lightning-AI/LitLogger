# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for adding and retrieving metrics via the experiment dict-like API."""

import sys
from unittest.mock import MagicMock

import pytest

from litlogger.experiment import Experiment
from litlogger.primitives import File, MetricWrite, PrimitiveWrite
from litlogger.series import Series
from litlogger.session import ExperimentSession

experiment_module = sys.modules["litlogger.experiment"]


def _session_of(exp):
    """Build a session view over the experiment's current mock infrastructure."""

    def part(name):
        value = getattr(exp, name, None)
        return value if value is not None else MagicMock()

    metrics_api = part("_metrics_api")
    return ExperimentSession._from_components(
        name=exp.name,
        metrics_api=metrics_api,
        media_api=part("_media_api"),
        artifacts_api=part("_artifacts_api"),
        teamspace=part("_teamspace"),
        metrics_store=part("_metrics_store"),
        experiment=exp,
        queue_=part("_metrics_queue"),
        stats=part("_stats"),
        printer=part("_printer"),
        store_step=bool(getattr(exp, "store_step", True)),
        store_created_at=bool(getattr(exp, "store_created_at", False)),
        last_x=getattr(exp, "_resumed_steps", None) or {},
        background=getattr(exp, "_manager", None),
    )


def _make_exp(**overrides):
    """Create a MagicMock wired for the dict-like experiment API."""
    exp = MagicMock(spec=Experiment)
    exp.name = "exp"
    exp._series = {}
    exp._key_types = {}
    exp._metadata_values = {}
    exp._static_files = {}
    exp._manager = MagicMock()
    exp._manager.exception = None
    exp.store_step = True
    exp.store_created_at = False
    exp._metrics_queue = MagicMock()
    # The dict API queues writes; execute them inline the way the worker would.
    exp._metrics_queue.put.side_effect = lambda item: (
        item.execute(exp._session) if isinstance(item, PrimitiveWrite) else None
    )
    exp._stats = MagicMock()
    exp._metrics_api = MagicMock()
    exp._media_api = MagicMock()
    exp._artifacts_api = MagicMock()
    exp._teamspace = MagicMock()
    exp._metrics_store = MagicMock()
    exp._metrics_store.id = "store-1"
    exp._metrics_store.name = "exp"
    exp._metrics_store.tags = []
    exp._metrics_store.cluster_id = "acc-1"
    # Metadata writes re-read the store from the API; keep the seeded one.
    exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store
    # Live session view so tests can reseed infrastructure after the factory.
    type(exp)._session = property(lambda self: _session_of(self))

    # Wire dunder methods on the type
    type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)
    type(exp).__setitem__ = lambda self, key, value: Experiment.__setitem__(self, key, value)

    # Wire regular methods
    exp.update = lambda data: Experiment.update(exp, data)
    exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
    exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
    exp._log_metric_value = lambda key, y, x=None: Experiment._log_metric_value(exp, key, y, x=x)
    exp._validate_file_primitive = lambda value: Experiment._validate_file_primitive(exp, value)

    for k, v in overrides.items():
        setattr(exp, k, v)
    return exp


# ---------------------------------------------------------------------------
# Adding metrics
# ---------------------------------------------------------------------------


class TestAddMetricAppend:
    """Test experiment['key'].append(value) for metrics."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_batch_size": 0}, "max_batch_size"),
            ({"rate_limiting_interval": -1}, "rate_limiting_interval"),
        ],
    )
    def test_rejects_invalid_batch_configuration_before_initialization(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            Experiment("invalid", **kwargs)

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

        command = exp._metrics_queue.put.call_args[0][0]
        assert command == MetricWrite(key="loss", y=0.5, x=3, created_at=None)

    def test_x_and_step_are_mutually_exclusive(self):
        exp = _make_exp()

        with pytest.raises(ValueError, match="mutually exclusive"):
            exp["loss"].append(y=0.5, step=3, x=1.5)

        exp._metrics_queue.put.assert_not_called()

    @pytest.mark.parametrize("x", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_x_does_not_mutate_series(self, x):
        exp = _make_exp()

        with pytest.raises(ValueError, match="finite"):
            exp["loss"].append(y=0.5, x=x)

        assert list(exp["loss"]) == []
        assert "loss" not in exp._key_types
        exp._metrics_queue.put.assert_not_called()

    def test_append_respects_store_step_false(self):
        exp = _make_exp(store_step=False)
        exp["loss"].append(0.5, step=99)

        command = exp._metrics_queue.put.call_args[0][0]
        assert command == MetricWrite(key="loss", y=0.5, x=99, created_at=None)

    def test_append_records_stats(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        exp._stats.record_metric.assert_called_once_with("loss", 0.5)

    def test_append_raises_on_background_exception(self):
        exp = _make_exp()
        exp._manager.exception = RuntimeError("bg error")

        with pytest.raises(RuntimeError, match="bg error"):
            exp["loss"].append(0.5)

    def test_queue_failure_does_not_mutate_new_series(self):
        exp = _make_exp()
        series = exp["loss"]
        exp._metrics_queue.put.side_effect = RuntimeError("queue failed")

        with pytest.raises(RuntimeError, match="queue failed"):
            series.append(0.5)

        assert list(series) == []
        assert series._type is None
        assert "loss" not in exp._key_types


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
        assert calls[0].args[0].x == 10
        assert calls[1].args[0].x == 11

    def test_extend_start_x_and_start_step_are_mutually_exclusive(self):
        exp = _make_exp()

        with pytest.raises(ValueError, match="mutually exclusive"):
            exp["loss"].extend([1.0, 0.5], start_step=10, start_x=0.5)

        exp._metrics_queue.put.assert_not_called()

    def test_extend_rejects_non_finite_start_x_before_mutation(self):
        exp = _make_exp()

        with pytest.raises(ValueError, match="finite"):
            exp["loss"].extend([1.0, 0.5], start_x=float("inf"))

        assert list(exp["loss"]) == []
        exp._metrics_queue.put.assert_not_called()

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
