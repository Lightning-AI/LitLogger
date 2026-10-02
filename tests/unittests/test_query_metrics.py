# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for querying and filtering metrics."""

from unittest.mock import MagicMock

import pytest
from litlogger.experiment import Experiment
from litlogger.primitives import File, PrimitiveWrite
from litlogger.session import ExperimentSession
from litlogger.types import MetricSummary


# ---------------------------------------------------------------------------
# Helpers (reuse the same factory from test_experiment_metrics)
# ---------------------------------------------------------------------------


def _session_of(exp):
    def part(name):
        value = getattr(exp, name, None)
        return value if value is not None else MagicMock()

    return ExperimentSession._from_components(
        name=exp.name,
        metrics_api=part("_metrics_api"),
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
    exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store
    type(exp)._session = property(lambda self: _session_of(self))
    type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)
    type(exp).__setitem__ = lambda self, key, value: Experiment.__setitem__(self, key, value)
    exp.update = lambda data: Experiment.update(exp, data)
    exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
    exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
    exp._log_metric_value = lambda key, y, x=None: Experiment._log_metric_value(exp, key, y, x=x)
    exp._validate_file_primitive = lambda value: Experiment._validate_file_primitive(exp, value)
    exp.query_metrics = lambda **kw: Experiment.query_metrics(exp, **kw)

    for k, v in overrides.items():
        setattr(exp, k, v)
    return exp


# ---------------------------------------------------------------------------
# MetricSummary.from_values()
# ---------------------------------------------------------------------------


class TestMetricSummaryFromValues:
    """Test the MetricSummary.from_values() classmethod."""

    def test_basic(self):
        s = MetricSummary.from_values("loss", [1.0, 0.5, 0.25, 0.1])
        assert s.name == "loss"
        assert s.count == 4
        assert s.min == 0.1
        assert s.max == 1.0
        assert s.mean == pytest.approx(0.4625)
        assert s.first == 1.0
        assert s.last == 0.1

    def test_std(self):
        s = MetricSummary.from_values("x", [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert s.std == pytest.approx(2.0)

    def test_median_odd(self):
        s = MetricSummary.from_values("x", [1.0, 3.0, 5.0])
        assert s.median == 3.0

    def test_median_even(self):
        s = MetricSummary.from_values("x", [1.0, 2.0, 3.0, 4.0])
        assert s.median == 2.5

    def test_single_value(self):
        s = MetricSummary.from_values("x", [3.14])
        assert s.count == 1
        assert s.min == s.max == s.mean == s.first == s.last == s.median == 3.14
        assert s.std == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty values"):
            MetricSummary.from_values("x", [])


# ---------------------------------------------------------------------------
# Series.summary()
# ---------------------------------------------------------------------------


class TestSeriesSummary:
    """Test Series.summary() method."""

    def test_summary_basic(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.25, 0.1])

        s = exp["loss"].summary()

        assert isinstance(s, MetricSummary)
        assert s.name == "loss"
        assert s.count == 4
        assert s.min == 0.1
        assert s.max == 1.0
        assert s.mean == pytest.approx(0.4625)
        assert s.first == 1.0
        assert s.last == 0.1
        assert s.std == pytest.approx(0.34164, rel=1e-3)
        assert s.median == pytest.approx(0.375)

    def test_summary_single_value(self):
        exp = _make_exp()
        exp["loss"].append(3.14)

        s = exp["loss"].summary()
        assert s.count == 1
        assert s.min == s.max == s.mean == s.first == s.last == s.median == 3.14
        assert s.std == 0.0

    def test_summary_empty_series_raises(self):
        exp = _make_exp()
        series = exp["loss"]

        with pytest.raises(ValueError, match="empty"):
            series.summary()

    def test_summary_to_dict(self):
        exp = _make_exp()
        exp["acc"].extend([0.8, 0.9, 0.95])

        d = exp["acc"].summary().to_dict()
        assert d["name"] == "acc"
        assert d["count"] == 3
        assert d["min"] == 0.8
        assert d["max"] == 0.95
        assert d["mean"] == pytest.approx(0.8833333)
        assert d["last"] == 0.95
        assert d["first"] == 0.8
        assert "std" in d
        assert "median" in d


# ---------------------------------------------------------------------------
# Series.filter()
# ---------------------------------------------------------------------------


class TestSeriesFilter:
    """Test Series.filter() method."""

    def test_filter_no_constraints(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.1])

        assert exp["loss"].filter() == [1.0, 0.5, 0.1]

    def test_filter_min_value(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.25, 0.1])

        assert exp["loss"].filter(min_value=0.3) == [1.0, 0.5]

    def test_filter_max_value(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.25, 0.1])

        assert exp["loss"].filter(max_value=0.5) == [0.5, 0.25, 0.1]

    def test_filter_min_and_max(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.25, 0.1])

        assert exp["loss"].filter(min_value=0.2, max_value=0.6) == [0.5, 0.25]

    def test_filter_index_range(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.8, 0.6, 0.4, 0.2])

        assert exp["loss"].filter(start_index=1, end_index=4) == [0.8, 0.6, 0.4]

    def test_filter_index_and_value(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.8, 0.6, 0.4, 0.2])

        result = exp["loss"].filter(start_index=1, end_index=4, min_value=0.5)
        assert result == [0.8, 0.6]

    def test_filter_empty_result(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 2.0, 3.0])

        assert exp["loss"].filter(min_value=10.0) == []

    def test_filter_untyped_series_returns_empty(self):
        """An untyped (never-appended-to) series is treated as empty metric series."""
        exp = _make_exp()
        _ = exp["loss"]

        assert exp["loss"].filter() == []

    def test_filter_inclusive_boundaries(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 2.0, 3.0, 4.0, 5.0])

        assert exp["loss"].filter(min_value=2.0, max_value=4.0) == [2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Series type guards for filter/summary
# ---------------------------------------------------------------------------


class TestSeriesTypeGuards:
    """Ensure filter/summary reject file series."""

    def test_summary_rejects_file_series(self):
        exp = _make_exp()
        exp._log_file_series_value = MagicMock()
        exp["images"].append(File("a.png"))

        with pytest.raises(TypeError, match="metric series"):
            exp["images"].summary()

    def test_filter_rejects_file_series(self):
        exp = _make_exp()
        exp._log_file_series_value = MagicMock()
        exp["images"].append(File("a.png"))

        with pytest.raises(TypeError, match="metric series"):
            exp["images"].filter()


# ---------------------------------------------------------------------------
# Experiment.query_metrics()
# ---------------------------------------------------------------------------


class TestQueryMetrics:
    """Test Experiment.query_metrics() method."""

    def test_query_all_metrics(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.1])
        exp["acc"].extend([0.5, 0.8, 0.95])

        result = exp.query_metrics()

        assert "loss" in result
        assert "acc" in result
        assert result["loss"] == [1.0, 0.5, 0.1]
        assert result["acc"] == [0.5, 0.8, 0.95]

    def test_query_by_name_exact(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5])
        exp["acc"].extend([0.8, 0.9])

        result = exp.query_metrics(name="loss")
        assert list(result.keys()) == ["loss"]

    def test_query_by_name_glob(self):
        exp = _make_exp()
        exp["train/loss"].extend([1.0, 0.5])
        exp["train/acc"].extend([0.8, 0.9])
        exp["val/loss"].extend([1.2, 0.6])

        result = exp.query_metrics(name="train/*")
        assert set(result.keys()) == {"train/loss", "train/acc"}

    def test_query_by_name_wildcard_all(self):
        exp = _make_exp()
        exp["train/loss"].extend([1.0])
        exp["val/loss"].extend([1.2])

        result = exp.query_metrics(name="*/loss")
        assert set(result.keys()) == {"train/loss", "val/loss"}

    def test_query_with_value_filter(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.25, 0.1])

        result = exp.query_metrics(min_value=0.2, max_value=0.6)
        assert result["loss"] == [0.5, 0.25]

    def test_query_with_index_range(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.8, 0.6, 0.4, 0.2])

        result = exp.query_metrics(start_index=2, end_index=4)
        assert result["loss"] == [0.6, 0.4]

    def test_query_excludes_non_metric_series(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5])
        exp["tag"] = "v1"

        result = exp.query_metrics()
        assert "loss" in result
        assert "tag" not in result

    def test_query_summary_only(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.1])
        exp["acc"].extend([0.5, 0.8, 0.95])

        result = exp.query_metrics(summary_only=True)

        assert isinstance(result["loss"], MetricSummary)
        assert result["loss"].count == 3
        assert result["loss"].min == 0.1
        assert result["loss"].max == 1.0
        assert result["loss"].std > 0
        assert result["acc"].last == 0.95

    def test_query_summary_with_filter(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 0.5, 0.25, 0.1])

        result = exp.query_metrics(min_value=0.3, summary_only=True)

        assert result["loss"].count == 2
        assert result["loss"].min == 0.5
        assert result["loss"].max == 1.0

    def test_query_summary_skips_empty_after_filter(self):
        exp = _make_exp()
        exp["loss"].extend([1.0, 2.0])
        exp["acc"].extend([0.1, 0.2])

        result = exp.query_metrics(min_value=5.0, summary_only=True)
        assert result == {}

    def test_query_no_match_returns_empty(self):
        exp = _make_exp()
        exp["loss"].extend([1.0])

        result = exp.query_metrics(name="nonexistent")
        assert result == {}

    def test_query_combined_name_and_value_filter(self):
        exp = _make_exp()
        exp["train/loss"].extend([1.0, 0.5, 0.1])
        exp["val/loss"].extend([1.2, 0.3, 0.05])
        exp["train/acc"].extend([0.5, 0.8, 0.95])

        result = exp.query_metrics(name="*/loss", max_value=0.5)
        assert set(result.keys()) == {"train/loss", "val/loss"}
        assert result["train/loss"] == [0.5, 0.1]
        assert result["val/loss"] == [0.3, 0.05]

    def test_query_name_is_keyword_only(self):
        """name must be passed as a keyword argument."""
        exp = _make_exp()
        exp["loss"].extend([1.0])

        with pytest.raises(TypeError):
            exp.query_metrics("loss")  # type: ignore[misc]
