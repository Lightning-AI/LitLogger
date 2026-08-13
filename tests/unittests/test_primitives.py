# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for the logging primitives against a mocked session."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from litlogger.primitives import Metadata, Metric, Primitive, _QueuedWrite
from litlogger.session import ExperimentSession
from litlogger.types import PhaseType


def make_session(**overrides):
    """Build a real ExperimentSession wired with mock infrastructure."""
    experiment = MagicMock()
    experiment.name = "exp"
    store = MagicMock()
    store.id = "ms-1"
    store.name = "exp"
    store.tags = []
    store.cluster_id = "acc-1"
    experiment._metrics_store = store
    teamspace = MagicMock()
    teamspace.id = "ts-1"
    background = MagicMock()
    background.exception = None
    session = ExperimentSession(
        client=MagicMock(),
        metrics_api=MagicMock(),
        media_api=MagicMock(),
        artifacts_api=MagicMock(),
        teamspace=teamspace,
        experiment=experiment,
        queue=MagicMock(),
        stats=MagicMock(),
        store_step=True,
        store_created_at=False,
        last_steps={},
        background=background,
    )
    for key, value in overrides.items():
        setattr(session, key, value)
    return session


class TestPrimitiveContract:
    """Every primitive satisfies the runtime Primitive protocol."""

    @pytest.mark.parametrize(
        "primitive",
        [
            Metric("loss", 0.5),
            Metadata("lr", "0.001"),
        ],
        ids=["metric", "metadata"],
    )
    def test_satisfies_protocol(self, primitive):
        assert isinstance(primitive, Primitive)

    def test_static_conformance(self):
        # Typed assignments verified by mypy; runtime smoke check of the same shape.
        _p: Primitive = Metric("loss", 1.0)
        _q: Primitive = Metadata("k", "v")
        assert callable(_p.log) and callable(_p.enqueue)
        assert callable(_q.log) and callable(_q.enqueue)


class TestMetricEnqueue:
    """Metric.enqueue feeds the background batching path."""

    def test_puts_single_value_batch(self):
        session = make_session()

        Metric("loss", 0.5, step=3).enqueue(session)

        session.queue.put.assert_called_once()
        batch = session.queue.put.call_args[0][0]
        assert list(batch.keys()) == ["loss"]
        metrics = batch["loss"]
        assert metrics.name == "loss"
        assert len(metrics.values) == 1
        assert metrics.values[0].value == 0.5
        assert metrics.values[0].step == 3
        assert metrics.values[0].created_at is None
        session.stats.record_metric.assert_called_once_with("loss", 0.5)

    def test_store_step_false_drops_step(self):
        session = make_session(store_step=False)

        Metric("loss", 0.5, step=3).enqueue(session)

        batch = session.queue.put.call_args[0][0]
        assert batch["loss"].values[0].step is None

    def test_store_created_at_sets_timestamp(self):
        session = make_session(store_created_at=True)

        Metric("loss", 0.5).enqueue(session)

        batch = session.queue.put.call_args[0][0]
        assert isinstance(batch["loss"].values[0].created_at, datetime)

    def test_raises_before_put_on_background_failure(self):
        session = make_session()
        session.background.exception = RuntimeError("bg error")

        with pytest.raises(RuntimeError, match="bg error"):
            Metric("loss", 0.5).enqueue(session)

        session.queue.put.assert_not_called()
        session.stats.record_metric.assert_not_called()


class TestMetricLog:
    """Metric.log appends synchronously with worker-equivalent auto-stepping."""

    def test_appends_immediately(self):
        session = make_session()

        Metric("loss", 0.5, step=7).log(session)

        session.metrics_api.append_experiment_metrics.assert_called_once()
        kwargs = session.metrics_api.append_experiment_metrics.call_args.kwargs
        assert kwargs["teamspace_id"] == "ts-1"
        assert kwargs["metrics_store_id"] == "ms-1"
        (metrics,) = kwargs["metrics"]
        assert metrics.name == "loss"
        assert metrics.values[0].value == 0.5
        assert metrics.values[0].step == 7
        session.stats.record_metric.assert_called_once_with("loss", 0.5)

    def test_auto_step_continues_sequence(self):
        session = make_session(last_steps={"loss": 4})

        Metric("loss", 0.5).log(session)

        kwargs = session.metrics_api.append_experiment_metrics.call_args.kwargs
        assert kwargs["metrics"][0].values[0].step == 5
        assert session.last_steps["loss"] == 5

    def test_explicit_step_does_not_update_sequence(self):
        # Mirrors the background worker: only auto-assigned steps advance last_steps.
        session = make_session(last_steps={"loss": 4})

        Metric("loss", 0.5, step=100).log(session)

        assert session.last_steps["loss"] == 4

    def test_store_step_false_still_auto_steps(self):
        # store_step=False means "ignore user-provided steps"; the worker then
        # auto-assigns, and the synchronous path mirrors that.
        session = make_session(store_step=False)

        Metric("loss", 0.5, step=100).log(session)

        kwargs = session.metrics_api.append_experiment_metrics.call_args.kwargs
        assert kwargs["metrics"][0].values[0].step == 0
        assert session.last_steps["loss"] == 0


class TestMetricRestore:
    """Metric._restore_values pulls every series' values through the API layer."""

    def test_delegates_to_metrics_api(self):
        session = make_session()
        session.metrics_api.get_metric_values.return_value = {"loss": [1.0, 0.5]}

        result = Metric._restore_values(session)

        session.metrics_api.get_metric_values.assert_called_once_with("ts-1", "ms-1")
        assert result == {"loss": [1.0, 0.5]}


class TestMetadataLog:
    """Metadata.log read-modify-writes the full code-tag collection."""

    def _tag(self, name, value, from_code=True):
        tag = MagicMock()
        tag.name = name
        tag.value = value
        tag.from_code = from_code
        return tag

    def test_merges_into_freshly_read_tags(self):
        session = make_session()
        refreshed = MagicMock()
        refreshed.id = "ms-1"
        refreshed.name = "exp"
        refreshed.tags = [self._tag("lr", "0.001")]
        session.metrics_api.get_experiment_metrics_by_name.return_value = refreshed

        Metadata("batch_size", "32").log(session)

        session.metrics_api.get_experiment_metrics_by_name.assert_called_once_with("ts-1", name="exp")
        kwargs = session.metrics_api.update_experiment_metrics.call_args.kwargs
        assert kwargs["teamspace_id"] == "ts-1"
        assert kwargs["metrics_store_id"] == "ms-1"
        assert kwargs["phase"] == PhaseType.RUNNING
        assert kwargs["metadata"] == {"lr": "0.001", "batch_size": "32"}

    def test_non_code_tags_are_dropped(self):
        session = make_session()
        refreshed = MagicMock()
        refreshed.id = "ms-1"
        refreshed.name = "exp"
        refreshed.tags = [self._tag("lr", "0.001"), self._tag("ui-tag", "x", from_code=False)]
        session.metrics_api.get_experiment_metrics_by_name.return_value = refreshed

        Metadata("batch_size", "32").log(session)

        kwargs = session.metrics_api.update_experiment_metrics.call_args.kwargs
        assert kwargs["metadata"] == {"lr": "0.001", "batch_size": "32"}

    def test_overwrites_existing_key(self):
        session = make_session()
        refreshed = MagicMock()
        refreshed.id = "ms-1"
        refreshed.name = "exp"
        refreshed.tags = [self._tag("lr", "0.001")]
        session.metrics_api.get_experiment_metrics_by_name.return_value = refreshed

        Metadata("lr", "0.01").log(session)

        kwargs = session.metrics_api.update_experiment_metrics.call_args.kwargs
        assert kwargs["metadata"] == {"lr": "0.01"}


class TestMetadataEnqueue:
    """Metadata.enqueue defers the read-modify-write to the background worker."""

    def test_puts_queued_write(self):
        session = make_session()
        entry = Metadata("lr", "0.001")

        entry.enqueue(session)

        session.queue.put.assert_called_once()
        item = session.queue.put.call_args[0][0]
        assert isinstance(item, _QueuedWrite)
        assert item.primitive is entry

    def test_raises_before_put_on_background_failure(self):
        session = make_session()
        session.background.exception = RuntimeError("bg error")

        with pytest.raises(RuntimeError, match="bg error"):
            Metadata("lr", "0.001").enqueue(session)

        session.queue.put.assert_not_called()
