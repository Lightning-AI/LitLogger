# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for the shared ExperimentSession."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from litlogger.session import ExperimentSession


def _make_fake_experiment():
    """Build a plain object with the attributes from_experiment reads."""
    metrics_api = MagicMock()
    experiment = SimpleNamespace(
        name="exp",
        _metrics_api=metrics_api,
        _media_api=MagicMock(),
        _artifacts_api=MagicMock(),
        _teamspace=MagicMock(),
        _metrics_store=MagicMock(),
        _metrics_queue=MagicMock(),
        _stats=MagicMock(),
        store_step=True,
        store_created_at=False,
        _resumed_steps={"loss": 3},
        _manager=MagicMock(),
    )
    experiment._teamspace.id = "ts-1"
    experiment._metrics_store.id = "ms-1"
    experiment._metrics_store.name = "exp"
    experiment._manager.exception = None
    return experiment


class TestFromExperiment:
    """Test ExperimentSession.from_experiment wiring."""

    def test_shares_experiment_infrastructure(self):
        exp = _make_fake_experiment()

        session = ExperimentSession.from_experiment(exp)

        assert session.client is exp._metrics_api.client
        assert session.metrics_api is exp._metrics_api
        assert session.media_api is exp._media_api
        assert session.artifacts_api is exp._artifacts_api
        assert session.teamspace is exp._teamspace
        assert session.experiment is exp
        assert session.queue is exp._metrics_queue
        assert session.stats is exp._stats
        assert session.store_step is True
        assert session.store_created_at is False
        assert session.last_steps is exp._resumed_steps
        assert session.background is exp._manager

    def test_missing_manager_yields_no_background(self):
        exp = _make_fake_experiment()
        del exp._manager

        session = ExperimentSession.from_experiment(exp)

        assert session.background is None

    def test_identity_properties(self):
        exp = _make_fake_experiment()

        session = ExperimentSession.from_experiment(exp)

        assert session.experiment_name == "exp"
        assert session.teamspace_id == "ts-1"
        assert session.metrics_store_id == "ms-1"
        assert session.metrics_store is exp._metrics_store


class TestRefreshMetricsStore:
    """Test refresh_metrics_store keep-if-None semantics."""

    def test_updates_store_on_response(self):
        exp = _make_fake_experiment()
        session = ExperimentSession.from_experiment(exp)
        new_store = MagicMock()
        exp._metrics_api.get_experiment_metrics_by_name.return_value = new_store

        session.refresh_metrics_store()

        exp._metrics_api.get_experiment_metrics_by_name.assert_called_once_with("ts-1", name="exp")
        assert exp._metrics_store is new_store
        assert session.metrics_store is new_store

    def test_keeps_store_when_response_is_none(self):
        exp = _make_fake_experiment()
        old_store = exp._metrics_store
        session = ExperimentSession.from_experiment(exp)
        exp._metrics_api.get_experiment_metrics_by_name.return_value = None

        session.refresh_metrics_store()

        assert exp._metrics_store is old_store


class TestBackgroundFailure:
    """Test background failure propagation and flush."""

    def test_no_background_is_noop(self):
        exp = _make_fake_experiment()
        del exp._manager
        session = ExperimentSession.from_experiment(exp)

        session.raise_if_background_failed()

    def test_no_exception_is_noop(self):
        exp = _make_fake_experiment()
        session = ExperimentSession.from_experiment(exp)

        session.raise_if_background_failed()

    def test_raises_background_exception(self):
        exp = _make_fake_experiment()
        exp._manager.exception = RuntimeError("bg error")
        session = ExperimentSession.from_experiment(exp)

        with pytest.raises(RuntimeError, match="bg error"):
            session.raise_if_background_failed()

    def test_flush_joins_queue_then_checks_failure(self):
        exp = _make_fake_experiment()
        exp._manager.exception = RuntimeError("late failure")
        session = ExperimentSession.from_experiment(exp)

        with pytest.raises(RuntimeError, match="late failure"):
            session.flush()

        exp._metrics_queue.join.assert_called_once_with()
