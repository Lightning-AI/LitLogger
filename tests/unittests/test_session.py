# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");

import queue
from unittest.mock import Mock

import pytest

from litlogger.primitives import MetricWrite
from litlogger.session import ExperimentSession


def _make_session(*, queue_=None, last_x=None):
    metrics_api = Mock()
    metrics_api.client = Mock()
    teamspace = Mock(id="ts-1")
    store = Mock(id="ms-1", name="exp")
    background = Mock(exception=None)
    return ExperimentSession._from_components(
        name="exp",
        metrics_api=metrics_api,
        media_api=Mock(),
        artifacts_api=Mock(),
        teamspace=teamspace,
        metrics_store=store,
        queue_=queue_ or queue.Queue(),
        stats=Mock(),
        printer=Mock(),
        background=background,
        last_x=last_x,
    )


class TestOwnedInfrastructure:
    def test_identity_properties_come_from_session_state(self):
        session = _make_session(last_x={"loss": 3})

        assert session.client is session.metrics_api.client
        assert session.experiment_name == "exp"
        assert session.teamspace_id == "ts-1"
        assert session.metrics_store_id == "ms-1"
        assert session.last_steps is session.last_x

    def test_refresh_replaces_session_store(self):
        session = _make_session()
        refreshed = Mock(id="ms-2", name="exp")
        session.metrics_api.get_experiment_metrics_by_name.return_value = refreshed

        session.refresh_metrics_store()

        assert session.metrics_store is refreshed

    def test_refresh_keeps_store_on_miss(self):
        session = _make_session()
        original = session.metrics_store
        session.metrics_api.get_experiment_metrics_by_name.return_value = None

        session.refresh_metrics_store()

        assert session.metrics_store is original


class TestCoordinates:
    def test_explicit_x_updates_auto_increment_source_even_when_not_persisted(self):
        session = _make_session(last_x={})
        session.store_step = False

        assert session.resolve_x("loss", 10) == 10
        assert session.resolve_x("loss", None) == 11


class TestSubmissionLifecycle:
    def test_failure_drains_pending_commands_and_rejects_new_ones(self):
        session = _make_session()
        session.queue.put(MetricWrite("loss", 1.0, None, None))

        session._record_background_failure(RuntimeError("worker failed"))

        session.queue.join()
        with pytest.raises(RuntimeError, match="worker failed"):
            session.submit(MetricWrite("loss", 2.0, None, None))

    def test_closed_session_rejects_submission(self):
        session = _make_session()
        session._accepting = False

        with pytest.raises(RuntimeError, match="no longer accepting"):
            session.submit(MetricWrite("loss", 1.0, None, None))

    def test_submission_uses_one_session_owned_queue(self):
        session = _make_session()
        command = MetricWrite("loss", 1.0, 3, None)

        session.submit(command)

        assert session.queue.get_nowait() is command
