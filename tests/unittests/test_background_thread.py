import queue
from unittest.mock import Mock, patch

import pytest
from lightning_sdk.lightning_cloud.openapi.rest import ApiException

from litlogger.background import _BackgroundThread
from litlogger.primitives import MetricWrite, PrimitiveWrite
from litlogger.session import ExperimentSession
from litlogger.types import Metrics, MetricValue, PhaseType


def _make_manager(*, last_x=None, store_step=True, max_batch_size=1000, interval=100):
    metrics_api = Mock()
    metrics_api.client = Mock()
    teamspace = Mock(id="ts-1")
    store = Mock(id="ms-1", name="exp")
    background = Mock(exception=None)
    session = ExperimentSession._from_components(
        name="exp",
        metrics_api=metrics_api,
        media_api=Mock(),
        artifacts_api=Mock(),
        teamspace=teamspace,
        metrics_store=store,
        queue_=queue.Queue(),
        stats=Mock(),
        printer=Mock(),
        background=background,
        store_step=store_step,
        last_x=last_x,
    )
    manager = _BackgroundThread(
        session=session,
        rate_limiting_interval=interval,
        max_batch_size=max_batch_size,
    )
    session.background = manager
    return session, manager


class TestBackgroundThreadInit:
    def test_reads_owned_components_from_session(self):
        session, manager = _make_manager(max_batch_size=500, interval=2)

        assert manager.teamspace_id == "ts-1"
        assert manager.metrics_store_id == "ms-1"
        assert manager.metrics_api is session.metrics_api
        assert manager.metrics_queue is session.queue
        assert manager.rate_limiting_interval == 2
        assert manager.max_batch_size == 500
        assert manager.daemon is True


class TestMetricCommands:
    def test_auto_x_starts_at_zero_and_increments(self):
        session, manager = _make_manager(last_x={})
        session.queue.put(MetricWrite("loss", 0.5, None, None))
        session.queue.put(MetricWrite("loss", 0.4, None, None))

        manager.step()

        assert [value.x for value in manager.metrics["loss"].values] == [0, 1]
        assert session.last_x == {"loss": 1}

    def test_explicit_fractional_x_drives_next_auto_increment(self):
        session, manager = _make_manager(last_x={})
        session.queue.put(MetricWrite("loss", 0.5, 1.5, None))
        session.queue.put(MetricWrite("loss", 0.4, None, None))

        manager.step()

        assert [value.x for value in manager.metrics["loss"].values] == [1.5, 2.5]

    def test_store_step_false_keeps_x_for_sequence_but_omits_wire_value(self):
        session, manager = _make_manager(last_x={}, store_step=False)
        session.queue.put(MetricWrite("loss", 0.5, 10, None))
        session.queue.put(MetricWrite("loss", 0.4, None, None))

        manager.step()

        assert [value.x for value in manager.metrics["loss"].values] == [None, None]
        assert session.last_x == {"loss": 11}

    def test_max_batch_triggers_send(self):
        session, manager = _make_manager(max_batch_size=2)
        session.queue.put(MetricWrite("loss", 0.5, None, None))
        session.queue.put(MetricWrite("loss", 0.4, None, None))

        with patch.object(manager, "_send") as send:
            manager.step()

        send.assert_called_once_with()


class TestPrimitiveCommands:
    def test_operations_execute_in_arrival_order(self):
        session, manager = _make_manager()
        order = []
        session.queue.put(PrimitiveWrite(lambda _: order.append("first")))
        session.queue.put(PrimitiveWrite(lambda _: order.append("second")))

        manager.step()

        assert order == ["first", "second"]

    def test_metrics_merge_around_primitive_operations(self):
        session, manager = _make_manager()
        operation = Mock()
        session.queue.put(MetricWrite("loss", 1.0, None, None))
        session.queue.put(PrimitiveWrite(operation))
        session.queue.put(MetricWrite("loss", 2.0, None, None))

        manager.step()

        operation.assert_called_once_with(session)
        assert [value.value for value in manager.metrics["loss"].values] == [1.0, 2.0]

    def test_unknown_command_fails_explicitly(self):
        session, manager = _make_manager()
        session.queue.put(object())

        with pytest.raises(TypeError, match="Unsupported queue command"):
            manager.step()


class TestBackgroundFailure:
    def test_failed_write_closes_submission_and_drains_pending_commands(self):
        session, manager = _make_manager()
        never_run = Mock()
        session.queue.put(PrimitiveWrite(lambda _: (_ for _ in ()).throw(RuntimeError("upload failed"))))
        session.queue.put(PrimitiveWrite(never_run))

        manager._run()

        assert isinstance(manager.exception, RuntimeError)
        never_run.assert_not_called()
        session.queue.join()
        with pytest.raises(RuntimeError, match="upload failed"):
            session.submit(MetricWrite("loss", 1.0, None, None))


class TestSendMetrics:
    @pytest.mark.parametrize(("count", "calls"), [(500, 1), (1000, 1), (2500, 3)])
    def test_chunks_one_metric(self, count, calls):
        session, manager = _make_manager(max_batch_size=1000, interval=0)
        metrics = [Metrics(name="loss", values=[MetricValue(value=i) for i in range(count)])]

        manager._send_metrics(metrics)

        assert session.metrics_api.append_experiment_metrics.call_count == calls

    def test_chunks_across_metric_names(self):
        session, manager = _make_manager(max_batch_size=100, interval=0)
        metrics = [
            Metrics(name="loss", values=[MetricValue(value=i) for i in range(60)]),
            Metrics(name="accuracy", values=[MetricValue(value=i) for i in range(60)]),
        ]

        manager._send_metrics(metrics)

        assert session.metrics_api.append_experiment_metrics.call_count == 2

    def test_deleted_stream_has_clear_error(self):
        session, manager = _make_manager()
        session.metrics_api.append_experiment_metrics.side_effect = ApiException(status=404, reason="not found")
        manager.metrics = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}

        with pytest.raises(Exception, match="metrics stream has been deleted"):
            manager._send()

    def test_other_api_errors_propagate(self):
        session, manager = _make_manager()
        session.metrics_api.append_experiment_metrics.side_effect = ApiException(status=500, reason="failed")
        manager.metrics = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}

        with pytest.raises(ApiException):
            manager._send()


class TestLifecycle:
    def test_run_sets_done_event(self):
        session, manager = _make_manager()

        with patch.object(manager, "_run"):
            manager.run()

        assert session.done_event.is_set()

    def test_inform_done_updates_owned_stream(self):
        session, manager = _make_manager()

        manager.inform_done()

        session.metrics_api.update_experiment_metrics.assert_called_once_with(
            teamspace_id="ts-1",
            metrics_store_id="ms-1",
            persisted=True,
            phase=PhaseType.COMPLETED,
        )
