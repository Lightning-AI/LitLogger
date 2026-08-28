import queue
from multiprocessing import Event, Queue
from queue import Queue as ThreadQueue
from unittest.mock import MagicMock, Mock, patch

import pytest
from lightning_sdk.lightning_cloud.openapi.rest import ApiException

from litlogger.background import _BackgroundThread
from litlogger.primitives import _QueuedWrite
from litlogger.types import Metrics, MetricValue, PhaseType


class TestBackgroundThreadInit:
    """Test _BackgroundThread initialization."""

    def test_init_with_all_parameters(self):
        """Test initialization with all required parameters."""
        mock_metrics_api = Mock()
        mock_queue = Queue()
        is_ready_event = Event()
        stop_event = Event()
        done_event = Event()

        manager = _BackgroundThread(
            teamspace_id="test_teamspace",
            metrics_store_id="test_stream",
            metrics_api=mock_metrics_api,
            metrics_queue=mock_queue,
            is_ready_event=is_ready_event,
            stop_event=stop_event,
            done_event=done_event,
            store_step=True,
            store_created_at=False,
            rate_limiting_interval=2,
            max_batch_size=500,
        )

        assert manager.teamspace_id == "test_teamspace"
        assert manager.metrics_store_id == "test_stream"
        assert manager.metrics_api == mock_metrics_api
        assert manager.metrics_queue == mock_queue
        assert manager.rate_limiting_interval == 2
        assert manager.max_batch_size == 500
        assert manager.is_ready_event == is_ready_event
        assert manager.stop_event == stop_event
        assert manager.done_event == done_event
        assert manager.store_step is True
        assert manager.store_created_at is False
        assert manager.metrics == {}
        assert manager.exception is None
        assert manager.daemon is True

    def test_init_with_default_values(self):
        """Test initialization with default rate_limiting_interval and max_batch_size."""
        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=Queue(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        assert manager.rate_limiting_interval == 1
        assert manager.max_batch_size == 1000


class TestBackgroundThreadLastSteps:
    """Test _BackgroundThread step resumption from last_steps."""

    def _make_manager(self, last_steps=None, session=None):
        mock_queue = Mock()
        mock_queue.get.side_effect = [
            {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])},
            {"loss": Metrics(name="loss", values=[MetricValue(value=0.4)])},
            {"loss": Metrics(name="loss", values=[MetricValue(value=0.3)])},
            queue.Empty(),
        ]
        return _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=mock_queue,
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
            rate_limiting_interval=100,
            last_steps=last_steps,
            session=session,
        )

    def test_steps_start_at_zero_when_last_steps_is_none(self):
        manager = self._make_manager(last_steps=None)
        manager.step()
        steps = [v.step for v in manager.metrics["loss"].values]
        assert steps == [0, 1, 2]

    def test_steps_start_at_zero_when_last_steps_is_empty(self):
        last_steps = {}
        manager = self._make_manager(last_steps=last_steps)
        manager.step()
        steps = [v.step for v in manager.metrics["loss"].values]
        assert steps == [0, 1, 2]
        assert manager.last_steps is last_steps
        assert last_steps == {"loss": 2}

    def test_steps_resume_from_last_steps_when_provided(self):
        manager = self._make_manager(last_steps={"loss": 5})
        manager.step()
        steps = [v.step for v in manager.metrics["loss"].values]
        assert steps == [6, 7, 8]

    def test_explicit_fractional_x_drives_next_auto_increment(self):
        mock_queue = Mock()
        mock_queue.get.side_effect = [
            {"loss": Metrics(name="loss", values=[MetricValue(value=0.5, step=1.5)])},
            {"loss": Metrics(name="loss", values=[MetricValue(value=0.4)])},
            queue.Empty(),
        ]
        manager = self._make_manager(last_steps={})
        manager.metrics_queue = mock_queue

        manager.step()

        steps = [v.step for v in manager.metrics["loss"].values]
        assert steps == [1.5, 2.5]
        assert manager.last_steps["loss"] == 2.5

    def test_auto_steps_use_the_session_lock(self):
        session = Mock()
        session.last_steps_lock = MagicMock()
        manager = self._make_manager(last_steps={}, session=session)

        manager.step()

        assert session.last_steps_lock.__enter__.call_count == 3
        assert session.last_steps_lock.__exit__.call_count == 3


class TestBackgroundThreadStepBatching:
    """Test _BackgroundThread.step batching behavior."""

    def test_step_reads_from_queue_success(self):
        """Test successfully reading metrics from the queue."""
        mock_queue = Mock()
        metrics_data = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}
        # First call returns data, second call raises Empty to exit the loop
        mock_queue.get.side_effect = [metrics_data, queue.Empty()]

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=mock_queue,
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
            rate_limiting_interval=100,  # High value to prevent time-based send
        )

        result = manager.step()

        assert result is True
        assert "loss" in manager.metrics

    def test_step_from_empty_queue(self):
        """Test reading from empty queue returns False."""
        mock_queue = Mock()
        mock_queue.get.side_effect = queue.Empty()

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=mock_queue,
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        result = manager.step()

        assert result is False
        assert len(manager.metrics) == 0

    def test_step_sends_when_max_batch_reached(self):
        """Test that step sends when max_batch_size is reached."""
        mock_queue = Mock()
        # Create enough values to trigger send
        metrics_data = {"loss": Metrics(name="loss", values=[MetricValue(value=i) for i in range(100)])}
        mock_queue.get.side_effect = [metrics_data, queue.Empty()]

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=mock_queue,
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
            max_batch_size=50,  # Set low threshold
        )

        with patch.object(manager, "_send") as mock_send:
            manager.step()

            mock_send.assert_called_once()

    def test_step_merges_metrics(self):
        """Test that step correctly merges metrics from multiple queue items."""
        mock_queue = Mock()
        metrics_data1 = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}
        metrics_data2 = {"loss": Metrics(name="loss", values=[MetricValue(value=0.3)])}
        mock_queue.get.side_effect = [metrics_data1, metrics_data2, queue.Empty()]

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=mock_queue,
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
            rate_limiting_interval=100,  # High value to prevent time-based send
        )

        manager.step()

        # Both values should be merged into the same metric
        assert "loss" in manager.metrics
        assert len(manager.metrics["loss"].values) == 2
        assert manager.metrics["loss"].values[0].value == 0.5
        assert manager.metrics["loss"].values[1].value == 0.3


class TestBackgroundThreadSend:
    """Test _BackgroundThread._send method."""

    def test_send_empty_metrics(self):
        """Test _send does nothing when metrics are empty."""
        mock_metrics_api = Mock()

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        manager.metrics = {}
        manager._send()

        mock_metrics_api.append_experiment_metrics.assert_not_called()

    def test_send_metrics_success(self):
        """Test successfully sending metrics."""
        mock_metrics_api = Mock()

        manager = _BackgroundThread(
            teamspace_id="test_teamspace",
            metrics_store_id="test_stream",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        manager.metrics = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}

        manager._send()

        assert mock_metrics_api.append_experiment_metrics.called
        assert manager.metrics == {}

    def test_send_raises_api_exception_for_deleted_stream(self):
        """Test _send raises exception when stream is deleted."""
        mock_metrics_api = Mock()
        mock_metrics_api.append_experiment_metrics.side_effect = ApiException(status=404, reason="not found")

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        manager.metrics = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}

        with pytest.raises(Exception, match="The metrics stream has been deleted"):
            manager._send()

    def test_send_raises_other_api_exceptions(self):
        """Test _send raises other API exceptions."""
        mock_metrics_api = Mock()
        mock_metrics_api.append_experiment_metrics.side_effect = ApiException(
            status=500, reason="Internal server error"
        )

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        manager.metrics = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}

        with pytest.raises(ApiException):
            manager._send()


class TestBackgroundThreadSendMetrics:
    """Test _BackgroundThread._send_metrics method."""

    def test_send_metrics_within_limit(self):
        """Test sending metrics within the max_batch_size limit."""
        mock_metrics_api = Mock()

        manager = _BackgroundThread(
            teamspace_id="test_teamspace",
            metrics_store_id="test_stream",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        metrics = [Metrics(name="loss", values=[MetricValue(value=i) for i in range(500)])]

        manager._send_metrics(metrics)

        mock_metrics_api.append_experiment_metrics.assert_called_once()

    def test_send_metrics_at_exact_limit(self):
        """Test sending exactly max_batch_size values."""
        mock_metrics_api = Mock()

        manager = _BackgroundThread(
            teamspace_id="test_teamspace",
            metrics_store_id="test_stream",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        metrics = [Metrics(name="loss", values=[MetricValue(value=i) for i in range(1000)])]

        manager._send_metrics(metrics)

        mock_metrics_api.append_experiment_metrics.assert_called_once()

    def test_send_metrics_exceeds_limit_chunks_automatically(self):
        """Test sending more than max_batch_size values chunks into multiple requests."""
        mock_metrics_api = Mock()

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
            max_batch_size=1000,
        )

        # Send 2500 values - should result in 3 API calls (1000 + 1000 + 500)
        metrics = [Metrics(name="loss", values=[MetricValue(value=i) for i in range(2500)])]

        manager._send_metrics(metrics)

        assert mock_metrics_api.append_experiment_metrics.call_count == 3

    def test_send_metrics_chunks_multiple_metrics(self):
        """Test chunking works correctly with multiple metrics."""
        mock_metrics_api = Mock()

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
            max_batch_size=100,
        )

        # Two metrics with 60 values each = 120 values, should split into 2 requests
        metrics = [
            Metrics(name="loss", values=[MetricValue(value=i) for i in range(60)]),
            Metrics(name="accuracy", values=[MetricValue(value=i) for i in range(60)]),
        ]

        manager._send_metrics(metrics)

        # Should chunk: first 100 values, then remaining 20
        assert mock_metrics_api.append_experiment_metrics.call_count == 2

    def test_send_metrics_with_custom_batch_size(self):
        """Test chunking respects custom max_batch_size."""
        mock_metrics_api = Mock()

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
            max_batch_size=50,
        )

        # Send 120 values with max_batch_size=50 - should result in 3 API calls
        metrics = [Metrics(name="loss", values=[MetricValue(value=i) for i in range(120)])]

        manager._send_metrics(metrics)

        assert mock_metrics_api.append_experiment_metrics.call_count == 3


class TestBackgroundThreadStep:
    """Test _BackgroundThread.step method return values."""

    def test_step_returns_true_with_metrics(self):
        """Test step returns True when metrics are available in queue."""
        mock_queue = Mock()
        metrics_data = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}
        mock_queue.get.side_effect = [metrics_data, queue.Empty()]

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=mock_queue,
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
            rate_limiting_interval=100,  # Prevent time-based send
        )

        result = manager.step()

        assert result is True

    def test_step_returns_false_with_empty_queue(self):
        """Test step returns False when queue is empty."""
        mock_queue = Mock()
        mock_queue.get.side_effect = queue.Empty()

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=mock_queue,
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        result = manager.step()

        assert result is False


class TestBackgroundThreadInformDone:
    """Test _BackgroundThread.inform_done method."""

    def test_inform_done(self):
        """Test inform_done updates stream with COMPLETED phase."""
        mock_metrics_api = Mock()

        manager = _BackgroundThread(
            teamspace_id="test_teamspace",
            metrics_store_id="test_stream",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        manager.inform_done()

        mock_metrics_api.update_experiment_metrics.assert_called_once()
        call_args = mock_metrics_api.update_experiment_metrics.call_args
        assert call_args.kwargs["teamspace_id"] == "test_teamspace"
        assert call_args.kwargs["metrics_store_id"] == "test_stream"
        assert call_args.kwargs["persisted"] is True
        assert call_args.kwargs["phase"] == PhaseType.COMPLETED


class TestBackgroundThreadRun:
    """Test _BackgroundThread.run and _run methods."""

    def test_run_sets_done_event(self):
        """Test that run method sets done_event after _run completes."""
        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        with patch.object(manager, "_run"):
            manager.run()

            assert manager.done_event.is_set()

    def test_run_lifecycle_success(self):
        """Test successful _run lifecycle."""
        mock_metrics_api = Mock()
        stop_event = Event()

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=mock_metrics_api,
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=stop_event,
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        # Mock step to return False (empty queue) and set stop event
        call_count = [0]

        def mock_step():
            call_count[0] += 1
            if call_count[0] == 1:
                stop_event.set()
            return False

        with patch.object(manager, "step", side_effect=mock_step):
            manager._run()

            assert manager.is_ready_event.is_set()
            assert manager.done_event.is_set()
            mock_metrics_api.update_experiment_metrics.assert_called_once()

    def test_run_handles_exception(self):
        """Test _run handles exceptions and sets exception attribute."""
        stop_event = Event()

        manager = _BackgroundThread(
            teamspace_id="test",
            metrics_store_id="test",
            metrics_api=Mock(),
            metrics_queue=Mock(),
            is_ready_event=Event(),
            stop_event=stop_event,
            done_event=Event(),
            store_step=True,
            store_created_at=False,
        )

        test_exception = Exception("Test error")

        with patch.object(manager, "step", side_effect=test_exception):
            manager._run()

            assert manager.exception == test_exception
            assert manager.done_event.is_set()


class TestBackgroundThreadIntegration:
    """Integration tests for _BackgroundThread."""

    def test_full_workflow_with_metrics(self):
        """Test complete workflow from queue to send."""
        mock_metrics_api = Mock()
        metrics_queue = Queue()
        stop_event = Event()
        is_ready_event = Event()
        done_event = Event()

        # Add metrics to queue
        metrics_data = {
            "loss": Metrics(
                name="loss",
                values=[MetricValue(value=0.5, step=0), MetricValue(value=0.3, step=1)],
            )
        }
        metrics_queue.put(metrics_data)

        manager = _BackgroundThread(
            teamspace_id="test_teamspace",
            metrics_store_id="test_stream",
            metrics_api=mock_metrics_api,
            metrics_queue=metrics_queue,
            is_ready_event=is_ready_event,
            stop_event=stop_event,
            done_event=done_event,
            store_step=True,
            store_created_at=False,
        )

        # Simulate running until queue is empty, then stop
        call_count = [0]
        original_step = manager.step

        def mock_step():
            result = original_step()
            call_count[0] += 1
            if call_count[0] >= 2:
                stop_event.set()
            return result

        with patch.object(manager, "step", side_effect=mock_step):
            manager._run()

        # Verify the workflow
        assert is_ready_event.is_set()
        assert done_event.is_set()
        assert manager.exception is None
        assert mock_metrics_api.update_experiment_metrics.called


class TestQueuedWrites:
    """The worker executes queued non-metric primitives via the session."""

    @staticmethod
    def _make_manager(session=None):
        return _BackgroundThread(
            teamspace_id="ts-1",
            metrics_store_id="ms-1",
            metrics_api=Mock(),
            metrics_queue=ThreadQueue(),
            is_ready_event=Event(),
            stop_event=Event(),
            done_event=Event(),
            store_step=True,
            store_created_at=False,
            session=session,
        )

    def test_executes_queued_primitive(self):
        session = Mock()
        manager = self._make_manager(session=session)
        primitive = Mock()
        manager.metrics_queue.put(_QueuedWrite(primitive))

        manager.step()

        primitive.log.assert_called_once_with(session)

    def test_queued_writes_execute_in_arrival_order(self):
        session = Mock()
        manager = self._make_manager(session=session)
        order = []
        first, second = Mock(), Mock()
        first.log.side_effect = lambda s: order.append("first")
        second.log.side_effect = lambda s: order.append("second")
        manager.metrics_queue.put(_QueuedWrite(first))
        manager.metrics_queue.put(_QueuedWrite(second))

        manager.step()

        assert order == ["first", "second"]

    def test_metric_batches_still_merge_around_queued_writes(self):
        session = Mock()
        manager = self._make_manager(session=session)
        primitive = Mock()
        manager.metrics_queue.put({"loss": Metrics(name="loss", values=[MetricValue(value=1.0)])})
        manager.metrics_queue.put(_QueuedWrite(primitive))
        manager.metrics_queue.put({"loss": Metrics(name="loss", values=[MetricValue(value=2.0)])})

        with patch.object(manager, "_send"):
            manager.step()

        primitive.log.assert_called_once_with(session)
        assert [v.value for v in manager.metrics["loss"].values] == [1.0, 2.0]

    def test_queued_write_without_session_fails(self):
        manager = self._make_manager(session=None)
        manager.metrics_queue.put(_QueuedWrite(Mock()))

        with pytest.raises(RuntimeError, match="session-aware"):
            manager.step()

    def test_failed_queued_write_poisons_worker_and_unblocks_join(self):
        session = Mock()
        manager = self._make_manager(session=session)
        failing, never_run = Mock(), Mock()
        failing.log.side_effect = RuntimeError("upload failed")
        manager.metrics_queue.put(_QueuedWrite(failing))
        manager.metrics_queue.put(_QueuedWrite(never_run))

        manager._run()

        assert isinstance(manager.exception, RuntimeError)
        assert manager.done_event.is_set()
        never_run.log.assert_not_called()
        # join() must not deadlock on the abandoned item
        manager.metrics_queue.join()

    def test_failure_is_visible_before_done_is_signaled(self):
        session = Mock()
        done_event = Mock()
        manager = self._make_manager(session=session)
        manager.done_event = done_event
        failing = Mock()
        failing.log.side_effect = RuntimeError("upload failed")
        manager.metrics_queue.put(_QueuedWrite(failing))
        observed_exceptions = []
        done_event.set.side_effect = lambda: observed_exceptions.append(manager.exception)

        manager._run()

        assert isinstance(observed_exceptions[0], RuntimeError)
