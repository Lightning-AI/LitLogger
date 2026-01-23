import queue
from collections import defaultdict
from datetime import datetime
from multiprocessing import Event, Queue
from unittest.mock import Mock, patch

import pytest
from lightning_sdk.lightning_cloud.openapi.rest import ApiException
from litlogger.background import _BackgroundThread
from litlogger.types import Metrics, MetricsTracker, MetricValue, PhaseType


class TestBackgroundThreadInit:
    """Test _BackgroundThread initialization."""

    def test_init_with_trackers_init(self):
        """Test initialization with existing trackers."""
        mock_metrics_api = Mock()
        mock_queue = Queue()
        is_ready_event = Event()
        stop_event = Event()
        done_event = Event()

        initial_trackers = {
            "loss": MetricsTracker(name="loss", num_rows=100, min_value=0.1, max_value=1.0),
            "accuracy": MetricsTracker(name="accuracy", num_rows=50, min_value=0.8, max_value=0.99),
        }

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test_teamspace",
                metrics_store_id="test_stream",
                cloud_account="test_account",
                metrics_api=mock_metrics_api,
                metrics_queue=mock_queue,
                is_ready_event=is_ready_event,
                stop_event=stop_event,
                done_event=done_event,
                log_dir="/test/logs",
                version="v1.0.0",
                store_step=True,
                store_created_at=False,
                trackers_init=initial_trackers,
            )

            # Verify trackers were initialized from trackers_init
            assert "loss" in manager.trackers
            assert "accuracy" in manager.trackers
            assert manager.trackers["loss"].num_rows == 100
            assert manager.trackers["accuracy"].num_rows == 50

    def test_init_with_trackers_init_none(self):
        """Test initialization with trackers_init=None creates empty trackers."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Queue(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
                trackers_init=None,
            )

            assert manager.trackers == {}

    def test_init_with_all_parameters(self):
        """Test initialization with all required parameters."""
        mock_metrics_api = Mock()
        mock_queue = Queue()
        is_ready_event = Event()
        stop_event = Event()
        done_event = Event()

        with patch("litlogger.background.BinaryFileWriter") as mock_file_writer:
            manager = _BackgroundThread(
                teamspace_id="test_teamspace",
                metrics_store_id="test_stream",
                cloud_account="test_account",
                metrics_api=mock_metrics_api,
                metrics_queue=mock_queue,
                is_ready_event=is_ready_event,
                stop_event=stop_event,
                done_event=done_event,
                log_dir="/test/logs",
                version="v1.0.0",
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
            assert manager.trackers == {}
            assert manager.exception is None
            assert manager.daemon is True

            # Verify BinaryFileWriter was initialized correctly
            mock_file_writer.assert_called_once_with(
                log_dir="/test/logs",
                version="v1.0.0",
                store_step=True,
                store_created_at=False,
                teamspace_id="test_teamspace",
                metrics_store_id="test_stream",
                cloud_account="test_account",
                client=mock_metrics_api.client,
            )

    def test_init_with_default_values(self):
        """Test initialization with default rate_limiting_interval and max_batch_size."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Queue(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            assert manager.rate_limiting_interval == 1
            assert manager.max_batch_size == 1000


class TestBackgroundThreadStepBatching:
    """Test _BackgroundThread.step batching behavior."""

    def test_step_reads_from_queue_success(self):
        """Test successfully reading metrics from the queue."""
        mock_queue = Mock()
        metrics_data = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}
        # First call returns data, second call raises Empty to exit the loop
        mock_queue.get.side_effect = [metrics_data, queue.Empty()]

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=mock_queue,
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
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

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=mock_queue,
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            result = manager.step()

            assert result is False
            assert len(manager.metrics) == 0

    def test_step_updates_tracker(self):
        """Test that step calls _update_tracker for each metric."""
        mock_queue = Mock()
        metrics_data = {
            "loss": Metrics(name="loss", values=[MetricValue(value=0.5)]),
            "acc": Metrics(name="acc", values=[MetricValue(value=0.9)]),
        }
        mock_queue.get.side_effect = [metrics_data, queue.Empty()]

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=mock_queue,
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
                rate_limiting_interval=100,  # High value to prevent time-based send
            )

            with patch.object(manager, "_update_tracker") as mock_update:
                manager.step()

                assert mock_update.call_count == 2
                mock_update.assert_any_call("loss", metrics_data["loss"])
                mock_update.assert_any_call("acc", metrics_data["acc"])

    def test_step_sends_when_max_batch_reached(self):
        """Test that step sends when max_batch_size is reached."""
        mock_queue = Mock()
        # Create enough values to trigger send
        metrics_data = {"loss": Metrics(name="loss", values=[MetricValue(value=i) for i in range(100)])}
        mock_queue.get.side_effect = [metrics_data, queue.Empty()]

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=mock_queue,
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
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

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=mock_queue,
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
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


class TestBackgroundThreadStepAugmentation:
    """Test step augmentation from trackers when metrics don't have steps."""

    def test_augments_step_when_none(self):
        """Test that metrics without steps get augmented with tracker num_rows."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            # Create values without steps
            values = Metrics(
                name="loss",
                values=[
                    MetricValue(value=0.5, step=None),
                    MetricValue(value=0.4, step=None),
                    MetricValue(value=0.3, step=None),
                ],
            )

            manager._update_tracker("loss", values)

            # Verify steps were augmented sequentially
            assert values.values[0].step == 0
            assert values.values[1].step == 1
            assert values.values[2].step == 2

    def test_preserves_explicit_steps(self):
        """Test that metrics with explicit steps are not modified."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            # Create values with explicit steps
            values = Metrics(
                name="loss",
                values=[
                    MetricValue(value=0.5, step=10),
                    MetricValue(value=0.4, step=20),
                    MetricValue(value=0.3, step=30),
                ],
            )

            manager._update_tracker("loss", values)

            # Verify explicit steps were preserved
            assert values.values[0].step == 10
            assert values.values[1].step == 20
            assert values.values[2].step == 30

    def test_augments_step_from_initialized_tracker(self):
        """Test that step augmentation continues from initialized tracker num_rows."""
        initial_trackers = {
            "loss": MetricsTracker(name="loss", num_rows=100),
        }

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
                trackers_init=initial_trackers,
            )

            # Create values without steps
            values = Metrics(
                name="loss",
                values=[
                    MetricValue(value=0.5, step=None),
                    MetricValue(value=0.4, step=None),
                ],
            )

            manager._update_tracker("loss", values)

            # Verify steps continue from tracker's num_rows (100)
            assert values.values[0].step == 100
            assert values.values[1].step == 101

    def test_mixed_explicit_and_none_steps(self):
        """Test handling of mixed explicit and None steps."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            # Create values with mixed steps
            values = Metrics(
                name="loss",
                values=[
                    MetricValue(value=0.5, step=None),  # Should get 0
                    MetricValue(value=0.4, step=50),  # Should stay 50
                    MetricValue(value=0.3, step=None),  # Should get 2
                ],
            )

            manager._update_tracker("loss", values)

            # Verify correct step handling
            assert values.values[0].step == 0
            assert values.values[1].step == 50
            assert values.values[2].step == 2


class TestBackgroundThreadUpdateTracker:
    """Test _BackgroundThread._update_tracker method."""

    def test_create_new_tracker(self):
        """Test creating a new tracker for a metric."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            values = Mock()
            values.values = [MetricValue(value=1.0, step=0)]

            manager._update_tracker("new_metric", values)

            assert "new_metric" in manager.trackers
            assert manager.trackers["new_metric"].name == "new_metric"
            assert manager.trackers["new_metric"].num_rows == 1

    def test_update_existing_tracker(self):
        """Test updating an existing tracker."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            # Create initial tracker
            values1 = Mock()
            values1.values = [MetricValue(value=1.0, step=0)]
            manager._update_tracker("metric", values1)

            # Update with new values
            values2 = Mock()
            values2.values = [MetricValue(value=2.0, step=1)]
            manager._update_tracker("metric", values2)

            assert manager.trackers["metric"].num_rows == 2

    def test_tracker_min_max_values(self):
        """Test that tracker correctly tracks min and max values."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            values = Mock()
            values.values = [
                MetricValue(value=5.0, step=0),
                MetricValue(value=2.0, step=1),
                MetricValue(value=8.0, step=2),
                MetricValue(value=3.0, step=3),
            ]

            manager._update_tracker("metric", values)

            tracker = manager.trackers["metric"]
            assert tracker.min_value == 2.0
            assert tracker.min_index == 1
            assert tracker.max_value == 8.0
            assert tracker.max_index == 2
            assert tracker.last_value == 3.0
            assert tracker.last_index == 3

    def test_tracker_internal_start_step_assignment(self):
        """Test internal_start_step is correctly assigned across batches."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="test",
                version="test",
                store_step=True,
                store_created_at=False,
            )

            # Create test values for two metrics with multiple batches
            metric_names = ("metric1", "metric2")
            values = defaultdict(list)

            for _ in range(2):
                for metric_name in metric_names:
                    for _ in range(2):
                        batch = Metrics(
                            name=metric_name, values=[MetricValue(value=float(i), step=i) for i in range(3)]
                        )
                        values[metric_name].append(batch)

            # Update trackers sequentially
            for metric_name in metric_names:
                for batch in values[metric_name]:
                    manager._update_tracker(metric_name, batch)

            # Verify internal steps are correctly assigned
            for metric_name in metric_names:
                assert values[metric_name][0].internal_start_step == 0
                assert values[metric_name][1].internal_start_step == 3
                assert values[metric_name][2].internal_start_step == 6
                assert values[metric_name][3].internal_start_step == 9
                assert manager.trackers[metric_name].num_rows == 12

    def test_tracker_with_created_at_timestamps(self):
        """Test tracker handles created_at timestamps when store_created_at is True."""
        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=True,
            )

            timestamp1 = "2024-03-14T12:00:00.000000+00:00"
            timestamp2 = "2024-03-14T12:01:00.000000+00:00"

            values = Mock()
            values.values = [
                MetricValue(value=1.0, step=0, created_at=timestamp1),
                MetricValue(value=2.0, step=1, created_at=timestamp2),
            ]

            manager._update_tracker("metric", values)

            tracker = manager.trackers["metric"]
            assert tracker.started_at is not None
            assert tracker.updated_at is not None


class TestBackgroundThreadSend:
    """Test _BackgroundThread._send method."""

    def test_send_empty_metrics(self):
        """Test _send does nothing when metrics are empty."""
        mock_metrics_api = Mock()
        mock_file_store = Mock()

        with patch("litlogger.background.BinaryFileWriter", return_value=mock_file_store):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            manager.metrics = {}
            manager._send()

            mock_file_store.store.assert_not_called()
            mock_metrics_api.append_experiment_metrics.assert_not_called()

    def test_send_metrics_success(self):
        """Test successfully sending metrics."""
        mock_metrics_api = Mock()
        mock_file_store = Mock()

        with patch("litlogger.background.BinaryFileWriter", return_value=mock_file_store):
            manager = _BackgroundThread(
                teamspace_id="test_teamspace",
                metrics_store_id="test_stream",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            metrics_data = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}
            manager.metrics = metrics_data
            manager.trackers = {"loss": MetricsTracker(name="loss", num_rows=1)}

            manager._send()

            # Verify file store was called
            mock_file_store.store.assert_called_once()

            # Verify API was called
            assert mock_metrics_api.append_experiment_metrics.called
            assert manager.metrics == {}

    def test_send_raises_api_exception_for_deleted_stream(self):
        """Test _send raises exception when stream is deleted."""
        mock_metrics_api = Mock()
        mock_metrics_api.append_experiment_metrics.side_effect = ApiException(status=404, reason="not found")
        mock_file_store = Mock()

        with patch("litlogger.background.BinaryFileWriter", return_value=mock_file_store):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            manager.metrics = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}
            manager.trackers = {"loss": MetricsTracker(name="loss", num_rows=1)}

            with pytest.raises(Exception, match="The metrics stream has been deleted"):
                manager._send()

    def test_send_raises_other_api_exceptions(self):
        """Test _send raises other API exceptions."""
        mock_metrics_api = Mock()
        mock_metrics_api.append_experiment_metrics.side_effect = ApiException(
            status=500, reason="Internal server error"
        )
        mock_file_store = Mock()

        with patch("litlogger.background.BinaryFileWriter", return_value=mock_file_store):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            manager.metrics = {"loss": Metrics(name="loss", values=[MetricValue(value=0.5)])}
            manager.trackers = {"loss": MetricsTracker(name="loss", num_rows=1)}

            with pytest.raises(ApiException):
                manager._send()


class TestBackgroundThreadSendMetrics:
    """Test _BackgroundThread._send_metrics method."""

    def test_send_metrics_within_limit(self):
        """Test sending metrics within the max_batch_size limit."""
        mock_metrics_api = Mock()

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test_teamspace",
                metrics_store_id="test_stream",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            metrics = [Metrics(name="loss", values=[MetricValue(value=i) for i in range(500)])]

            manager._send_metrics(metrics)

            mock_metrics_api.append_experiment_metrics.assert_called_once()

    def test_send_metrics_at_exact_limit(self):
        """Test sending exactly max_batch_size values."""
        mock_metrics_api = Mock()

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test_teamspace",
                metrics_store_id="test_stream",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            metrics = [Metrics(name="loss", values=[MetricValue(value=i) for i in range(1000)])]

            manager._send_metrics(metrics)

            mock_metrics_api.append_experiment_metrics.assert_called_once()

    def test_send_metrics_exceeds_limit_chunks_automatically(self):
        """Test sending more than max_batch_size values chunks into multiple requests."""
        mock_metrics_api = Mock()

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
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

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
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

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
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

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=mock_queue,
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
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

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=mock_queue,
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            result = manager.step()

            assert result is False


class TestBackgroundThreadInformDone:
    """Test _BackgroundThread.inform_done method."""

    def test_inform_done_without_timestamps(self):
        """Test inform_done updates stream without timestamp conversion."""
        mock_metrics_api = Mock()

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test_teamspace",
                metrics_store_id="test_stream",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            manager.trackers = {
                "loss": MetricsTracker(
                    name="loss",
                    num_rows=10,
                    min_value=0.1,
                    max_value=1.0,
                )
            }

            manager.inform_done()

            mock_metrics_api.update_experiment_metrics.assert_called_once()
            call_args = mock_metrics_api.update_experiment_metrics.call_args
            assert call_args.kwargs["teamspace_id"] == "test_teamspace"
            assert call_args.kwargs["metrics_store_id"] == "test_stream"
            assert call_args.kwargs["persisted"] is True
            assert call_args.kwargs["phase"] == PhaseType.COMPLETED

    def test_inform_done_with_timestamps(self):
        """Test inform_done converts timestamps when store_created_at is True."""
        mock_metrics_api = Mock()

        with patch("litlogger.background.BinaryFileWriter"):
            manager = _BackgroundThread(
                teamspace_id="test_teamspace",
                metrics_store_id="test_stream",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=True,
            )

            # Use actual datetime objects (not timestamps)
            started_datetime = datetime(2024, 3, 14, 12, 0, 0)
            updated_datetime = datetime(2024, 3, 14, 12, 5, 0)

            manager.trackers = {
                "loss": MetricsTracker(
                    name="loss",
                    num_rows=10,
                    started_at=started_datetime,
                    updated_at=updated_datetime,
                )
            }

            manager.inform_done()

            # Verify datetime objects are preserved (not converted to timestamps or strings in user-facing code)
            tracker = manager.trackers["loss"]
            assert isinstance(tracker.started_at, datetime)
            assert isinstance(tracker.updated_at, datetime)
            assert tracker.started_at == started_datetime
            assert tracker.updated_at == updated_datetime

            mock_metrics_api.update_experiment_metrics.assert_called_once()


class TestBackgroundThreadRun:
    """Test _BackgroundThread.run and _run methods."""

    def test_run_sets_done_event(self):
        """Test that run method sets done_event after _run completes."""
        mock_file_store = Mock()

        with patch("litlogger.background.BinaryFileWriter", return_value=mock_file_store):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=Event(),
                done_event=Event(),
                log_dir="/test",
                version="v1",
                store_step=True,
                store_created_at=False,
            )

            with patch.object(manager, "_run"):
                manager.run()

                assert manager.done_event.is_set()

    def test_run_lifecycle_success(self):
        """Test successful _run lifecycle."""
        mock_metrics_api = Mock()
        mock_file_store = Mock()
        stop_event = Event()

        with patch("litlogger.background.BinaryFileWriter", return_value=mock_file_store):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=mock_metrics_api,
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=stop_event,
                done_event=Event(),
                log_dir="/test",
                version="v1",
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
                mock_file_store.upload.assert_called_once()
                mock_metrics_api.update_experiment_metrics.assert_called_once()

    def test_run_handles_exception(self):
        """Test _run handles exceptions and sets exception attribute."""
        mock_file_store = Mock()
        stop_event = Event()

        with patch("litlogger.background.BinaryFileWriter", return_value=mock_file_store):
            manager = _BackgroundThread(
                teamspace_id="test",
                metrics_store_id="test",
                cloud_account="test",
                metrics_api=Mock(),
                metrics_queue=Mock(),
                is_ready_event=Event(),
                stop_event=stop_event,
                done_event=Event(),
                log_dir="/test",
                version="v1",
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
        """Test complete workflow from queue to upload."""
        mock_metrics_api = Mock()
        mock_file_store = Mock()
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

        with patch("litlogger.background.BinaryFileWriter", return_value=mock_file_store):
            manager = _BackgroundThread(
                teamspace_id="test_teamspace",
                metrics_store_id="test_stream",
                cloud_account="test_account",
                metrics_api=mock_metrics_api,
                metrics_queue=metrics_queue,
                is_ready_event=is_ready_event,
                stop_event=stop_event,
                done_event=done_event,
                log_dir="/test",
                version="v1.0.0",
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
            mock_file_store.upload.assert_called_once()
            assert mock_metrics_api.update_experiment_metrics.called
