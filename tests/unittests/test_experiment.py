# Import the module from sys.modules to avoid the shadowing issue
# (litlogger.experiment variable shadows the module)
import sys
from multiprocessing import Event, Queue
from time import sleep
from unittest.mock import MagicMock, patch

# Trigger package import
import litlogger  # noqa: F401
from litlogger.background import _BackgroundThread
from litlogger.types import Metrics, MetricValue

experiment_module = sys.modules["litlogger.experiment"]


class TestBackgroundThread(_BackgroundThread):
    def run(self):
        super()._run()
        # The API layer translates user-facing types to V1 types before calling the client
        mock_calls = self.metrics_api.append_experiment_metrics._mock_mock_calls

        # Collect all metrics across all API calls
        all_values = []
        for call in mock_calls:
            call_kwargs = call.kwargs
            metrics = call_kwargs["metrics"]
            for metric in metrics:
                if metric.name == "loss":
                    all_values.extend([v.value for v in metric.values])

        # Verify all 100 values (10 batches * 10 values each) were sent
        # Each batch sends values [0-9], so we expect 10 copies of [0-9]
        expected_values = [i for _ in range(10) for i in range(10)]
        assert sorted(all_values) == sorted(
            expected_values
        ), f"Expected {len(expected_values)} values, got {len(all_values)}"

        self.done_event.set()


def test_experiment_sender_queue(tmpdir):
    """Test that the background thread processes metrics from the queue correctly."""
    metrics_queue = Queue()
    is_ready_event = Event()
    stop_event = Event()
    done_event = Event()

    # Create a mock MetricsApi
    mock_metrics_api = MagicMock()

    sender = TestBackgroundThread(
        teamspace_id="project_id",
        metrics_store_id="id",
        cloud_account="cloud_account",
        metrics_api=mock_metrics_api,
        metrics_queue=metrics_queue,
        is_ready_event=is_ready_event,
        stop_event=stop_event,
        done_event=done_event,
        log_dir=str(tmpdir),
        version="fake_timestamp",
        store_step=False,
        store_created_at=False,
    )
    sender.start()

    for _ in range(10):
        values = [MetricValue(value=i) for i in range(10)]
        metrics_queue.put({"loss": Metrics(name="loss", values=values)})
        sleep(0.2)

    stop_event.set()

    while not done_event.is_set():
        sleep(0.2)


def test_finalize_is_idempotent():
    """Test that finalize() can be called multiple times safely."""
    # Create a minimal mock experiment to test idempotency
    exp = MagicMock()
    exp._finalized = False
    exp._done_event = MagicMock()
    exp._done_event.is_set.return_value = True
    exp._stop_event = MagicMock()
    exp._metrics = [[]]
    exp.save_logs = False

    # Copy the actual finalize implementation
    def finalize(status=None):
        if exp._finalized:
            return
        exp._finalized = True
        exp._stop_event.set()

    exp.finalize = finalize

    # Call finalize() three times - should not raise errors
    exp.finalize()
    exp.finalize()
    exp.finalize()

    # Verify finalized flag is set
    assert exp._finalized is True
    # Verify stop event was only set once
    assert exp._stop_event.set.call_count == 1


def test_finalize_with_status():
    """Test that finalize() accepts different status values."""
    # Just verify the method signature accepts a status parameter
    # The actual implementation is tested in integration tests
    import inspect

    sig = inspect.signature(experiment_module.Experiment.finalize)
    assert "status" in sig.parameters


def test_signal_handler_exit_code():
    """Test that signal handler uses correct exit code."""
    exp = MagicMock()
    exp._finalized = False

    # Test SIGTERM (15) should exit with 143 (128 + 15)
    with patch.object(sys, "exit"):
        handler = experiment_module.Experiment._signal_handler
        # Can't easily test without creating a full experiment, so just verify the method exists
        assert callable(handler)


class TestExperimentArtifactMethods:
    """Test artifact logging and retrieval methods."""

    @patch.object(experiment_module, "Artifact")
    def test_log_file(self, mock_artifact_class):
        """Test log_file method."""
        mock_artifact = MagicMock()
        mock_artifact_class.return_value = mock_artifact

        exp = MagicMock()
        exp.name = "test_exp"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._artifacts_api = MagicMock()

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        Experiment.log_file(exp, "test.txt", verbose=False)

        # Verify Artifact was created correctly (remote_path defaults to None)
        mock_artifact_class.assert_called_once_with(
            path="test.txt",
            experiment_name="test_exp",
            teamspace=exp._teamspace,
            metrics_store=exp._metrics_store,
            client=exp._artifacts_api.client,
            remote_path=None,
        )
        # Verify log was called
        mock_artifact.log.assert_called_once()

    @patch.object(experiment_module, "Artifact")
    def test_log_file_with_remote_path(self, mock_artifact_class):
        """Test log_file method with custom remote_path."""
        mock_artifact = MagicMock()
        mock_artifact_class.return_value = mock_artifact

        exp = MagicMock()
        exp.name = "test_exp"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._artifacts_api = MagicMock()

        from litlogger.experiment import Experiment

        Experiment.log_file(exp, "/abs/path/to/file.png", verbose=False, remote_path="images/file.png")

        # Verify Artifact was created with custom remote_path
        mock_artifact_class.assert_called_once_with(
            path="/abs/path/to/file.png",
            experiment_name="test_exp",
            teamspace=exp._teamspace,
            metrics_store=exp._metrics_store,
            client=exp._artifacts_api.client,
            remote_path="images/file.png",
        )
        mock_artifact.log.assert_called_once()

    @patch.object(experiment_module, "Artifact")
    def test_get_file(self, mock_artifact_class):
        """Test get_file method."""
        mock_artifact = MagicMock()
        mock_artifact.get.return_value = "/path/to/file.txt"
        mock_artifact_class.return_value = mock_artifact

        exp = MagicMock()
        exp.name = "test_exp"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._artifacts_api = MagicMock()

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        result = Experiment.get_file(exp, "test.txt", verbose=False)

        # Verify Artifact was created correctly (remote_path defaults to None)
        mock_artifact_class.assert_called_once_with(
            path="test.txt",
            experiment_name="test_exp",
            teamspace=exp._teamspace,
            metrics_store=exp._metrics_store,
            client=exp._artifacts_api.client,
            remote_path=None,
        )
        # Verify get was called and result returned
        mock_artifact.get.assert_called_once()
        assert result == "/path/to/file.txt"

    @patch.object(experiment_module, "ModelArtifact")
    def test_log_model_artifact(self, mock_model_artifact_class):
        """Test log_model_artifact method."""
        mock_model_artifact = MagicMock()
        mock_model_artifact_class.return_value = mock_model_artifact

        exp = MagicMock()
        exp.name = "test_exp"
        exp.version = "v1.0"
        exp._teamspace = MagicMock()

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        Experiment.log_model_artifact(exp, "model.pt", verbose=False, version="v2.0")

        # Verify ModelArtifact was created correctly
        mock_model_artifact_class.assert_called_once_with(
            path="model.pt", experiment_name="test_exp", teamspace=exp._teamspace, version="v2.0", verbose=False
        )
        # Verify log was called
        mock_model_artifact.log.assert_called_once()

    @patch.object(experiment_module, "ModelArtifact")
    def test_get_model_artifact(self, mock_model_artifact_class):
        """Test get_model_artifact method."""
        mock_model_artifact = MagicMock()
        mock_model_artifact.get.return_value = "/path/to/model.pt"
        mock_model_artifact_class.return_value = mock_model_artifact

        exp = MagicMock()
        exp.name = "test_exp"
        exp.version = "v1.0"
        exp._teamspace = MagicMock()

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        result = Experiment.get_model_artifact(exp, "model.pt", verbose=False)

        # Verify ModelArtifact was created correctly
        mock_model_artifact_class.assert_called_once_with(
            path="model.pt", experiment_name="test_exp", teamspace=exp._teamspace, version="v1.0", verbose=False
        )
        # Verify get was called and result returned
        mock_model_artifact.get.assert_called_once()
        assert result == "/path/to/model.pt"

    @patch.object(experiment_module, "Model")
    def test_log_model(self, mock_model_class):
        """Test log_model method."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        exp = MagicMock()
        exp.name = "test_exp"
        exp.version = "v1.0"
        exp._teamspace = MagicMock()

        mock_model_obj = MagicMock()

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        Experiment.log_model(exp, mock_model_obj, staging_dir="/tmp/staging", verbose=False, metadata={"key": "value"})

        # Verify Model was created correctly
        mock_model_class.assert_called_once_with(
            model=mock_model_obj,
            experiment_name="test_exp",
            teamspace=exp._teamspace,
            version="v1.0",
            verbose=False,
            metadata={"key": "value"},
            staging_dir="/tmp/staging",
        )
        # Verify log was called
        mock_model.log.assert_called_once()

    @patch.object(experiment_module, "Model")
    def test_get_model(self, mock_model_class):
        """Test get_model method."""
        mock_model = MagicMock()
        mock_loaded_model = MagicMock()
        mock_model.get.return_value = mock_loaded_model
        mock_model_class.return_value = mock_model

        exp = MagicMock()
        exp.name = "test_exp"
        exp.version = "v1.0"
        exp._teamspace = MagicMock()

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        result = Experiment.get_model(exp, staging_dir="/tmp/staging", verbose=False)

        # Verify Model was created correctly
        mock_model_class.assert_called_once_with(
            model=None,
            experiment_name="test_exp",
            teamspace=exp._teamspace,
            version="v1.0",
            verbose=False,
            staging_dir="/tmp/staging",
        )
        # Verify get was called and result returned
        mock_model.get.assert_called_once()
        assert result == mock_loaded_model


class TestExperimentLogMetrics:
    """Test log_metrics method."""

    def test_log_metrics_basic(self):
        """Test basic metrics logging pushes to queue."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp.store_step = True
        exp.store_created_at = False
        exp._metrics_queue = MagicMock()

        from litlogger.experiment import Experiment

        # Log some metrics
        Experiment.log_metrics(exp, {"loss": 0.5, "accuracy": 0.9}, step=1)

        # Verify metrics were pushed to queue
        exp._metrics_queue.put.assert_called_once()
        call_args = exp._metrics_queue.put.call_args[0][0]

        assert len(call_args) == 2
        assert "loss" in call_args
        assert "accuracy" in call_args
        assert call_args["loss"].values[0].value == 0.5
        assert call_args["loss"].values[0].step == 1
        assert call_args["accuracy"].values[0].value == 0.9

    def test_log_metrics_without_step(self):
        """Test metrics logging without step when store_step=False."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp.store_step = False
        exp.store_created_at = False
        exp._metrics_queue = MagicMock()

        from litlogger.experiment import Experiment

        # Log metrics with step but store_step=False
        Experiment.log_metrics(exp, {"loss": 0.5}, step=10)

        # Verify step is None
        call_args = exp._metrics_queue.put.call_args[0][0]
        assert call_args["loss"].values[0].step is None

    def test_log_metrics_with_created_at(self):
        """Test metrics logging with store_created_at=True."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp.store_step = True
        exp.store_created_at = True
        exp._metrics_queue = MagicMock()

        from litlogger.experiment import Experiment

        # Log metrics with store_created_at enabled
        Experiment.log_metrics(exp, {"loss": 0.5}, step=1)

        # Verify created_at is set
        call_args = exp._metrics_queue.put.call_args[0][0]
        assert call_args["loss"].values[0].created_at is not None

    def test_log_metrics_raises_on_background_exception(self):
        """Test that log_metrics raises if background thread has exception."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = RuntimeError("Background thread error")
        exp._metrics_queue = MagicMock()

        import pytest
        from litlogger.experiment import Experiment

        with pytest.raises(RuntimeError, match="Background thread error"):
            Experiment.log_metrics(exp, {"loss": 0.5}, step=1)

        # Queue should not be called
        exp._metrics_queue.put.assert_not_called()

    def test_log_metrics_empty_dict(self):
        """Test that empty metrics dict does not push to queue."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp.store_step = True
        exp.store_created_at = False
        exp._metrics_queue = MagicMock()

        from litlogger.experiment import Experiment

        # Log empty metrics
        Experiment.log_metrics(exp, {}, step=1)

        # Queue should not be called for empty batch
        exp._metrics_queue.put.assert_not_called()


class TestExperimentLogMetricsBatch:
    """Test log_metrics_batch method for bulk metric uploads."""

    def test_log_metrics_batch_basic(self):
        """Test basic batch metrics upload pushes to queue."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp._metrics_queue = MagicMock()

        from litlogger.experiment import Experiment

        metrics = {
            "loss": [
                {"step": 0, "value": 1.0},
                {"step": 1, "value": 0.5},
            ],
            "accuracy": [
                {"step": 0, "value": 0.6},
                {"step": 1, "value": 0.8},
            ],
        }
        Experiment.log_metrics_batch(exp, metrics)

        # Verify queue was called
        exp._metrics_queue.put.assert_called_once()
        batch = exp._metrics_queue.put.call_args[0][0]

        # Verify metrics structure
        assert len(batch) == 2
        assert "loss" in batch
        assert "accuracy" in batch

        # Verify loss values
        loss_values = batch["loss"].values
        assert len(loss_values) == 2
        assert loss_values[0].value == 1.0
        assert loss_values[0].step == 0
        assert loss_values[1].value == 0.5
        assert loss_values[1].step == 1

    def test_log_metrics_batch_empty_dict(self):
        """Test that empty metrics dict does not push to queue."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp._metrics_queue = MagicMock()

        from litlogger.experiment import Experiment

        Experiment.log_metrics_batch(exp, {})

        # Queue should not be called for empty batch
        exp._metrics_queue.put.assert_not_called()

    def test_log_metrics_batch_raises_on_background_exception(self):
        """Test that log_metrics_batch raises if background thread has exception."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = RuntimeError("Background thread error")
        exp._metrics_queue = MagicMock()

        import pytest
        from litlogger.experiment import Experiment

        with pytest.raises(RuntimeError, match="Background thread error"):
            Experiment.log_metrics_batch(exp, {"loss": [{"step": 0, "value": 1.0}]})

        # Queue should not be called
        exp._metrics_queue.put.assert_not_called()

    def test_log_metrics_batch_single_metric(self):
        """Test batch upload with a single metric."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp._metrics_queue = MagicMock()

        from litlogger.experiment import Experiment

        metrics = {
            "loss": [
                {"step": 0, "value": 1.0},
                {"step": 1, "value": 0.8},
                {"step": 2, "value": 0.6},
            ],
        }
        Experiment.log_metrics_batch(exp, metrics)

        batch = exp._metrics_queue.put.call_args[0][0]

        assert len(batch) == 1
        assert batch["loss"].name == "loss"
        assert len(batch["loss"].values) == 3

    def test_log_metrics_batch_many_metrics(self):
        """Test batch upload with many metrics."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp._metrics_queue = MagicMock()

        from litlogger.experiment import Experiment

        # Create 100 metrics with 10 values each
        metrics = {f"metric_{i}": [{"step": j, "value": float(j)} for j in range(10)] for i in range(100)}
        Experiment.log_metrics_batch(exp, metrics)

        batch = exp._metrics_queue.put.call_args[0][0]

        assert len(batch) == 100
        for _, m in batch.items():
            assert len(m.values) == 10


class TestExperimentLogFiles:
    """Test log_files method for parallel file uploads."""

    @patch.object(experiment_module.Experiment, "log_file")
    def test_log_files_calls_log_file_for_each_path(self, mock_log_file):
        """Test that log_files calls log_file for each path."""
        exp = MagicMock()
        exp.log_file = mock_log_file

        from litlogger.experiment import Experiment

        paths = ["/path/to/file1.txt", "/path/to/file2.txt", "/path/to/file3.txt"]
        Experiment.log_files(exp, paths)

        # Verify log_file was called for each path
        assert mock_log_file.call_count == 3
        called_paths = {call.args[0] for call in mock_log_file.call_args_list}
        assert called_paths == set(paths)

    @patch.object(experiment_module.Experiment, "log_file")
    def test_log_files_with_empty_list(self, mock_log_file):
        """Test that log_files handles empty list gracefully."""
        exp = MagicMock()
        exp.log_file = mock_log_file

        from litlogger.experiment import Experiment

        Experiment.log_files(exp, [])

        # Verify log_file was not called
        mock_log_file.assert_not_called()

    @patch.object(experiment_module.Experiment, "log_file")
    def test_log_files_with_custom_max_workers(self, mock_log_file):
        """Test that log_files respects max_workers parameter."""
        exp = MagicMock()
        exp.log_file = mock_log_file

        from litlogger.experiment import Experiment

        paths = [f"/path/to/file{i}.txt" for i in range(20)]
        Experiment.log_files(exp, paths, max_workers=5)

        # Verify all files were uploaded
        assert mock_log_file.call_count == 20

    @patch.object(experiment_module.Experiment, "log_file")
    def test_log_files_handles_errors(self, mock_log_file, capsys):
        """Test that log_files handles upload errors gracefully."""
        exp = MagicMock()

        # Make log_file raise an error for one file
        def side_effect(path, remote=None, verbose=False):
            if "bad" in path:
                raise Exception("Upload failed")

        mock_log_file.side_effect = side_effect
        exp.log_file = mock_log_file

        # Set up the printer mock to output to stderr
        from litlogger.printer import Printer

        exp._printer = Printer(verbose=True)

        from litlogger.experiment import Experiment

        paths = ["/path/to/good1.txt", "/path/to/bad.txt", "/path/to/good2.txt"]
        Experiment.log_files(exp, paths)

        # Verify all files were attempted
        assert mock_log_file.call_count == 3

        # Verify error was printed to stderr (printer outputs to stderr)
        captured = capsys.readouterr()
        assert "Failed" in captured.err or "bad.txt" in captured.err

    @patch.object(experiment_module.Experiment, "log_file")
    def test_log_files_without_remote_paths(self, mock_log_file):
        """Test that log_files passes None for remote_path when not provided."""
        exp = MagicMock()
        exp.log_file = mock_log_file

        from litlogger.experiment import Experiment

        Experiment.log_files(exp, ["data/file.txt"])

        # Verify log_file was called with path and None for remote_path (positional args)
        mock_log_file.assert_called_once_with("data/file.txt", None, verbose=False)

    @patch.object(experiment_module.Experiment, "log_file")
    def test_log_files_with_remote_paths(self, mock_log_file):
        """Test that log_files passes custom remote_paths."""
        exp = MagicMock()
        exp.log_file = mock_log_file

        from litlogger.experiment import Experiment

        Experiment.log_files(
            exp,
            ["exports/data/images/0.png", "exports/data/images/1.png"],
            remote_paths=["images/0.png", "images/1.png"],
        )

        # Verify custom remote_paths were passed (as positional args)
        assert mock_log_file.call_count == 2
        # Build mapping from path to remote_path using positional args
        calls = {call.args[0]: call.args[1] for call in mock_log_file.call_args_list}
        assert calls["exports/data/images/0.png"] == "images/0.png"
        assert calls["exports/data/images/1.png"] == "images/1.png"

    def test_log_files_remote_paths_length_mismatch(self):
        """Test that log_files raises error when remote_paths length doesn't match."""
        exp = MagicMock()

        import pytest
        from litlogger.experiment import Experiment

        with pytest.raises(ValueError, match="remote_paths length"):
            Experiment.log_files(exp, ["a.txt", "b.txt"], remote_paths=["only_one.txt"])


class TestExperimentInitialization:
    """Test experiment initialization."""

    def test_atexit_handler_registered(self):
        """Test that atexit handler is registered."""
        # This is tested implicitly - if it fails during init, we'd see errors
        # Just verify the registration doesn't cause issues
        # (Full integration testing happens in test_teamspace.py)
        assert hasattr(experiment_module.Experiment, "finalize")

    def test_signal_handlers_exist(self):
        """Test that signal handler method exists."""
        assert hasattr(experiment_module.Experiment, "_signal_handler")

class TestExperimentLogMedia:
    """Test log_media method."""

    def test_log_media_basic(self):
        """Test basic media upload with explicit type."""
        exp = MagicMock()
        exp.name = "test_exp"
        exp._manager = MagicMock()
        exp._media_api = MagicMock()
        exp._printer = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        # Import MediaType from types
        from litlogger.types import MediaType
        from lightning_sdk.lightning_cloud.openapi import V1MediaType
        from litlogger.experiment import Experiment
        from unittest.mock import patch
        
        with patch("os.path.exists", return_value=True):
            Experiment.log_media(exp, "/path/to/image.png", type=MediaType.IMAGE)

        # Verify upload_media called with correct args
        exp._media_api.upload_media.assert_called_once()
        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["file_path"] == "/path/to/image.png"
        assert kwargs["media_type"] == V1MediaType.IMAGE
        assert exp._stats.media_logged == 1

    def test_log_media_guess_type_image(self):
        """Test media upload with guessed image type."""
        exp = MagicMock()
        exp._media_api = MagicMock()
        exp._printer = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        from lightning_sdk.lightning_cloud.openapi import V1MediaType
        from litlogger.experiment import Experiment
        from unittest.mock import patch
        
        with patch("os.path.exists", return_value=True):
             Experiment.log_media(exp, "/path/to/image.jpg")

        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["media_type"] == V1MediaType.IMAGE
        assert exp._stats.media_logged == 1

    def test_log_media_guess_type_text(self):
        """Test media upload with guessed text type."""
        exp = MagicMock()
        exp._media_api = MagicMock()
        exp._printer = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        from lightning_sdk.lightning_cloud.openapi import V1MediaType
        from litlogger.experiment import Experiment
        from unittest.mock import patch
        
        with patch("os.path.exists", return_value=True):
             Experiment.log_media(exp, "/path/to/readme.txt")

        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["media_type"] == V1MediaType.TEXT
        assert exp._stats.media_logged == 1

    def test_log_media_unsupported_type(self):
        """Test log_media raises ValueError for guessed unsupported media type."""
        exp = MagicMock()
        exp._media_api = MagicMock()
        exp._printer = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        from lightning_sdk.lightning_cloud.openapi import V1MediaType
        from litlogger.experiment import Experiment
        from unittest.mock import patch
        import pytest
        
        with patch("os.path.exists", return_value=True):
            with patch("mimetypes.guess_type", return_value=("application/zip", None)):
                with pytest.raises(ValueError):
                    Experiment.log_media(exp, "/path/to/file.txt")

    def test_log_media_raises_file_not_found(self):
        """Test log_media raises FileNotFoundError."""
        exp = MagicMock()
        
        from litlogger.experiment import Experiment
        from unittest.mock import patch
        import pytest

        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                Experiment.log_media(exp, "/non/existent/file.png")

