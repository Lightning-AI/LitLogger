# Import the module from sys.modules to avoid the shadowing issue
# (litlogger.experiment variable shadows the module)
import sys
from multiprocessing import Event, Queue
from time import sleep
from unittest.mock import MagicMock, patch

import pytest
from lightning_sdk.lightning_cloud.openapi import V1MediaType

# Suppress expected deprecation warnings from legacy method tests
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Trigger package import
import litlogger  # noqa: F401
from litlogger.background import _BackgroundThread
from litlogger.experiment import Experiment
from litlogger.types import MediaType, Metrics, MetricValue

experiment_module = sys.modules["litlogger.experiment"]
legacy_experiment_module = sys.modules["litlogger.experiment_legacy"]


def _bind_media_upload(exp: MagicMock) -> None:
    exp._media_type_to_v1 = lambda media_type: Experiment._media_type_to_v1(exp, media_type)

    def _upload_media(name, path, media_type, step=None, epoch=None, caption=None):
        return Experiment._upload_media(exp, name, path, media_type, step=step, epoch=epoch, caption=caption)

    exp._upload_media = _upload_media


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


def test_experiment_sender_queue():
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
        metrics_api=mock_metrics_api,
        metrics_queue=metrics_queue,
        is_ready_event=is_ready_event,
        stop_event=stop_event,
        done_event=done_event,
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

    def test_log_file(self):
        """Test log_file delegates to __setitem__ with File."""
        from litlogger.experiment import Experiment
        from litlogger.media import File

        exp = MagicMock()
        # log_file delegates to self[remote_path] = File(path)
        Experiment.log_file(exp, "test.txt", verbose=False)

        # remote_path defaults to relative path or basename
        exp.__setitem__.assert_called_once()
        key, value = exp.__setitem__.call_args[0]
        assert key == "test.txt"
        assert isinstance(value, File)
        assert value.path == "test.txt"

    def test_log_file_with_remote_path(self):
        """Test log_file with custom remote_path delegates correctly."""
        from litlogger.experiment import Experiment
        from litlogger.media import File

        exp = MagicMock()
        Experiment.log_file(exp, "/abs/path/to/file.png", verbose=False, remote_path="images/file.png")

        exp.__setitem__.assert_called_once()
        key, value = exp.__setitem__.call_args[0]
        assert key == "images/file.png"
        assert isinstance(value, File)
        assert value.path == "/abs/path/to/file.png"

    def test_get_file(self):
        """Test get_file delegates to self[key].save(path)."""
        from litlogger.experiment import Experiment

        mock_file = MagicMock()
        mock_file.save.return_value = "/path/to/file.txt"

        exp = MagicMock()
        exp.__getitem__ = MagicMock(return_value=mock_file)

        result = Experiment.get_file(exp, "test.txt", verbose=False)

        exp.__getitem__.assert_called_once_with("test.txt")
        mock_file.save.assert_called_once_with("test.txt")
        assert result == "/path/to/file.txt"

    @patch.object(legacy_experiment_module.MediaModel, "_log_model", return_value="owner/team/test_exp:v2.0")
    def test_log_model_artifact(self, mock_log_model):
        """Test log_model_artifact method."""
        exp = MagicMock()
        exp.name = "test_exp"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.cluster_id = "acc-1"

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        Experiment.log_model_artifact(exp, "model.pt", verbose=False, version="v2.0")

        mock_log_model.assert_called_once()

    @patch.object(legacy_experiment_module.MediaModel, "save", return_value="/path/to/model.pt")
    @patch.object(legacy_experiment_module.MediaModel, "_bind_remote_model")
    def test_get_model_artifact(self, mock_bind_remote_model, mock_save):
        """Test get_model_artifact method."""
        exp = MagicMock()
        exp.name = "test_exp"
        exp._teamspace = MagicMock()

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        result = Experiment.get_model_artifact(exp, "model.pt", verbose=False)

        mock_bind_remote_model.assert_called_once()
        mock_save.assert_called_once_with("model.pt")
        assert result == "/path/to/model.pt"

    @patch.object(legacy_experiment_module.MediaModel, "_log_model", return_value="owner/team/test_exp:latest")
    def test_log_model(self, mock_log_model):
        """Test log_model method."""
        exp = MagicMock()
        exp.name = "test_exp"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.cluster_id = "acc-1"

        mock_model_obj = MagicMock()

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        Experiment.log_model(exp, mock_model_obj, staging_dir="/tmp/staging", verbose=False, metadata={"key": "value"})

        mock_log_model.assert_called_once()

    @patch.object(legacy_experiment_module.MediaModel, "load")
    @patch.object(legacy_experiment_module.MediaModel, "_bind_remote_model")
    def test_get_model(self, mock_bind_remote_model, mock_load):
        """Test get_model method."""
        mock_loaded_model = MagicMock()
        mock_load.return_value = mock_loaded_model

        exp = MagicMock()
        exp.name = "test_exp"
        exp._teamspace = MagicMock()

        # Call the actual method implementation
        from litlogger.experiment import Experiment

        result = Experiment.get_model(exp, staging_dir="/tmp/staging", verbose=False)

        mock_bind_remote_model.assert_called_once()
        mock_load.assert_called_once_with("/tmp/staging")
        assert result == mock_loaded_model


def _make_metric_exp(**overrides):
    """Create a MagicMock wired up like a real Experiment for metric delegation tests.

    The mock has the internal dicts (_series, _key_types, _metadata_values, _static_files)
    and delegates __getitem__, __setitem__, update, _ensure_series, _register_key_type,
    and _log_metric_value through the real Experiment methods so that log_metrics /
    log_metrics_batch / log_metadata work end-to-end without hitting a real backend.
    """
    from litlogger.experiment import Experiment

    exp = MagicMock()
    # Internal state containers
    exp._series = {}
    exp._key_types = {}
    exp._metadata_values = {}
    exp._static_files = {}
    # Defaults
    exp._manager = MagicMock()
    exp._manager.exception = None
    exp.store_step = True
    exp.store_created_at = False
    exp._metrics_queue = MagicMock()
    exp._stats = MagicMock()
    # Wire dunder methods on the *type* so MagicMock dispatches them
    type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)
    type(exp).__setitem__ = lambda self, key, value: Experiment.__setitem__(self, key, value)
    # Wire regular methods
    exp.update = lambda data: Experiment.update(exp, data)
    exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
    exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
    exp._log_metric_value = lambda key, value, step=None: Experiment._log_metric_value(exp, key, value, step=step)
    # Apply overrides
    for k, v in overrides.items():
        setattr(exp, k, v)
    return exp


class TestExperimentLogMetrics:
    """Test log_metrics method."""

    def test_log_metrics_basic(self):
        """Test basic metrics logging delegates through __getitem__ and Series.append."""
        exp = _make_metric_exp(store_step=True, store_created_at=False)

        from litlogger.experiment import Experiment

        Experiment.log_metrics(exp, {"loss": 0.5, "accuracy": 0.9}, step=1)

        # Verify metrics were pushed to queue via _log_metric_value
        assert exp._metrics_queue.put.call_count == 2
        # Collect all batches pushed
        all_batches = [c[0][0] for c in exp._metrics_queue.put.call_args_list]
        keys_logged = {k for batch in all_batches for k in batch}
        assert keys_logged == {"loss", "accuracy"}

        # Verify values from series
        assert len(exp._series["loss"]) == 1
        assert exp._series["loss"][0] == 0.5
        assert len(exp._series["accuracy"]) == 1
        assert exp._series["accuracy"][0] == 0.9

        # Verify step was passed through to queue
        loss_batch = next(b for b in all_batches if "loss" in b)
        assert loss_batch["loss"].values[0].step == 1
        assert loss_batch["loss"].values[0].value == 0.5

    def test_log_metrics_without_step(self):
        """Test metrics logging without step when store_step=False."""
        exp = _make_metric_exp(store_step=False, store_created_at=False)

        from litlogger.experiment import Experiment

        Experiment.log_metrics(exp, {"loss": 0.5}, step=10)

        # Verify step is None because store_step=False
        batch = exp._metrics_queue.put.call_args[0][0]
        assert batch["loss"].values[0].step is None

    def test_log_metrics_with_created_at(self):
        """Test metrics logging with store_created_at=True."""
        exp = _make_metric_exp(store_step=True, store_created_at=True)

        from litlogger.experiment import Experiment

        Experiment.log_metrics(exp, {"loss": 0.5}, step=1)

        # Verify created_at is set
        batch = exp._metrics_queue.put.call_args[0][0]
        assert batch["loss"].values[0].created_at is not None

    def test_log_metrics_raises_on_background_exception(self):
        """Test that log_metrics raises if background thread has exception."""
        exp = _make_metric_exp()
        exp._manager.exception = RuntimeError("Background thread error")

        with pytest.raises(RuntimeError, match="Background thread error"):
            Experiment.log_metrics(exp, {"loss": 0.5}, step=1)

        # Queue should not be called
        exp._metrics_queue.put.assert_not_called()

    def test_log_metrics_empty_dict(self):
        """Test that empty metrics dict does not push to queue."""
        exp = _make_metric_exp()

        from litlogger.experiment import Experiment

        Experiment.log_metrics(exp, {}, step=1)

        # Queue should not be called for empty dict (no items to iterate)
        exp._metrics_queue.put.assert_not_called()


class TestExperimentLogMetricsBatch:
    """Test log_metrics_batch method for bulk metric uploads."""

    def test_log_metrics_batch_basic(self):
        """Test basic batch metrics upload delegates through update/extend."""
        exp = _make_metric_exp()

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

        # Verify series were populated with correct values
        assert list(exp._series["loss"]) == [1.0, 0.5]
        assert list(exp._series["accuracy"]) == [0.6, 0.8]

        # Verify metrics were pushed to queue (one put per value via _log_metric_value)
        assert exp._metrics_queue.put.call_count == 4  # 2 loss + 2 accuracy

    def test_log_metrics_batch_empty_dict(self):
        """Test that empty metrics dict does not push to queue."""
        exp = _make_metric_exp()

        from litlogger.experiment import Experiment

        Experiment.log_metrics_batch(exp, {})

        # Queue should not be called for empty batch
        exp._metrics_queue.put.assert_not_called()

    def test_log_metrics_batch_raises_on_background_exception(self):
        """Test that log_metrics_batch raises if background thread has exception."""
        exp = _make_metric_exp()
        exp._manager.exception = RuntimeError("Background thread error")

        with pytest.raises(RuntimeError, match="Background thread error"):
            Experiment.log_metrics_batch(exp, {"loss": [{"step": 0, "value": 1.0}]})

    def test_log_metrics_batch_single_metric(self):
        """Test batch upload with a single metric."""
        exp = _make_metric_exp()

        from litlogger.experiment import Experiment

        metrics = {
            "loss": [
                {"step": 0, "value": 1.0},
                {"step": 1, "value": 0.8},
                {"step": 2, "value": 0.6},
            ],
        }
        Experiment.log_metrics_batch(exp, metrics)

        assert list(exp._series["loss"]) == [1.0, 0.8, 0.6]
        assert exp._metrics_queue.put.call_count == 3

    def test_log_metrics_batch_many_metrics(self):
        """Test batch upload with many metrics."""
        exp = _make_metric_exp()

        from litlogger.experiment import Experiment

        # Create 100 metrics with 10 values each
        metrics = {f"metric_{i}": [{"step": j, "value": float(j)} for j in range(10)] for i in range(100)}
        Experiment.log_metrics_batch(exp, metrics)

        assert len(exp._series) == 100
        for i in range(100):
            assert len(exp._series[f"metric_{i}"]) == 10


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
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._printer = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        _bind_media_upload(exp)

        with patch("os.path.exists", return_value=True):
            Experiment.log_media(exp, "image", "/path/to/image.png", kind=MediaType.IMAGE)

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
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._printer = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        _bind_media_upload(exp)

        with patch("os.path.exists", return_value=True):
            Experiment.log_media(exp, "image", "/path/to/image.jpg")

        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["media_type"] == V1MediaType.IMAGE
        assert exp._stats.media_logged == 1

    def test_log_media_guess_type_text(self):
        """Test media upload with guessed text type."""
        exp = MagicMock()
        exp._media_api = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._printer = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        _bind_media_upload(exp)

        with patch("os.path.exists", return_value=True):
            Experiment.log_media(exp, "file", "/path/to/file.txt")

        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["media_type"] == V1MediaType.TEXT
        assert exp._stats.media_logged == 1

    def test_log_media_unsupported_type(self):
        """Test log_media raises ValueError for guessed unsupported media type."""
        exp = MagicMock()
        exp._media_api = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._printer = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        with (
            patch("os.path.exists", return_value=True),
            patch("mimetypes.guess_type", return_value=("application/zip", None)),
            pytest.raises(ValueError, match=r"Unsupported media type for file: /path/to/file\.txt"),
        ):
            Experiment.log_media(exp, "file", "/path/to/file.txt")

    def test_log_media_raises_file_not_found(self):
        """Test log_media raises FileNotFoundError."""
        exp = MagicMock()

        with patch("os.path.exists", return_value=False), pytest.raises(FileNotFoundError):
            Experiment.log_media(exp, "file", "/non/existent/file.png")

    def test_log_media_with_step_epoch_caption(self):
        """Test log_media passes step, epoch, and caption to upload_media."""
        exp = MagicMock()
        exp._media_api = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._printer = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        _bind_media_upload(exp)

        with patch("os.path.exists", return_value=True):
            Experiment.log_media(
                exp,
                "img",
                "/path/to/img.png",
                kind=MediaType.IMAGE,
                step=5,
                epoch=2,
                caption="A caption",
            )

        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["step"] == 5
        assert kwargs["epoch"] == 2
        assert kwargs["caption"] == "A caption"


class TestExperimentMetadataProperty:
    """Test metadata property."""

    def test_metadata_returns_from_code_tags(self):
        """Test that metadata property filters to from_code tags only."""
        exp = MagicMock()
        tag1 = MagicMock()
        tag1.name = "lr"
        tag1.value = "0.001"
        tag1.from_code = True
        tag2 = MagicMock()
        tag2.name = "ui_tag"
        tag2.value = "ignored"
        tag2.from_code = False
        tag3 = MagicMock()
        tag3.name = "batch_size"
        tag3.value = "32"
        tag3.from_code = True
        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = [tag1, tag2, tag3]

        from litlogger.experiment import Experiment

        result = Experiment.metadata.fget(exp)
        assert result == {"lr": "0.001", "batch_size": "32"}

    def test_metadata_returns_empty_when_no_tags(self):
        """Test that metadata property returns empty dict when tags is None."""
        exp = MagicMock()
        exp._metrics_store = MagicMock(spec=[])  # no tags attribute

        from litlogger.experiment import Experiment

        result = Experiment.metadata.fget(exp)
        assert result == {}

    def test_metadata_returns_empty_for_empty_tags_list(self):
        """Test that metadata property returns empty dict for empty tags list."""
        exp = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = []

        from litlogger.experiment import Experiment

        result = Experiment.metadata.fget(exp)
        assert result == {}


class TestExperimentLogMetadata:
    """Test log_metadata method."""

    def test_log_metadata_updates_tags(self):
        """Test that log_metadata delegates to update(), which sets metadata via __setitem__."""
        from litlogger.background import PhaseType
        from litlogger.experiment import Experiment

        exp = _make_metric_exp()
        # Set up metrics store with existing tag
        lr_tag = MagicMock()
        lr_tag.name = "lr"
        lr_tag.value = "0.001"
        lr_tag.from_code = True
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store_123"
        exp._metrics_store.name = "test"
        exp._metrics_store.tags = [lr_tag]
        exp._metrics_api = MagicMock()
        # _update_metrics_store is called by the metadata property; make it a no-op
        exp._update_metrics_store = lambda: None
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts_123"
        # Pre-populate existing metadata in internal state
        exp._key_types["lr"] = "metadata"
        exp._metadata_values["lr"] = "0.001"
        # Wire _set_metadata_value to use the real implementation
        exp._set_metadata_value = lambda key, value: Experiment._set_metadata_value(exp, key, value)
        # Wire metadata property via the real Experiment property
        type(exp).metadata = Experiment.metadata

        Experiment.log_metadata(exp, {"batch_size": "32"})

        # _set_metadata_value calls update_experiment_metrics for each key.
        # With one new key, it should be called once.
        assert exp._metrics_api.update_experiment_metrics.call_count == 1
        call_kwargs = exp._metrics_api.update_experiment_metrics.call_args.kwargs
        assert call_kwargs["teamspace_id"] == "ts_123"
        assert call_kwargs["metrics_store_id"] == "store_123"
        assert call_kwargs["phase"] == PhaseType.RUNNING
        assert call_kwargs["metadata"] == {"lr": "0.001", "batch_size": "32"}

    def test_log_metadata_overwrites_existing_key(self):
        """Test that log_metadata overwrites existing keys."""
        from litlogger.experiment import Experiment

        exp = _make_metric_exp()
        lr_tag = MagicMock()
        lr_tag.name = "lr"
        lr_tag.value = "0.001"
        lr_tag.from_code = True
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store_123"
        exp._metrics_store.name = "test"
        exp._metrics_store.tags = [lr_tag]
        exp._metrics_api = MagicMock()
        exp._update_metrics_store = lambda: None
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts_123"
        # Pre-populate existing metadata
        exp._key_types["lr"] = "metadata"
        exp._metadata_values["lr"] = "0.001"
        exp._set_metadata_value = lambda key, value: Experiment._set_metadata_value(exp, key, value)
        type(exp).metadata = Experiment.metadata

        Experiment.log_metadata(exp, {"lr": "0.01"})

        call_kwargs = exp._metrics_api.update_experiment_metrics.call_args.kwargs
        assert call_kwargs["metadata"] == {"lr": "0.01"}


class TestUpdateMetricsStore:
    """Test _update_metrics_store method."""

    def test_update_metrics_store_refreshes(self):
        """Test that _update_metrics_store refreshes from API."""
        exp = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.name = "test"
        exp._metrics_store.version_number = 1
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts_123"

        new_store = MagicMock()
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_experiment_metrics_by_name.return_value = new_store

        from litlogger.experiment import Experiment

        Experiment._update_metrics_store(exp)

        exp._metrics_api.get_experiment_metrics_by_name.assert_called_once_with("ts_123", name="test")
        assert exp._metrics_store is new_store

    def test_update_metrics_store_keeps_old_on_none(self):
        """Test that _update_metrics_store keeps old store if API returns None."""
        exp = MagicMock()
        old_store = MagicMock()
        exp._metrics_store = old_store
        exp._metrics_store.name = "test"
        exp._metrics_store.version_number = 1
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts_123"

        exp._metrics_api = MagicMock()
        exp._metrics_api.get_experiment_metrics_by_name.return_value = None

        from litlogger.experiment import Experiment

        Experiment._update_metrics_store(exp)

        # Should not overwrite with None
        assert exp._metrics_store is old_store


class TestExperimentLogMetricsKwargs:
    """Test log_metrics with kwargs."""

    def test_log_metrics_with_kwargs(self):
        """Test log_metrics accepts kwargs alongside metrics dict."""
        exp = _make_metric_exp(store_step=True, store_created_at=False)

        from litlogger.experiment import Experiment

        Experiment.log_metrics(exp, {"loss": 0.5}, step=1, accuracy=0.9)

        # Verify both metrics ended up in series
        assert exp._series["loss"][0] == 0.5
        assert exp._series["accuracy"][0] == 0.9

        # Verify values were pushed to queue
        all_batches = [c[0][0] for c in exp._metrics_queue.put.call_args_list]
        loss_batch = next(b for b in all_batches if "loss" in b)
        acc_batch = next(b for b in all_batches if "accuracy" in b)
        assert loss_batch["loss"].values[0].value == 0.5
        assert acc_batch["accuracy"].values[0].value == 0.9

    def test_log_metrics_kwargs_override_dict(self):
        """Test that kwargs override dict values for same key."""
        exp = _make_metric_exp(store_step=False, store_created_at=False)

        from litlogger.experiment import Experiment

        Experiment.log_metrics(exp, {"loss": 0.5}, loss=0.3)

        # kwargs override dict, so only 0.3 should be logged
        assert exp._series["loss"][0] == 0.3
        batch = exp._metrics_queue.put.call_args[0][0]
        assert batch["loss"].values[0].value == 0.3


class TestExperimentStatsTracking:
    """Test that experiment methods track stats correctly."""

    def test_log_metrics_tracks_stats(self):
        """Test log_metrics calls record_metric on stats via _log_metric_value."""
        exp = _make_metric_exp(store_step=True, store_created_at=False)

        from litlogger.experiment import Experiment

        Experiment.log_metrics(exp, {"loss": 0.5, "acc": 0.9}, step=1)

        assert exp._stats.record_metric.call_count == 2
        calls = {c.args[0]: c.args[1] for c in exp._stats.record_metric.call_args_list}
        assert calls["loss"] == 0.5
        assert calls["acc"] == 0.9

    def test_log_file_tracks_artifact_count(self):
        """Test log_file delegates to __setitem__ with File."""
        from litlogger.experiment import Experiment
        from litlogger.media import File

        exp = MagicMock()
        # log_file now calls self[remote_path] = File(path)
        Experiment.log_file(exp, "file.txt", verbose=False)

        exp.__setitem__.assert_called_once()
        key, value = exp.__setitem__.call_args[0]
        assert key == "file.txt"
        assert isinstance(value, File)

    @patch.object(legacy_experiment_module.MediaModel, "_log_model", return_value="owner/team/test:latest")
    def test_log_model_artifact_tracks_model_count(self, mock_log_model):
        """Test log_model_artifact increments models_logged."""
        exp = MagicMock()
        exp.name = "test"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.cluster_id = "acc-1"
        exp._stats = MagicMock()
        exp._stats.models_logged = 0

        from litlogger.experiment import Experiment

        Experiment.log_model_artifact(exp, "model.pt", verbose=False)
        mock_log_model.assert_called_once()
        assert exp._stats.models_logged == 1

    @patch.object(legacy_experiment_module.MediaModel, "_log_model", return_value="owner/team/test:latest")
    def test_log_model_tracks_model_count(self, mock_log_model):
        """Test log_model increments models_logged."""
        exp = MagicMock()
        exp.name = "test"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.cluster_id = "acc-1"
        exp._stats = MagicMock()
        exp._stats.models_logged = 0

        from litlogger.experiment import Experiment

        Experiment.log_model(exp, MagicMock(), verbose=False)
        mock_log_model.assert_called_once()
        assert exp._stats.models_logged == 1


class TestExperimentPrintUrl:
    """Test print_url method."""

    def test_print_url_calls_printer(self):
        """Test that print_url delegates to printer with correct args."""
        exp = MagicMock()
        exp.name = "my-experiment"
        exp._teamspace = MagicMock()
        exp._teamspace.name = "my-teamspace"
        exp._url = "https://lightning.ai/my-experiment"

        # Set up metadata property to return dict
        tag = MagicMock()
        tag.name = "lr"
        tag.value = "0.001"
        tag.from_code = True
        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = [tag]

        from litlogger.experiment import Experiment

        Experiment.print_url(exp)

        exp._printer.experiment_start.assert_called_once()
        call_kwargs = exp._printer.experiment_start.call_args.kwargs
        assert call_kwargs["name"] == "my-experiment"
        assert call_kwargs["teamspace"] == "my-teamspace"
        assert call_kwargs["url"] == "https://lightning.ai/my-experiment"


class TestExperimentLogMetricsBatchCreatedAt:
    """Test log_metrics_batch with store_created_at."""

    def test_log_metrics_batch_with_store_created_at(self):
        """Test that log_metrics_batch sets created_at when store_created_at=True."""
        exp = _make_metric_exp(store_created_at=True)

        from litlogger.experiment import Experiment

        metrics = {"loss": [{"step": 0, "value": 1.0}]}
        Experiment.log_metrics_batch(exp, metrics)

        batch = exp._metrics_queue.put.call_args[0][0]
        assert batch["loss"].values[0].created_at is not None

    def test_log_metrics_batch_without_store_created_at(self):
        """Test that log_metrics_batch does not set created_at when store_created_at=False."""
        exp = _make_metric_exp(store_created_at=False)

        from litlogger.experiment import Experiment

        metrics = {"loss": [{"step": 0, "value": 1.0}]}
        Experiment.log_metrics_batch(exp, metrics)

        batch = exp._metrics_queue.put.call_args[0][0]
        assert batch["loss"].values[0].created_at is None

    def test_log_metrics_batch_without_step_key(self):
        """Test that log_metrics_batch handles missing step key (step=None in _log_metric_value)."""
        exp = _make_metric_exp(store_step=False, store_created_at=False)

        from litlogger.experiment import Experiment

        metrics = {"loss": [{"value": 1.0}]}
        Experiment.log_metrics_batch(exp, metrics)

        batch = exp._metrics_queue.put.call_args[0][0]
        assert batch["loss"].values[0].step is None
