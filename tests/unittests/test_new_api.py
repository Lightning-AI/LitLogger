# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for the new dict-like experiment API."""

import sys
from unittest.mock import MagicMock, patch

# Trigger package import
import litlogger  # noqa: F401
import pytest
from litlogger.experiment import Experiment
from litlogger.media import File
from litlogger.series import Series

experiment_module = sys.modules["litlogger.experiment"]


class TestExperimentDictAPI:
    """Test __getitem__ and __setitem__ on Experiment."""

    def _make_experiment(self) -> MagicMock:
        """Create a mock experiment with required attributes for dict API testing."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._series = {}
        exp._metadata_values = {}
        exp._static_files = {}
        return exp

    def test_getitem_creates_series_for_new_key(self):
        """Test that accessing a new key returns a fresh Series."""
        exp = self._make_experiment()
        result = Experiment.__getitem__(exp, "new-key")

        assert isinstance(result, Series)
        assert "new-key" in exp._series

    def test_getitem_returns_same_series(self):
        """Test that accessing the same key returns the same Series object."""
        exp = self._make_experiment()
        s1 = Experiment.__getitem__(exp, "loss")
        s2 = Experiment.__getitem__(exp, "loss")

        assert s1 is s2

    def test_getitem_returns_metadata(self):
        """Test that accessing a metadata key returns the string value."""
        exp = self._make_experiment()
        exp._key_types["tag"] = "metadata"
        exp._metadata_values["tag"] = "hello"

        result = Experiment.__getitem__(exp, "tag")
        assert result == "hello"

    def test_getitem_returns_static_file(self):
        """Test that accessing a static file key returns the File object."""
        exp = self._make_experiment()
        f = File("test.txt")
        exp._key_types["my-file"] = "static_file"
        exp._static_files["my-file"] = f

        result = Experiment.__getitem__(exp, "my-file")
        assert result is f

    def test_getitem_returns_metric_series(self):
        """Test that accessing a metric key returns the Series."""
        exp = self._make_experiment()
        series = Series(exp, "loss")
        series._type = "metric"
        exp._key_types["loss"] = "metric"
        exp._series["loss"] = series

        result = Experiment.__getitem__(exp, "loss")
        assert result is series

    def test_setitem_metadata(self):
        """Test setting a string value as metadata."""
        exp = self._make_experiment()
        Experiment.__setitem__(exp, "tag", "value")

        assert exp._key_types["tag"] == "metadata"
        assert exp._metadata_values["tag"] == "value"
        exp._set_metadata_value.assert_called_once_with("tag", "value")

    def test_setitem_static_file(self):
        """Test setting a File value as static file."""
        exp = self._make_experiment()
        f = File("test.txt")
        Experiment.__setitem__(exp, "my-file", f)

        assert exp._key_types["my-file"] == "static_file"
        assert exp._static_files["my-file"] is f
        exp._set_static_file.assert_called_once_with("my-file", f)

    def test_setitem_same_type_overwrite_ok(self):
        """Test that overwriting a key with the same type is allowed."""
        exp = self._make_experiment()
        exp._key_types["tag"] = "metadata"
        exp._metadata_values["tag"] = "old-value"

        Experiment.__setitem__(exp, "tag", "new-value")
        assert exp._metadata_values["tag"] == "new-value"

    def test_setitem_different_type_raises(self):
        """Test that overwriting a key with a different type raises KeyError."""
        exp = self._make_experiment()
        exp._key_types["tag"] = "metadata"
        exp._metadata_values["tag"] = "old-value"

        with pytest.raises(KeyError, match="already used"):
            Experiment.__setitem__(exp, "tag", File("some/path"))

    def test_setitem_over_typed_series_raises(self):
        """Test that assigning static value to a typed series key raises KeyError."""
        exp = self._make_experiment()
        series = Series(exp, "loss")
        series._type = "metric"
        exp._series["loss"] = series

        with pytest.raises(KeyError, match="time series"):
            Experiment.__setitem__(exp, "loss", "oops")

    def test_setitem_over_untyped_series_ok(self):
        """Test that assigning static value to an untyped series key works."""
        exp = self._make_experiment()
        exp._series["maybe"] = Series(exp, "maybe")

        Experiment.__setitem__(exp, "maybe", "value")
        assert exp._key_types["maybe"] == "metadata"
        assert "maybe" not in exp._series

    def test_setitem_invalid_type(self):
        """Test that setting unsupported type raises TypeError."""
        exp = self._make_experiment()

        with pytest.raises(TypeError, match="Can only assign"):
            Experiment.__setitem__(exp, "bad", 42)  # type: ignore[arg-type]


class TestMetricsAndArtifactsProperties:
    """Test the metrics and artifacts convenience properties."""

    def _make_experiment(self) -> MagicMock:
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._series = {}
        exp._metadata_values = {}
        exp._static_files = {}
        return exp

    def test_metrics_property_returns_metric_series(self):
        """Test that .metrics returns only metric series."""
        exp = self._make_experiment()
        loss = Series(exp, "loss")
        loss._type = "metric"
        loss._values = [0.5, 0.3]
        acc = Series(exp, "acc")
        acc._type = "metric"
        acc._values = [0.8]
        files = Series(exp, "images")
        files._type = "file"

        exp._series = {"loss": loss, "acc": acc, "images": files}

        result = Experiment.metrics.fget(exp)
        assert set(result.keys()) == {"loss", "acc"}
        assert result["loss"] is loss
        assert result["acc"] is acc

    def test_metrics_property_empty(self):
        """Test .metrics with no metric series."""
        exp = self._make_experiment()
        result = Experiment.metrics.fget(exp)
        assert result == {}

    def test_artifacts_property_returns_files(self):
        """Test that .artifacts returns static files and file series."""
        exp = self._make_experiment()
        static_file = File("config.yaml")
        exp._static_files = {"config": static_file}

        file_series = Series(exp, "checkpoints")
        file_series._type = "file"
        metric_series = Series(exp, "loss")
        metric_series._type = "metric"
        exp._series = {"checkpoints": file_series, "loss": metric_series}

        result = Experiment.artifacts.fget(exp)
        assert set(result.keys()) == {"config", "checkpoints"}
        assert result["config"] is static_file
        assert result["checkpoints"] is file_series

    def test_artifacts_property_empty(self):
        """Test .artifacts with no artifacts."""
        exp = self._make_experiment()
        result = Experiment.artifacts.fget(exp)
        assert result == {}


class TestKeyUniqueness:
    """Test key uniqueness across metadata, metrics, and artifacts."""

    def test_metric_then_metadata_raises(self):
        """Test that using a metric key for metadata raises."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {"loss": "metric"}
        exp._series = {}
        exp._metadata_values = {}
        exp._static_files = {}

        with pytest.raises(KeyError, match="already used"):
            Experiment.__setitem__(exp, "loss", "some-metadata")

    def test_metadata_then_metric_raises(self):
        """Test that using a metadata key for metrics raises."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {"tag": "metadata"}
        exp._metadata_values = {"tag": "value"}

        with pytest.raises(KeyError, match="already used"):
            Experiment._register_key_type(exp, "tag", "metric")

    def test_same_type_registration_is_idempotent(self):
        """Test that registering the same key with the same type is fine."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {"loss": "metric"}

        # Should not raise
        Experiment._register_key_type(exp, "loss", "metric")


class TestLogMetricValue:
    """Test _log_metric_value internal method."""

    def test_log_metric_value_basic(self):
        """Test that _log_metric_value pushes to queue."""
        exp = MagicMock(spec=Experiment)
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp.store_step = True
        exp.store_created_at = False
        exp._metrics_queue = MagicMock()
        exp._stats = MagicMock()

        Experiment._log_metric_value(exp, "loss", 0.5)

        exp._metrics_queue.put.assert_called_once()
        batch = exp._metrics_queue.put.call_args[0][0]
        assert "loss" in batch
        assert batch["loss"].values[0].value == 0.5
        assert batch["loss"].values[0].step is None
        exp._stats.record_metric.assert_called_once_with("loss", 0.5)

    def test_log_metric_value_with_step(self):
        """Test that _log_metric_value passes step for legacy API."""
        exp = MagicMock(spec=Experiment)
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp.store_step = True
        exp.store_created_at = False
        exp._metrics_queue = MagicMock()
        exp._stats = MagicMock()

        Experiment._log_metric_value(exp, "loss", 0.5, step=10)

        batch = exp._metrics_queue.put.call_args[0][0]
        assert batch["loss"].values[0].step == 10

    def test_log_metric_value_raises_on_background_exception(self):
        """Test that _log_metric_value raises on background thread error."""
        exp = MagicMock(spec=Experiment)
        exp._manager = MagicMock()
        exp._manager.exception = RuntimeError("thread error")

        with pytest.raises(RuntimeError, match="thread error"):
            Experiment._log_metric_value(exp, "loss", 0.5)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestExperimentLegacyWithNewTracking:
    """Test that legacy methods properly populate series tracking."""

    def test_log_metrics_populates_series(self):
        """Test that log_metrics populates the _series for reading."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp.store_step = True
        exp.store_created_at = False
        exp._metrics_queue = MagicMock()
        exp._stats = MagicMock()
        exp._key_types = {}
        exp._series = {}

        # Wire up dict-like API that log_metrics now delegates to
        exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
        exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
        type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)

        Experiment.log_metrics(exp, {"loss": 0.5, "acc": 0.9}, step=1)

        # Verify series were populated
        assert "loss" in exp._series
        assert "acc" in exp._series
        assert exp._series["loss"]._values == [0.5]
        assert exp._series["acc"]._values == [0.9]
        assert exp._key_types["loss"] == "metric"
        assert exp._key_types["acc"] == "metric"

    def test_log_metrics_batch_populates_series(self):
        """Test that log_metrics_batch populates the _series for reading."""
        exp = MagicMock()
        exp._manager = MagicMock()
        exp._manager.exception = None
        exp._metrics_queue = MagicMock()
        exp.store_created_at = False
        exp._key_types = {}
        exp._series = {}

        exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
        exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
        exp.update = lambda data: Experiment.update(exp, data)

        metrics = {
            "loss": [{"step": 0, "value": 1.0}, {"step": 1, "value": 0.5}],
        }
        Experiment.log_metrics_batch(exp, metrics)

        assert exp._series["loss"]._values == [1.0, 0.5]
        assert exp._key_types["loss"] == "metric"


class TestFileDownloadBinding:
    """Test that experiment binds name and _download_fn on files."""

    @patch.object(experiment_module, "Artifact")
    def test_set_static_file_binds_name_and_download(self, mock_artifact_cls):
        """_set_static_file should set file.name and file._download_fn."""
        mock_artifact_cls.return_value = MagicMock()

        exp = MagicMock(spec=Experiment)
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._artifacts_api = MagicMock()
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0
        exp._create_download_fn = lambda key: Experiment._create_download_fn(exp, key)

        f = File("local.txt")
        assert f.name == ""
        assert f._download_fn is None

        Experiment._set_static_file(exp, "remote/key", f)

        assert f.name == "remote/key"
        assert f._download_fn is not None
        assert callable(f._download_fn)

    @patch.object(experiment_module, "Artifact")
    def test_log_file_series_binds_name_and_download(self, mock_artifact_cls):
        """_log_file_series_value should set file.name and file._download_fn."""
        mock_artifact_cls.return_value = MagicMock()

        exp = MagicMock(spec=Experiment)
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._artifacts_api = MagicMock()
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0
        exp._create_download_fn = lambda key: Experiment._create_download_fn(exp, key)

        f = File("frame.png")
        Experiment._log_file_series_value(exp, "images", f, 3)

        assert f.name == "images/3"
        assert f._download_fn is not None

    @patch.object(experiment_module, "Artifact")
    def test_file_series_entry_is_downloadable(self, mock_artifact_cls):
        """Files in a series should be downloadable via .save() after upload."""
        mock_artifact = MagicMock()
        mock_artifact.get.return_value = "/downloaded/path"
        mock_artifact_cls.return_value = mock_artifact

        exp = MagicMock(spec=Experiment)
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._artifacts_api = MagicMock()
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0
        exp._create_download_fn = lambda key: Experiment._create_download_fn(exp, key)

        f = File("frame.png")
        Experiment._log_file_series_value(exp, "frames", f, 0)

        result = f.save("/output/frame.png")
        assert result == "/downloaded/path"

    def test_rebuild_state_binds_name_and_download(self):
        """_rebuild_state should set name and _download_fn on rebuilt files."""
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_trackers_from_metrics_store.return_value = []

        # Add a fake artifact
        art = MagicMock()
        art.path = "data/results.csv"
        exp._metrics_store.artifacts = [art]
        exp._create_download_fn = lambda key: lambda path: f"downloaded:{key}:{path}"

        Experiment._rebuild_state(exp)

        f = exp._static_files["data/results.csv"]
        assert f.name == "data/results.csv"
        assert f._download_fn is not None
        assert f.save("/tmp/out.csv") == "downloaded:data/results.csv:/tmp/out.csv"
