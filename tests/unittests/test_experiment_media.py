# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for adding and retrieving files/media via the experiment dict-like API."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from litlogger.experiment import Experiment
from litlogger.media import File, Image, Text
from litlogger.series import Series

experiment_module = sys.modules["litlogger.experiment"]


def _make_exp(**overrides):
    """Create a MagicMock wired for the dict-like experiment API."""
    exp = MagicMock(spec=Experiment)
    exp._series = {}
    exp._key_types = {}
    exp._metadata_values = {}
    exp._static_files = {}
    exp._manager = MagicMock()
    exp._manager.exception = None
    exp.store_step = True
    exp.store_created_at = False
    exp._metrics_queue = MagicMock()
    exp._stats = MagicMock()
    exp._stats.artifacts_logged = 0

    type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)
    type(exp).__setitem__ = lambda self, key, value: Experiment.__setitem__(self, key, value)

    exp.update = lambda data: Experiment.update(exp, data)
    exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
    exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
    exp._log_metric_value = lambda key, value, step=None: Experiment._log_metric_value(exp, key, value, step=step)
    exp._log_file_series_value = MagicMock()
    exp._set_static_file = MagicMock()

    for k, v in overrides.items():
        setattr(exp, k, v)
    return exp


# ---------------------------------------------------------------------------
# Adding static files
# ---------------------------------------------------------------------------


class TestAddStaticFile:
    """Test experiment['key'] = File(path) for static files."""

    def test_setitem_file(self):
        exp = _make_exp()
        f = File("data.csv")
        exp["dataset"] = f

        assert exp._key_types["dataset"] == "static_file"
        assert exp._static_files["dataset"] is f
        exp._set_static_file.assert_called_once_with("dataset", f)

    def test_setitem_image(self):
        exp = _make_exp()
        img = Image("photo.png")
        exp["photo"] = img

        assert exp._key_types["photo"] == "static_file"
        assert exp._static_files["photo"] is img
        exp._set_static_file.assert_called_once_with("photo", img)

    def test_setitem_text(self):
        exp = _make_exp()
        t = Text("hello world")
        exp["notes"] = t

        assert exp._key_types["notes"] == "static_file"
        assert exp._static_files["notes"] is t
        exp._set_static_file.assert_called_once_with("notes", t)

    def test_overwrite_same_type(self):
        """Overwriting a static_file key with another File is allowed."""
        exp = _make_exp()
        exp["config"] = File("v1.yaml")
        exp["config"] = File("v2.yaml")

        assert exp._static_files["config"].path == "v2.yaml"
        assert exp._set_static_file.call_count == 2

    def test_update_with_file(self):
        exp = _make_exp()
        exp.update({"config": File("config.yaml")})

        assert exp._key_types["config"] == "static_file"


class TestAddStaticFileBindings:
    """Test that _set_static_file binds name and _download_fn."""

    @patch.object(experiment_module, "Artifact")
    def test_binds_name(self, mock_artifact_cls):
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
        Experiment._set_static_file(exp, "remote/key", f)

        assert f.name == "remote/key"

    @patch.object(experiment_module, "Artifact")
    def test_binds_download_fn(self, mock_artifact_cls):
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
        Experiment._set_static_file(exp, "remote/key", f)

        assert f._download_fn is not None
        assert callable(f._download_fn)


# ---------------------------------------------------------------------------
# Adding file series
# ---------------------------------------------------------------------------


class TestAddFileSeries:
    """Test experiment['key'].append(File(...)) for file time series."""

    def test_append_file_to_series(self):
        exp = _make_exp()
        f = File("frame_0.png")
        exp["frames"].append(f)

        assert len(exp["frames"]) == 1
        assert exp["frames"][0] is f
        assert exp._key_types["frames"] == "file_series"
        exp._log_file_series_value.assert_called_once_with("frames", f, 0)

    def test_append_multiple_files(self):
        exp = _make_exp()
        f0 = File("frame_0.png")
        f1 = File("frame_1.png")
        f2 = File("frame_2.png")
        exp["frames"].append(f0)
        exp["frames"].append(f1)
        exp["frames"].append(f2)

        assert len(exp["frames"]) == 3
        calls = exp._log_file_series_value.call_args_list
        assert calls[0][0] == ("frames", f0, 0)
        assert calls[1][0] == ("frames", f1, 1)
        assert calls[2][0] == ("frames", f2, 2)

    def test_extend_files(self):
        exp = _make_exp()
        files = [File(f"img_{i}.png") for i in range(3)]
        exp["images"].extend(files)

        assert len(exp["images"]) == 3
        assert exp._log_file_series_value.call_count == 3

    def test_append_image_to_series(self):
        exp = _make_exp()
        img = Image("photo.png")
        exp["photos"].append(img)

        assert len(exp["photos"]) == 1
        assert exp._key_types["photos"] == "file_series"

    def test_append_text_to_series(self):
        exp = _make_exp()
        t = Text("log entry 1")
        exp["logs"].append(t)

        assert len(exp["logs"]) == 1
        assert exp._key_types["logs"] == "file_series"


class TestFileSeriesBindings:
    """Test that _log_file_series_value binds name and _download_fn."""

    @patch.object(experiment_module, "Artifact")
    def test_binds_name_with_index(self, mock_artifact_cls):
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
        Experiment._log_file_series_value(exp, "images", f, 5)

        assert f.name == "images/5"

    @patch.object(experiment_module, "Artifact")
    def test_binds_download_fn(self, mock_artifact_cls):
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
        Experiment._log_file_series_value(exp, "images", f, 0)

        assert f._download_fn is not None

    @patch.object(experiment_module, "Artifact")
    def test_file_series_entry_saveable(self, mock_artifact_cls):
        """Individual files in a series can be downloaded via .save()."""
        mock_artifact = MagicMock()
        mock_artifact.get.return_value = "/downloaded"
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
        assert result == "/downloaded"


# ---------------------------------------------------------------------------
# Retrieving files
# ---------------------------------------------------------------------------


class TestRetrieveStaticFile:
    """Test experiment['key'] retrieval for static files."""

    def test_getitem_returns_file(self):
        exp = _make_exp()
        f = File("data.csv")
        exp["dataset"] = f

        result = exp["dataset"]
        assert result is f
        assert isinstance(result, File)

    def test_getitem_returns_image(self):
        exp = _make_exp()
        img = Image("photo.png")
        exp["photo"] = img

        assert exp["photo"] is img
        assert isinstance(exp["photo"], Image)

    def test_getitem_returns_text(self):
        exp = _make_exp()
        t = Text("notes content")
        exp["notes"] = t

        assert exp["notes"] is t

    def test_save_without_upload_raises(self):
        """File.save() fails if not yet uploaded (no _download_fn)."""
        f = File("local.txt")
        with pytest.raises(RuntimeError, match="no remote context"):
            f.save("/output/file.txt")


class TestRetrieveFileSeries:
    """Test experiment['key'] retrieval for file time series."""

    def test_getitem_returns_series(self):
        exp = _make_exp()
        exp["frames"].append(File("f0.png"))
        exp["frames"].append(File("f1.png"))

        result = exp["frames"]
        assert isinstance(result, Series)
        assert len(result) == 2

    def test_series_indexing(self):
        exp = _make_exp()
        f0 = File("f0.png")
        f1 = File("f1.png")
        exp["frames"].append(f0)
        exp["frames"].append(f1)

        assert exp["frames"][0] is f0
        assert exp["frames"][1] is f1
        assert exp["frames"][-1] is f1

    def test_series_iteration(self):
        exp = _make_exp()
        files = [File(f"f{i}.png") for i in range(3)]
        for f in files:
            exp["frames"].append(f)

        assert list(exp["frames"]) == files


class TestRetrieveArtifactsProperty:
    """Test experiment.artifacts property."""

    def test_includes_static_files(self):
        exp = _make_exp()
        f = File("data.csv")
        exp["dataset"] = f

        result = Experiment.artifacts.fget(exp)
        assert "dataset" in result
        assert result["dataset"] is f

    def test_includes_file_series(self):
        exp = _make_exp()
        exp["frames"].append(File("f0.png"))

        result = Experiment.artifacts.fget(exp)
        assert "frames" in result
        assert isinstance(result["frames"], Series)

    def test_excludes_metrics(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        result = Experiment.artifacts.fget(exp)
        assert "loss" not in result

    def test_excludes_metadata(self):
        exp = _make_exp()
        exp._set_metadata_value = MagicMock()
        exp["tag"] = "v1"

        result = Experiment.artifacts.fget(exp)
        assert "tag" not in result

    def test_empty(self):
        exp = _make_exp()
        result = Experiment.artifacts.fget(exp)
        assert result == {}


# ---------------------------------------------------------------------------
# Rebuild state (download from resumed experiment)
# ---------------------------------------------------------------------------


class TestRebuildStateFiles:
    """Test that _rebuild_state populates downloadable files."""

    def test_rebuilds_artifacts_with_name_and_download(self):
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_trackers_from_metrics_store.return_value = []

        art = MagicMock()
        art.path = "results.csv"
        exp._metrics_store.artifacts = [art]
        exp._create_download_fn = lambda key: lambda path: f"dl:{key}:{path}"

        Experiment._rebuild_state(exp)

        f = exp._static_files["results.csv"]
        assert isinstance(f, File)
        assert f.name == "results.csv"
        assert f._download_fn is not None
        assert f.save("/tmp/out.csv") == "dl:results.csv:/tmp/out.csv"

    def test_rebuild_does_not_overwrite_existing_keys(self):
        exp = MagicMock(spec=Experiment)
        exp._key_types = {"existing": "metric"}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_trackers_from_metrics_store.return_value = []

        art = MagicMock()
        art.path = "existing"
        exp._metrics_store.artifacts = [art]
        exp._create_download_fn = lambda key: lambda path: path

        Experiment._rebuild_state(exp)

        # Should not overwrite the existing metric key
        assert exp._key_types["existing"] == "metric"
        assert "existing" not in exp._static_files


# ---------------------------------------------------------------------------
# Type conflicts
# ---------------------------------------------------------------------------


class TestMediaTypeConflicts:
    """Test that file keys conflict properly with other types."""

    def test_cannot_assign_string_to_file_key(self):
        exp = _make_exp()
        exp["config"] = File("config.yaml")

        with pytest.raises(KeyError, match="already used"):
            exp["config"] = "oops"

    def test_cannot_assign_file_to_metadata_key(self):
        exp = _make_exp()
        exp._set_metadata_value = MagicMock()
        exp["tag"] = "v1"

        with pytest.raises(KeyError, match="already used"):
            exp["tag"] = File("data.csv")

    def test_cannot_assign_file_to_metric_key(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        with pytest.raises(KeyError, match="time series"):
            exp["loss"] = File("data.csv")

    def test_cannot_append_metric_to_file_series(self):
        exp = _make_exp()
        exp["frames"].append(File("f0.png"))

        with pytest.raises(TypeError, match="file series"):
            exp["frames"].append(0.5)
