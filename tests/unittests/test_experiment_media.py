# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for adding and retrieving files/media via the experiment dict-like API."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from lightning_sdk.lightning_cloud.openapi import V1MediaType
from litlogger.experiment import Experiment
from litlogger.media import File, Image, Model, Text
from litlogger.series import Series


def _make_exp(**overrides):
    """Create a MagicMock wired for the dict-like experiment API."""
    exp = MagicMock(spec=Experiment)
    exp.name = "exp"
    exp._series = {}
    exp._key_types = {}
    exp._metadata_values = {}
    exp._static_files = {}
    exp._model_lookup_cache = {}
    exp._missing_model_keys = set()
    exp._manager = MagicMock()
    exp._manager.exception = None
    exp.store_step = True
    exp.store_created_at = False
    exp._metrics_queue = MagicMock()
    exp._media_api = MagicMock()
    exp._teamspace = MagicMock()
    exp._teamspace.name = "teamspace"
    exp._teamspace.owner.name = "owner"
    exp._teamspace.list_models.return_value = []
    exp._teamspace.list_model_versions.return_value = []
    exp._stats = MagicMock()
    exp._stats.artifacts_logged = 0
    exp._stats.media_logged = 0
    exp._stats.models_logged = 0

    type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)
    type(exp).__setitem__ = lambda self, key, value: Experiment.__setitem__(self, key, value)

    exp.update = lambda data: Experiment.update(exp, data)
    exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
    exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
    exp._log_metric_value = lambda key, value, step=None: Experiment._log_metric_value(exp, key, value, step=step)
    exp._resolve_remote_model = lambda key: Experiment._resolve_remote_model(exp, key)
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

    def test_binds_name(self):
        exp = MagicMock(spec=Experiment)
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.cluster_id = "acc-1"
        exp._artifacts_api = MagicMock()
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0

        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            f = File(tmp.name)
            Experiment._set_static_file(exp, "remote/key", f)

            assert f.name == "remote/key"

    def test_binds_download_fn(self):
        exp = MagicMock(spec=Experiment)
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.cluster_id = "acc-1"
        exp._artifacts_api = MagicMock()
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0

        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            f = File(tmp.name)
            Experiment._set_static_file(exp, "remote/key", f)

            assert f._download_fn is not None
            assert callable(f._download_fn)

    def test_non_file_media_uses_media_api(self):
        exp = MagicMock(spec=Experiment)
        exp._media_api = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0
        exp._media_type_to_v1 = lambda media_type: Experiment._media_type_to_v1(exp, media_type)
        exp._upload_media = (
            lambda name, file_path, media_type, step=None, epoch=None, caption=None: Experiment._upload_media(
                exp,
                name,
                file_path,
                media_type,
                step=step,
                epoch=epoch,
                caption=caption,
            )
        )
        exp._upload_media_value = (
            lambda key, value, name=None, step=None, epoch=None, caption=None: Experiment._upload_media_value(
                exp,
                key,
                value,
                name=name,
                step=step,
                epoch=epoch,
                caption=caption,
            )
        )

        image = Image("local.png")
        Experiment._set_static_file(exp, "photo", image)

        exp._media_api.upload_media.assert_called_once()
        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["name"] == "photo"
        assert kwargs["file_path"] == "local.png"
        assert kwargs["media_type"] == V1MediaType.IMAGE
        assert exp._stats.media_logged == 1

    @patch.object(Model, "_log_model", return_value="owner/team/exp-model:latest")
    def test_model_artifact_uses_litmodels(self, mock_log_model):
        exp = Experiment.__new__(Experiment)
        exp.name = "exp"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.cluster_id = "acc-1"
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        exp._update_metrics_store = MagicMock()
        exp._stats = MagicMock()
        exp._stats.models_logged = 0

        model = Model("model.ckpt")
        Experiment._set_static_file(exp, "checkpoint", model)

        mock_log_model.assert_called_once_with(
            experiment_name="exp",
            teamspace=exp._teamspace,
            experiment=exp,
            cloud_account="acc-1",
        )
        assert model._model_name == "owner/team/exp-model:latest"
        assert model._download_fn is not None
        assert exp._stats.models_logged == 1

    @patch.object(Model, "_log_model", return_value="owner/team/exp-model-object:latest")
    def test_model_object_uses_litmodels(self, mock_log_model):
        exp = Experiment.__new__(Experiment)
        exp.name = "exp"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.cluster_id = "acc-1"
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        exp._update_metrics_store = MagicMock()
        exp._stats = MagicMock()
        exp._stats.models_logged = 0

        model = Model(object())
        Experiment._set_static_file(exp, "model-object", model)

        mock_log_model.assert_called_once_with(
            experiment_name="exp",
            teamspace=exp._teamspace,
            experiment=exp,
            cloud_account="acc-1",
        )
        assert model._model_name == "owner/team/exp-model-object:latest"
        assert model._load_fn is not None
        assert exp._stats.models_logged == 1


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
        exp._log_file_series_value.assert_called_once_with("frames", f, 0, step=0)

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
        assert calls[0] == (("frames", f0, 0), {"step": 0})
        assert calls[1] == (("frames", f1, 1), {"step": 1})
        assert calls[2] == (("frames", f2, 2), {"step": 2})

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

    def test_binds_name_with_index(self):
        exp = MagicMock(spec=Experiment)
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.cluster_id = "acc-1"
        exp._artifacts_api = MagicMock()
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            f = File(tmp.name)
            Experiment._log_file_series_value(exp, "images", f, 5)

            assert f.name == "images/5"

    def test_binds_download_fn(self):
        exp = MagicMock(spec=Experiment)
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.cluster_id = "acc-1"
        exp._artifacts_api = MagicMock()
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            f = File(tmp.name)
            Experiment._log_file_series_value(exp, "images", f, 0)

            assert f._download_fn is not None

    def test_file_series_entry_saveable(self):
        """Individual files in a series can be downloaded via .save()."""
        exp = MagicMock(spec=Experiment)
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.cluster_id = "acc-1"
        exp._artifacts_api = MagicMock()
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp, tempfile.TemporaryDirectory() as tmpdir:
            f = File(tmp.name)
            Experiment._log_file_series_value(exp, "frames", f, 0)

            download_path = os.path.join(tmpdir, "frame.png")
            result = f.save(download_path)
            assert result == download_path

    def test_non_file_series_uses_media_api(self):
        exp = MagicMock(spec=Experiment)
        exp._media_api = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0
        exp._media_type_to_v1 = lambda media_type: Experiment._media_type_to_v1(exp, media_type)
        exp._upload_media = (
            lambda name, file_path, media_type, step=None, epoch=None, caption=None: Experiment._upload_media(
                exp,
                name,
                file_path,
                media_type,
                step=step,
                epoch=epoch,
                caption=caption,
            )
        )
        exp._upload_media_value = (
            lambda key, value, name=None, step=None, epoch=None, caption=None: Experiment._upload_media_value(
                exp,
                key,
                value,
                name=name,
                step=step,
                epoch=epoch,
                caption=caption,
            )
        )

        text = Text("hello world")
        Experiment._log_file_series_value(exp, "logs", text, 2, step=7)

        exp._media_api.upload_media.assert_called_once()
        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["name"] == "logs"
        assert kwargs["step"] == 7
        assert kwargs["media_type"] == V1MediaType.TEXT
        assert exp._stats.media_logged == 1

    @patch.object(Model, "_log_model", return_value="owner/team/exp-model-series:latest")
    def test_model_series_uses_series_key_for_remote_binding(self, mock_log_model):
        exp = Experiment.__new__(Experiment)
        exp.name = "exp"
        exp._teamspace = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.cluster_id = "acc-1"
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        exp._update_metrics_store = MagicMock()
        exp._stats = MagicMock()
        exp._stats.models_logged = 0

        model = Model("checkpoint.ckpt")
        Experiment._log_file_series_value(exp, "models", model, 2)

        mock_log_model.assert_called_once_with(
            experiment_name="exp",
            teamspace=exp._teamspace,
            experiment=exp,
            cloud_account="acc-1",
        )
        assert model.version == "v3"
        assert model.name == "models"
        assert exp._stats.models_logged == 1


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


class TestRetrieveRemoteModels:
    """Test lazy model lookup when a key is missing from rebuilt state."""

    def test_getitem_resolves_remote_model_from_teamspace_listing(self):
        exp = _make_exp()
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._teamspace.name = "teamspace"
        exp._teamspace.owner.name = "owner"

        model_info = MagicMock()
        model_info.name = "models-latest"
        exp._teamspace.list_models.return_value = [model_info]

        version_info = MagicMock()
        version_info.version = "new-artifact-v1"
        version_info.upload_complete = True
        version_info.metadata = {"litModels": "1.0.0"}
        exp._teamspace.list_model_versions.return_value = [version_info]

        result = exp["models/latest"]

        assert isinstance(result, Model)
        assert result._model_kind == "artifact"
        assert result._model_name == "owner/teamspace/models-latest:new-artifact-v1"
        assert exp._key_types["models/latest"] == "static_file"
        assert exp._static_files["models/latest"] is result

    def test_getitem_resolves_remote_model_series_from_multiple_versions(self):
        exp = _make_exp()
        exp.name = "exp1"
        exp._teamspace = MagicMock()
        exp._teamspace.name = "teamspace"
        exp._teamspace.owner.name = "owner"

        model_info = MagicMock()
        model_info.name = "checkpoints"
        exp._teamspace.list_models.return_value = [model_info]

        version0 = MagicMock()
        version0.version = "new-step-0"
        version0.upload_complete = True
        version0.metadata = {"litModels": "1.0.0"}

        version1 = MagicMock()
        version1.version = "new-step-1"
        version1.upload_complete = True
        version1.metadata = {"litModels": "1.0.0"}

        exp._teamspace.list_model_versions.return_value = [version1, version0]

        result = exp["checkpoints"]

        assert isinstance(result, Series)
        assert result._type == "file"
        assert len(result) == 2
        assert isinstance(result[0], Model)
        assert isinstance(result[1], Model)
        assert result[0]._model_name == "owner/teamspace/checkpoints:new-step-0"
        assert result[1]._model_name == "owner/teamspace/checkpoints:new-step-1"
        assert exp._key_types["checkpoints"] == "file_series"
        assert exp._series["checkpoints"] is result


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
        exp._update_metrics_store = MagicMock()
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        exp._resumed_steps = {}

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

    def test_rebuild_loads_artifacts_from_logger_artifacts_api(self):
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._metrics_api = MagicMock()
        art = MagicMock()
        art.path = "results.csv"
        exp._metrics_api.client.lit_logger_service_list_logger_artifacts.return_value.logger_artifacts = [art]
        exp._update_metrics_store = MagicMock()
        exp._create_download_fn = lambda key: lambda path: f"dl:{key}:{path}"
        exp._resumed_steps = {}

        Experiment._rebuild_state(exp)

        assert "results.csv" in exp._static_files
        exp._update_metrics_store.assert_called_once()

    def test_rebuilds_artifact_series_from_logger_artifacts_api(self):
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._metrics_api = MagicMock()
        art0 = MagicMock()
        art0.path = "reports/0"
        art1 = MagicMock()
        art1.path = "reports/1"
        exp._metrics_api.client.lit_logger_service_list_logger_artifacts.return_value.logger_artifacts = [art1, art0]
        exp._update_metrics_store = MagicMock()
        exp._create_download_fn = lambda key: lambda path: f"dl:{key}:{path}"
        exp._resumed_steps = {}

        Experiment._rebuild_state(exp)

        assert exp._key_types["reports"] == "file_series"
        assert isinstance(exp._series["reports"], Series)
        assert [item.name for item in exp._series["reports"]] == ["reports/0", "reports/1"]

    def test_rebuild_does_not_overwrite_existing_keys(self):
        exp = MagicMock(spec=Experiment)
        exp._key_types = {"existing": "metric"}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._update_metrics_store = MagicMock()
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        exp._resumed_steps = {}

        art = MagicMock()
        art.path = "existing"
        exp._metrics_store.artifacts = [art]
        exp._create_download_fn = lambda key: lambda path: path

        Experiment._rebuild_state(exp)

        # Should not overwrite the existing metric key
        assert exp._key_types["existing"] == "metric"
        assert "existing" not in exp._static_files

    def test_rebuilds_static_media_with_wrapper(self):
        media = MagicMock()
        media.name = "preview"
        media.storage_path = "media/preview.png"
        media.cluster_id = "cloud-1"
        media.media_type = V1MediaType.IMAGE
        media.id = "media-1"

        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._update_metrics_store = MagicMock()
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_api = MagicMock()
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._media_api = MagicMock()
        exp._media_api.client.lit_logger_service_list_lit_logger_media.return_value.media = [media]
        exp._wrap_media_file = lambda media_name, media_type: Experiment._wrap_media_file(exp, media_name, media_type)
        exp._create_media_download_fn = lambda storage_path, cloud_account=None: Experiment._create_media_download_fn(
            exp, storage_path, cloud_account
        )
        exp._resumed_steps = {}

        Experiment._rebuild_state(exp)

        wrapped = exp._static_files["preview"]
        assert isinstance(wrapped, Image)
        assert wrapped.name == "preview"
        assert wrapped._download_fn is not None

    def test_rebuilds_media_series_with_wrapper(self):
        media0 = MagicMock()
        media0.name = "logs/0"
        media0.storage_path = "media/logs-0.txt"
        media0.cluster_id = "cloud-1"
        media0.media_type = V1MediaType.TEXT
        media0.id = "media-0"

        media1 = MagicMock()
        media1.name = "logs/1"
        media1.storage_path = "media/logs-1.txt"
        media1.cluster_id = "cloud-1"
        media1.media_type = V1MediaType.TEXT
        media1.id = "media-1"

        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._update_metrics_store = MagicMock()
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_api = MagicMock()
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._media_api = MagicMock()
        exp._media_api.client.lit_logger_service_list_lit_logger_media.return_value.media = [media1, media0]
        exp._wrap_media_file = lambda media_name, media_type: Experiment._wrap_media_file(exp, media_name, media_type)
        exp._create_media_download_fn = lambda storage_path, cloud_account=None: Experiment._create_media_download_fn(
            exp, storage_path, cloud_account
        )
        exp._resumed_steps = {}

        Experiment._rebuild_state(exp)

        assert exp._key_types["logs"] == "file_series"
        assert isinstance(exp._series["logs"], Series)
        assert len(exp._series["logs"]) == 2
        assert all(isinstance(item, Text) for item in exp._series["logs"])

    def test_rebuilds_same_name_media_series_with_wrapper(self):
        media0 = MagicMock()
        media0.name = "logs"
        media0.step = 0
        media0.storage_path = "media/logs-0.txt"
        media0.cluster_id = "cloud-1"
        media0.media_type = V1MediaType.TEXT
        media0.id = "media-0"

        media1 = MagicMock()
        media1.name = "logs"
        media1.step = 1
        media1.storage_path = "media/logs-1.txt"
        media1.cluster_id = "cloud-1"
        media1.media_type = V1MediaType.TEXT
        media1.id = "media-1"

        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._update_metrics_store = MagicMock()
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_api = MagicMock()
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._media_api = MagicMock()
        exp._media_api.client.lit_logger_service_list_lit_logger_media.return_value.media = [media1, media0]
        exp._wrap_media_file = lambda media_name, media_type: Experiment._wrap_media_file(exp, media_name, media_type)
        exp._create_media_download_fn = lambda storage_path, cloud_account=None: Experiment._create_media_download_fn(
            exp, storage_path, cloud_account
        )
        exp._resumed_steps = {}

        Experiment._rebuild_state(exp)

        assert exp._key_types["logs"] == "file_series"
        assert isinstance(exp._series["logs"], Series)
        assert len(exp._series["logs"]) == 2
        assert [item.path for item in exp._series["logs"]] == ["logs", "logs"]


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
