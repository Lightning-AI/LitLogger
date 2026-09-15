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
from litlogger.primitives import File, Image, Model, PrimitiveWrite, Text, Video, WritePlacement
from litlogger.primitives._utils import series_storage_name, static_storage_name
from litlogger.series import Series
from litlogger.session import ExperimentSession


def _session_of(exp):
    """Build a session view over the experiment's current mock infrastructure."""

    def part(name):
        value = getattr(exp, name, None)
        return value if value is not None else MagicMock()

    metrics_api = part("_metrics_api")
    return ExperimentSession._from_components(
        name=getattr(exp, "name", "exp"),
        metrics_api=metrics_api,
        media_api=part("_media_api"),
        artifacts_api=part("_artifacts_api"),
        teamspace=part("_teamspace"),
        metrics_store=part("_metrics_store"),
        experiment=exp,
        queue_=part("_metrics_queue"),
        stats=part("_stats"),
        printer=part("_printer"),
        store_step=bool(getattr(exp, "store_step", True)),
        store_created_at=bool(getattr(exp, "store_created_at", False)),
        last_x=getattr(exp, "_resumed_steps", None) or {},
        background=getattr(exp, "_manager", None),
    )


def _make_exp(**overrides):
    """Create a MagicMock wired for the dict-like experiment API."""
    exp = MagicMock(spec=Experiment)
    exp.name = "exp"
    exp._series = {}
    exp._key_types = {}
    exp._metadata_values = {}
    exp._static_files = {}
    exp._manager = MagicMock()
    exp._manager.exception = None
    exp.store_step = True
    exp.store_created_at = False
    exp._metrics_queue = MagicMock()
    # The dict API queues writes; execute them inline the way the worker would.
    exp._metrics_queue.put.side_effect = lambda item: (
        item.execute(exp._session) if isinstance(item, PrimitiveWrite) else None
    )
    exp._media_api = MagicMock()
    exp._teamspace = MagicMock()
    exp._teamspace.name = "teamspace"
    exp._teamspace.owner.name = "owner"
    exp._teamspace.list_models.return_value = []
    exp._teamspace.list_model_versions.return_value = []
    exp._stats = MagicMock()
    exp._metrics_api = MagicMock()
    exp._media_api = MagicMock()
    exp._artifacts_api = MagicMock()
    exp._teamspace = MagicMock()
    exp._metrics_store = MagicMock()
    exp._metrics_store.id = "store-1"
    exp._metrics_store.name = "exp"
    exp._metrics_store.tags = []
    exp._metrics_store.cluster_id = "acc-1"
    # Metadata writes re-read the store from the API; keep the seeded one.
    exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store
    # Live session view so tests can reseed infrastructure after the factory.
    type(exp)._session = property(lambda self: _session_of(self))
    exp._stats.artifacts_logged = 0
    exp._stats.media_logged = 0
    exp._stats.models_logged = 0

    type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)
    type(exp).__setitem__ = lambda self, key, value: Experiment.__setitem__(self, key, value)

    exp.update = lambda data: Experiment.update(exp, data)
    exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
    exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
    exp._log_metric_value = lambda key, y, x=None: Experiment._log_metric_value(exp, key, y, x=x)
    exp._validate_file_primitive = lambda value: Experiment._validate_file_primitive(exp, value)
    exp.resolve_model = lambda key: Experiment.resolve_model(exp, key)
    exp._log_file_series_value = MagicMock()

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
        assert f.name == "dataset"
        exp._artifacts_api.upload_experiment_file_artifact.assert_called_once()

    def test_setitem_image(self):
        exp = _make_exp()
        img = Image("photo.png")
        exp["photo"] = img

        assert exp._key_types["photo"] == "static_file"
        assert exp._static_files["photo"] is img
        assert img.name == "photo"
        exp._media_api.upload_media.assert_called_once()

    def test_setitem_text(self):
        exp = _make_exp()
        t = Text("hello world")
        exp["notes"] = t

        assert exp._key_types["notes"] == "static_file"
        assert exp._static_files["notes"] is t
        assert t.name == "notes"
        exp._media_api.upload_media.assert_called_once()

    def test_setitem_video(self):
        exp = _make_exp()
        video = Video("preview.mp4")
        exp["preview"] = video

        assert exp._key_types["preview"] == "static_file"
        assert exp._static_files["preview"] is video
        assert video.name == "preview"
        exp._media_api.upload_media.assert_called_once()

    def test_overwrite_same_type(self):
        """Overwriting a static_file key with another File is allowed."""
        exp = _make_exp()
        exp["config"] = File("v1.yaml")
        exp["config"] = File("v2.yaml")

        assert exp._static_files["config"].path == "v2.yaml"
        assert exp._artifacts_api.upload_experiment_file_artifact.call_count == 2

    def test_update_with_file(self):
        exp = _make_exp()
        exp.update({"config": File("config.yaml")})

        assert exp._key_types["config"] == "static_file"

    def test_queue_failure_does_not_mutate_static_file_state(self):
        exp = _make_exp()
        file = File("data.csv")
        exp._metrics_queue.put.side_effect = RuntimeError("queue failed")

        with pytest.raises(RuntimeError, match="queue failed"):
            exp["dataset"] = file

        assert "dataset" not in exp._key_types
        assert "dataset" not in exp._static_files
        assert file.name == ""
        assert file._read_barrier is None


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
            f.log(_session_of(exp), WritePlacement("remote/key"))

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
            f.log(_session_of(exp), WritePlacement("remote/key"))

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

        image = Image("local.png")
        image.log(_session_of(exp), WritePlacement("photo"))

        exp._media_api.upload_media.assert_called_once()
        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["name"] == "photo"
        assert kwargs["file_path"] == "local.png"
        assert kwargs["media_type"] == V1MediaType.IMAGE
        assert exp._stats.media_logged == 1

    def test_video_media_uses_media_api(self):
        exp = MagicMock(spec=Experiment)
        exp._media_api = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        video = Video("preview.mp4")
        video.log(_session_of(exp), WritePlacement("preview"))

        exp._media_api.upload_media.assert_called_once()
        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["name"] == "preview"
        assert kwargs["file_path"] == "preview.mp4"
        assert kwargs["media_type"] == V1MediaType.VIDEO
        assert exp._stats.media_logged == 1

    def test_numeric_static_media_key_uses_explicit_storage_name(self):
        exp = MagicMock(spec=Experiment)
        exp._media_api = MagicMock()
        exp._metrics_store = MagicMock(id="store-1")
        exp._teamspace = MagicMock()
        exp._stats = MagicMock(media_logged=0)
        image = Image("local.png")

        image.log(_session_of(exp), WritePlacement("reports/2024"))

        assert exp._media_api.upload_media.call_args.kwargs["name"] == static_storage_name("reports/2024")

    @patch.object(Model, "_log_model", return_value="owner/team/exp-model:latest")
    def test_model_artifact_uses_litmodels(self, mock_log_model):
        exp = MagicMock()
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
        model.log(_session_of(exp), WritePlacement("checkpoint"))

        mock_log_model.assert_called_once_with(
            experiment_name="exp",
            teamspace=exp._teamspace,
            key="checkpoint",
            experiment=exp,
            cloud_account="acc-1",
        )
        assert model._model_name == "owner/team/exp-model:latest"
        assert model._download_fn is not None
        assert exp._stats.models_logged == 1

    @patch.object(Model, "_log_model", return_value="owner/team/exp-model-object:latest")
    def test_model_object_uses_litmodels(self, mock_log_model):
        exp = MagicMock()
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
        model.log(_session_of(exp), WritePlacement("model-object"))

        mock_log_model.assert_called_once_with(
            experiment_name="exp",
            teamspace=exp._teamspace,
            key="model-object",
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

    def test_append_video_to_series(self):
        exp = _make_exp()
        video = Video("preview.mp4")
        exp["clips"].append(video)

        assert len(exp["clips"]) == 1
        assert exp._key_types["clips"] == "file_series"

    def test_queue_failure_does_not_mutate_file_series(self):
        exp = _make_exp()
        exp._log_file_series_value = lambda key, value, index, step=None: Experiment._log_file_series_value(
            exp, key, value, index, step
        )
        series = exp["frames"]
        file = File("frame.png")
        exp._metrics_queue.put.side_effect = RuntimeError("queue failed")

        with pytest.raises(RuntimeError, match="queue failed"):
            series.append(file)

        assert list(series) == []
        assert series._type is None
        assert "frames" not in exp._key_types
        assert file.name == ""
        assert file._read_barrier is None


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
            f.log(_session_of(exp), WritePlacement("images", index=5))

            assert f.name == "images"

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
            f.log(_session_of(exp), WritePlacement("images", index=0))

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
        exp._artifacts_api.download_file.side_effect = lambda teamspace, remote_path, local_path, cloud_account=None: (
            local_path
        )
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp, tempfile.TemporaryDirectory() as tmpdir:
            f = File(tmp.name)
            f.log(_session_of(exp), WritePlacement("frames", index=0))

            download_path = os.path.join(tmpdir, "frame.png")
            result = f.save(download_path)
            assert result == download_path
            kwargs = exp._artifacts_api.download_file.call_args.kwargs
            assert kwargs["remote_path"] == "experiments/exp1/.litlogger/series/ZnJhbWVz/0"

    def test_non_file_series_uses_media_api(self):
        exp = MagicMock(spec=Experiment)
        exp._media_api = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        text = Text("hello world")
        text.log(_session_of(exp), WritePlacement("logs", index=2, x=7))

        exp._media_api.upload_media.assert_called_once()
        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["name"] == series_storage_name("logs")
        assert kwargs["step"] == 7
        assert kwargs["media_type"] == V1MediaType.TEXT
        assert exp._stats.media_logged == 1

    def test_video_series_uses_media_api(self):
        exp = MagicMock(spec=Experiment)
        exp._media_api = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._teamspace = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        video = Video("preview.mp4")
        video.log(_session_of(exp), WritePlacement("clips", index=2, x=7))

        exp._media_api.upload_media.assert_called_once()
        _, kwargs = exp._media_api.upload_media.call_args
        assert kwargs["name"] == series_storage_name("clips")
        assert kwargs["step"] == 7
        assert kwargs["media_type"] == V1MediaType.VIDEO
        assert exp._stats.media_logged == 1

    @patch.object(Model, "_log_model", return_value="owner/team/exp-model-series:latest")
    def test_model_series_uses_series_key_for_remote_binding(self, mock_log_model):
        exp = MagicMock()
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
        model.log(_session_of(exp), WritePlacement("models", index=2))

        mock_log_model.assert_called_once_with(
            experiment_name="exp",
            teamspace=exp._teamspace,
            key="models",
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

    def test_getitem_returns_video(self):
        exp = _make_exp()
        video = Video("preview.mp4")
        exp["preview"] = video

        assert exp["preview"] is video

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
    """Test explicit model lookup without network I/O in generic indexing."""

    def test_getitem_does_not_query_the_registry(self):
        exp = _make_exp()
        exp._teamspace.list_models.side_effect = RuntimeError("registry unavailable")

        result = exp["checkpoint"]

        assert isinstance(result, Series)
        exp._teamspace.list_models.assert_not_called()

    def test_resolve_model_resolves_remote_model_from_teamspace_listing(self):
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

        result = exp.resolve_model("models/latest")

        assert isinstance(result, Model)
        assert result._model_kind == "artifact"
        assert result._model_name == "owner/teamspace/models-latest:new-artifact-v1"
        assert exp._key_types["models/latest"] == "static_file"
        assert exp._static_files["models/latest"] is result

    def test_resolve_model_resolves_remote_model_series_from_multiple_versions(self):
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

        result = exp.resolve_model("checkpoints")

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

    @staticmethod
    def _make_rebuild_exp():
        exp = MagicMock(spec=Experiment)
        type(exp)._session = property(lambda self: _session_of(self))
        exp.name = "exp1"
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._resumed_steps = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.name = "exp1"
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_store.cluster_id = "cloud-1"
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._metrics_api = MagicMock()
        # The rebuild re-reads the store from the API first; keep the seeded one.
        exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store
        exp._metrics_api.get_metric_values.return_value = {}
        exp._artifacts_api = MagicMock()
        exp._artifacts_api.list_experiment_artifacts.return_value = None
        exp._media_api = MagicMock()
        exp._media_api.list_media.return_value = []
        exp._merge_restored = lambda restored: Experiment._merge_restored(exp, restored)
        return exp

    def test_rebuilds_artifacts_with_name_and_download(self):
        exp = self._make_rebuild_exp()
        art = MagicMock()
        art.path = "results.csv"
        exp._metrics_store.artifacts = [art]
        exp._artifacts_api.download_file.side_effect = lambda teamspace, remote_path, local_path, cloud_account=None: (
            local_path
        )

        Experiment._rebuild_state(exp)

        f = exp._static_files["results.csv"]
        assert isinstance(f, File)
        assert f.name == "results.csv"
        assert f._download_fn is not None
        assert f.save("/tmp/out.csv") == "/tmp/out.csv"
        kwargs = exp._artifacts_api.download_file.call_args.kwargs
        assert kwargs["remote_path"] == "experiments/exp1/results.csv"

    def test_rebuild_loads_artifacts_from_logger_artifacts_api(self):
        exp = self._make_rebuild_exp()
        art = MagicMock()
        art.path = "results.csv"
        exp._artifacts_api.list_experiment_artifacts.return_value = [art]

        Experiment._rebuild_state(exp)

        assert "results.csv" in exp._static_files
        exp._metrics_api.get_experiment_metrics_by_name.assert_called_once()

    def test_rebuilds_artifact_series_from_logger_artifacts_api(self):
        exp = self._make_rebuild_exp()
        art0 = MagicMock()
        art0.path = "reports/0"
        art1 = MagicMock()
        art1.path = "reports/1"
        exp._artifacts_api.list_experiment_artifacts.return_value = [art1, art0]

        Experiment._rebuild_state(exp)

        assert exp._key_types["reports"] == "file_series"
        assert isinstance(exp._series["reports"], Series)
        assert [item.name for item in exp._series["reports"]] == ["reports/0", "reports/1"]

    def test_rebuild_does_not_overwrite_existing_keys(self):
        exp = self._make_rebuild_exp()
        exp._key_types = {"existing": "metric"}
        art = MagicMock()
        art.path = "existing"
        exp._metrics_store.artifacts = [art]

        Experiment._rebuild_state(exp)

        # Should not overwrite the existing metric key
        assert exp._key_types["existing"] == "metric"
        assert "existing" not in exp._static_files

    @staticmethod
    def _media_record(name, storage_path, media_type, media_id, step=None):
        media = MagicMock()
        media.name = name
        media.storage_path = storage_path
        media.cluster_id = "cloud-1"
        media.media_type = media_type
        media.id = media_id
        media.step = step
        return media

    def test_rebuilds_static_media_with_wrapper(self):
        exp = self._make_rebuild_exp()
        media = self._media_record("preview", "media/preview.png", V1MediaType.IMAGE, "media-1")
        exp._media_api.list_media.return_value = [media]

        Experiment._rebuild_state(exp)

        wrapped = exp._static_files["preview"]
        assert isinstance(wrapped, Image)
        assert wrapped.name == "preview"
        assert wrapped._download_fn is not None

    def test_rebuilds_static_video_with_wrapper(self):
        exp = self._make_rebuild_exp()
        media = self._media_record("preview", "media/preview.mp4", V1MediaType.VIDEO, "media-1")
        exp._media_api.list_media.return_value = [media]

        Experiment._rebuild_state(exp)

        wrapped = exp._static_files["preview"]
        assert isinstance(wrapped, Video)
        assert wrapped.name == "preview"
        assert wrapped._download_fn is not None

    def test_rebuilds_explicit_numeric_static_media_key(self):
        exp = self._make_rebuild_exp()
        media = self._media_record(
            static_storage_name("reports/2024"),
            "media/report.png",
            V1MediaType.IMAGE,
            "media-1",
        )
        exp._media_api.list_media.return_value = [media]

        Experiment._rebuild_state(exp)

        assert exp._key_types["reports/2024"] == "static_file"
        assert isinstance(exp._static_files["reports/2024"], Image)

    def test_rebuilds_explicit_media_series(self):
        exp = self._make_rebuild_exp()
        name = series_storage_name("logs")
        media0 = self._media_record(name, "media/logs-0.txt", V1MediaType.TEXT, "media-0", step=0)
        media1 = self._media_record(name, "media/logs-1.txt", V1MediaType.TEXT, "media-1", step=1)
        exp._media_api.list_media.return_value = [media1, media0]

        Experiment._rebuild_state(exp)

        assert exp._key_types["logs"] == "file_series"
        assert [item.name for item in exp._series["logs"]] == ["logs", "logs"]

    def test_rebuilds_media_series_with_wrapper(self):
        exp = self._make_rebuild_exp()
        media0 = self._media_record("logs/0", "media/logs-0.txt", V1MediaType.TEXT, "media-0")
        media1 = self._media_record("logs/1", "media/logs-1.txt", V1MediaType.TEXT, "media-1")
        exp._media_api.list_media.return_value = [media1, media0]

        Experiment._rebuild_state(exp)

        assert exp._key_types["logs"] == "file_series"
        assert isinstance(exp._series["logs"], Series)
        assert len(exp._series["logs"]) == 2
        assert all(isinstance(item, Text) for item in exp._series["logs"])

    def test_rebuilds_same_name_media_series_with_wrapper(self):
        exp = self._make_rebuild_exp()
        media0 = self._media_record("logs", "media/logs-0.txt", V1MediaType.TEXT, "media-0", step=0)
        media1 = self._media_record("logs", "media/logs-1.txt", V1MediaType.TEXT, "media-1", step=1)
        exp._media_api.list_media.return_value = [media1, media0]

        Experiment._rebuild_state(exp)

        assert exp._key_types["logs"] == "file_series"
        assert isinstance(exp._series["logs"], Series)
        assert len(exp._series["logs"]) == 2
        assert [item.path for item in exp._series["logs"]] == ["logs", "logs"]

    def test_rebuilds_same_name_media_series_ordered_by_string_step(self):
        exp = self._make_rebuild_exp()
        media10 = self._media_record("logs", "media/logs-10.txt", V1MediaType.TEXT, "media-10", step="10")
        media2 = self._media_record("logs", "media/logs-2.txt", V1MediaType.TEXT, "media-2", step="2")
        exp._media_api.list_media.return_value = [media10, media2]

        Experiment._rebuild_state(exp)

        exp._series["logs"][0].save("/tmp/out.txt")
        assert exp._teamspace.download_file.call_args.args[0] == "media/logs-2.txt"

    def test_rebuilds_same_name_media_series_ordered_by_fractional_step(self):
        exp = self._make_rebuild_exp()
        media15 = self._media_record("logs", "media/logs-15.txt", V1MediaType.TEXT, "media-15", step="1.5")
        media05 = self._media_record("logs", "media/logs-05.txt", V1MediaType.TEXT, "media-05", step="0.5")
        exp._media_api.list_media.return_value = [media15, media05]

        Experiment._rebuild_state(exp)

        exp._series["logs"][0].save("/tmp/out.txt")
        assert exp._teamspace.download_file.call_args.args[0] == "media/logs-05.txt"


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
