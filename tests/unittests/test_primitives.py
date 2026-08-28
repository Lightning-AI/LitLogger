# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for the logging primitives against a mocked session."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from lightning_sdk.lightning_cloud.openapi import V1MediaType

from litlogger.primitives import (
    File,
    Metadata,
    Metric,
    MetricWrite,
    Model,
    Primitive,
    PrimitiveWrite,
    Text,
    WritePlacement,
)
from litlogger.primitives._utils import natural_sort_key, series_storage_name, static_storage_name
from litlogger.session import ExperimentSession
from litlogger.types import PhaseType


def make_session(**overrides):
    """Build a real ExperimentSession wired with mock infrastructure."""
    experiment = MagicMock()
    experiment.name = "exp"
    store = MagicMock()
    store.id = "ms-1"
    store.name = "exp"
    store.tags = []
    store.cluster_id = "acc-1"
    experiment._metrics_store = store
    teamspace = MagicMock()
    teamspace.id = "ts-1"
    background = MagicMock()
    background.exception = None
    stats = MagicMock()
    stats.artifacts_logged = 0
    stats.media_logged = 0
    stats.models_logged = 0
    session = ExperimentSession._from_components(
        name="exp",
        metrics_api=MagicMock(),
        media_api=MagicMock(),
        artifacts_api=MagicMock(),
        teamspace=teamspace,
        metrics_store=store,
        experiment=experiment,
        queue_=MagicMock(),
        stats=stats,
        printer=MagicMock(),
        store_step=True,
        store_created_at=False,
        last_x={},
        background=background,
    )
    for key, value in overrides.items():
        if key == "last_steps":
            session.last_x = value
        else:
            setattr(session, key, value)
    return session


class TestPrimitiveContract:
    """Every primitive satisfies the runtime Primitive protocol."""

    @pytest.mark.parametrize(
        "primitive",
        [
            Metric("loss", 0.5),
            Metadata("lr", "0.001"),
        ],
        ids=["metric", "metadata"],
    )
    def test_satisfies_protocol(self, primitive):
        assert isinstance(primitive, Primitive)

    def test_static_conformance(self):
        # Typed assignments verified by mypy; runtime smoke check of the same shape.
        _p: Primitive = Metric("loss", 1.0)
        _q: Primitive = Metadata("k", "v")
        assert callable(_p.log)
        assert callable(_p.enqueue)
        assert callable(_q.log)
        assert callable(_q.enqueue)


class TestMetricEnqueue:
    """Metric.enqueue feeds the background batching path."""

    def test_puts_single_metric_command(self):
        session = make_session()

        Metric("loss", 0.5, step=3).enqueue(session)

        session.queue.put.assert_called_once()
        command = session.queue.put.call_args[0][0]
        assert command == MetricWrite(key="loss", y=0.5, x=3, created_at=None)
        session.stats.record_metric.assert_called_once_with("loss", 0.5)

    def test_x_and_step_are_mutually_exclusive(self):
        session = make_session()

        with pytest.raises(ValueError, match="mutually exclusive"):
            Metric("loss", 0.5, step=3, x=1.5).enqueue(session)

        session.queue.put.assert_not_called()

    @pytest.mark.parametrize("x", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_x(self, x):
        session = make_session()

        with pytest.raises(ValueError, match="finite"):
            Metric("loss", 0.5, x=x).enqueue(session)

        session.queue.put.assert_not_called()

    def test_store_step_false_retains_x_for_worker_sequence(self):
        session = make_session(store_step=False)

        Metric("loss", 0.5, step=3).enqueue(session)

        command = session.queue.put.call_args[0][0]
        assert command.x == 3

    def test_store_created_at_sets_timestamp(self):
        session = make_session(store_created_at=True)

        Metric("loss", 0.5).enqueue(session)

        command = session.queue.put.call_args[0][0]
        assert isinstance(command.created_at, datetime)

    def test_raises_before_put_on_background_failure(self):
        session = make_session()
        session.background.exception = RuntimeError("bg error")

        with pytest.raises(RuntimeError, match="bg error"):
            Metric("loss", 0.5).enqueue(session)

        session.queue.put.assert_not_called()
        session.stats.record_metric.assert_not_called()


class TestMetricLog:
    """Metric.log appends synchronously with worker-equivalent auto-stepping."""

    def test_appends_immediately(self):
        session = make_session()

        Metric("loss", 0.5, step=7).log(session)

        session.metrics_api.append_experiment_metrics.assert_called_once()
        kwargs = session.metrics_api.append_experiment_metrics.call_args.kwargs
        assert kwargs["teamspace_id"] == "ts-1"
        assert kwargs["metrics_store_id"] == "ms-1"
        (metrics,) = kwargs["metrics"]
        assert metrics.name == "loss"
        assert metrics.values[0].value == 0.5
        assert metrics.values[0].step == 7
        session.stats.record_metric.assert_called_once_with("loss", 0.5)

    def test_auto_step_continues_sequence(self):
        session = make_session(last_steps={"loss": 4})

        Metric("loss", 0.5).log(session)

        kwargs = session.metrics_api.append_experiment_metrics.call_args.kwargs
        assert kwargs["metrics"][0].values[0].step == 5
        assert session.last_steps["loss"] == 5

    def test_explicit_x_updates_sequence(self):
        session = make_session(last_steps={"loss": 4})

        Metric("loss", 0.5, x=1.5).log(session)

        assert session.last_steps["loss"] == 1.5

        Metric("loss", 0.4).log(session)

        kwargs = session.metrics_api.append_experiment_metrics.call_args.kwargs
        assert kwargs["metrics"][0].values[0].step == 2.5

    def test_store_step_false_uses_explicit_x_for_sequence_without_persisting_it(self):
        session = make_session(store_step=False)

        Metric("loss", 0.5, step=100).log(session)

        kwargs = session.metrics_api.append_experiment_metrics.call_args.kwargs
        assert kwargs["metrics"][0].values[0].step is None
        assert session.last_steps["loss"] == 100


class TestMetricRestore:
    """Metric._restore_values pulls every series' values through the API layer."""

    def test_delegates_to_metrics_api(self):
        session = make_session()
        session.metrics_api.get_metric_values.return_value = {"loss": [1.0, 0.5]}

        result = Metric._restore_values(session)

        session.metrics_api.get_metric_values.assert_called_once_with("ts-1", "ms-1")
        assert result == {"loss": [1.0, 0.5]}


class TestMetadataLog:
    """Metadata.log read-modify-writes the full code-tag collection."""

    def _tag(self, name, value, from_code=True):
        tag = MagicMock()
        tag.name = name
        tag.value = value
        tag.from_code = from_code
        return tag

    def test_merges_into_freshly_read_tags(self):
        session = make_session()
        refreshed = MagicMock()
        refreshed.id = "ms-1"
        refreshed.name = "exp"
        refreshed.tags = [self._tag("lr", "0.001")]
        session.metrics_api.get_experiment_metrics_by_name.return_value = refreshed

        Metadata("batch_size", "32").log(session)

        session.metrics_api.get_experiment_metrics_by_name.assert_called_once_with("ts-1", name="exp")
        kwargs = session.metrics_api.update_experiment_metrics.call_args.kwargs
        assert kwargs["teamspace_id"] == "ts-1"
        assert kwargs["metrics_store_id"] == "ms-1"
        assert kwargs["phase"] == PhaseType.RUNNING
        assert kwargs["metadata"] == {"lr": "0.001", "batch_size": "32"}

    def test_non_code_tags_are_dropped(self):
        session = make_session()
        refreshed = MagicMock()
        refreshed.id = "ms-1"
        refreshed.name = "exp"
        refreshed.tags = [self._tag("lr", "0.001"), self._tag("ui-tag", "x", from_code=False)]
        session.metrics_api.get_experiment_metrics_by_name.return_value = refreshed

        Metadata("batch_size", "32").log(session)

        kwargs = session.metrics_api.update_experiment_metrics.call_args.kwargs
        assert kwargs["metadata"] == {"lr": "0.001", "batch_size": "32"}

    def test_overwrites_existing_key(self):
        session = make_session()
        refreshed = MagicMock()
        refreshed.id = "ms-1"
        refreshed.name = "exp"
        refreshed.tags = [self._tag("lr", "0.001")]
        session.metrics_api.get_experiment_metrics_by_name.return_value = refreshed

        Metadata("lr", "0.01").log(session)

        kwargs = session.metrics_api.update_experiment_metrics.call_args.kwargs
        assert kwargs["metadata"] == {"lr": "0.01"}


class TestMetadataEnqueue:
    """Metadata.enqueue defers the read-modify-write to the background worker."""

    def test_puts_queued_write(self):
        session = make_session()
        entry = Metadata("lr", "0.001")

        entry.enqueue(session)

        session.queue.put.assert_called_once()
        item = session.queue.put.call_args[0][0]
        assert isinstance(item, PrimitiveWrite)
        item.execute(session)
        session.metrics_api.update_experiment_metrics.assert_called_once()

    def test_raises_before_put_on_background_failure(self):
        session = make_session()
        session.background.exception = RuntimeError("bg error")

        with pytest.raises(RuntimeError, match="bg error"):
            Metadata("lr", "0.001").enqueue(session)

        session.queue.put.assert_not_called()


class TestFileLog:
    """File.log uploads a plain artifact synchronously."""

    def test_static_upload_uses_key_as_remote_path(self, tmp_path):
        session = make_session()
        local = tmp_path / "config.yaml"
        local.write_text("lr: 0.1")
        f = File(str(local))
        f.log(session, WritePlacement("config"))

        session.artifacts_api.upload_experiment_file_artifact.assert_called_once()
        kwargs = session.artifacts_api.upload_experiment_file_artifact.call_args.kwargs
        assert kwargs["remote_path"] == "config"
        assert kwargs["experiment_name"] == "exp"
        assert f.name == "config"
        assert f._download_fn is not None
        assert session.stats.artifacts_logged == 1

    def test_series_element_uploads_under_indexed_path(self, tmp_path):
        session = make_session()
        local = tmp_path / "frame.png"
        local.write_bytes(b"png")
        f = File(str(local))
        f.log(session, WritePlacement("frames", index=5))

        kwargs = session.artifacts_api.upload_experiment_file_artifact.call_args.kwargs
        assert kwargs["remote_path"] == series_storage_name("frames", 5)
        assert f.name == "frames"

    def test_numeric_static_key_uses_explicit_storage_name(self, tmp_path):
        session = make_session()
        local = tmp_path / "report.txt"
        local.write_text("done")
        file = File(str(local))

        file.log(session, WritePlacement("reports/2024"))

        kwargs = session.artifacts_api.upload_experiment_file_artifact.call_args.kwargs
        assert kwargs["remote_path"] == static_storage_name("reports/2024")
        assert file.name == "reports/2024"

    def test_unstamped_upload_derives_path_from_file(self, tmp_path, monkeypatch):
        session = make_session()
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "results.csv"
        local.write_text("a,b")
        f = File(str(local))

        f.log(session)

        kwargs = session.artifacts_api.upload_experiment_file_artifact.call_args.kwargs
        assert kwargs["remote_path"] == "results.csv"

    def test_download_binding_flows_through_session_api(self, tmp_path):
        session = make_session()
        session.artifacts_api.download_file.side_effect = (
            lambda teamspace, remote_path, local_path, cloud_account=None: local_path
        )
        local = tmp_path / "config.yaml"
        local.write_text("lr: 0.1")
        f = File(str(local))
        f.log(session, WritePlacement("config"))

        assert f.save(str(tmp_path / "out.yaml")) == str(tmp_path / "out.yaml")
        kwargs = session.artifacts_api.download_file.call_args.kwargs
        assert kwargs["remote_path"] == "experiments/exp/config"
        assert kwargs["cloud_account"] == "acc-1"

    def test_enqueue_puts_queued_write(self):
        session = make_session()
        f = File("config.yaml")

        f.enqueue(session, WritePlacement("config"))

        item = session.queue.put.call_args[0][0]
        assert isinstance(item, PrimitiveWrite)

    def test_reusing_wrapper_does_not_change_queued_placement(self, tmp_path):
        session = make_session()
        local = tmp_path / "artifact.txt"
        local.write_text("content")
        file = File(str(local))

        file.enqueue(session, WritePlacement("first"))
        first = session.queue.put.call_args_list[0].args[0]
        file.enqueue(session, WritePlacement("second"))
        second = session.queue.put.call_args_list[1].args[0]
        first.execute(session)

        kwargs = session.artifacts_api.upload_experiment_file_artifact.call_args.kwargs
        assert kwargs["remote_path"] == "first"
        assert file.name == "first"
        second.execute(session)
        assert file.name == "second"

    def test_failed_enqueue_cleans_prepared_snapshot(self, tmp_path, monkeypatch):
        session = make_session()
        session.queue.put.side_effect = RuntimeError("queue failed")
        source = tmp_path / "artifact.txt"
        source.write_text("content")
        snapshot = tmp_path / "snapshot.txt"

        def prepare(file):
            snapshot.write_text(source.read_text())
            file._temp_path = str(snapshot)
            return str(snapshot)

        monkeypatch.setattr(File, "_get_upload_path", prepare)
        file = File(str(source))

        with pytest.raises(RuntimeError, match="queue failed"):
            file.enqueue(session, WritePlacement("artifact"))

        assert not snapshot.exists()
        assert file._read_barrier is None

    def test_failed_upload_cleans_temporary_copy(self, tmp_path):
        source = tmp_path / "artifact.txt"
        source.write_text("content")
        session = make_session()
        session.artifacts_api.upload_experiment_file_artifact.side_effect = RuntimeError("upload failed")
        file = File(str(source))

        with pytest.raises(RuntimeError, match="upload failed"):
            file.log(session)

        assert file._temp_path is None

    def test_failed_copy_fallback_cleans_placeholder(self, tmp_path):
        source = tmp_path / "artifact.txt"
        source.write_text("content")
        session = make_session()
        file = File(str(source))

        with (
            patch("os.link", side_effect=OSError("hard links unavailable")),
            patch("shutil.copy2", side_effect=OSError("copy failed")),
            pytest.raises(OSError, match="copy failed"),
        ):
            file.log(session)

        assert file._temp_path is None

    def test_duplicate_restored_indices_remain_stable(self):
        session = make_session()
        first = MagicMock(path="frames/0")
        second = MagicMock(path="frames/0")
        session.metrics_store.artifacts = [first, second]
        session.artifacts_api.list_experiment_artifacts.return_value = None

        restored = File._restore_all(session, {})

        assert [file.name for file in restored.series["frames"]] == ["frames/0", "frames/0"]

    def test_explicit_numeric_static_key_is_not_restored_as_series(self):
        session = make_session()
        artifact = MagicMock(path=static_storage_name("reports/2024"))
        session.artifacts_api.list_experiment_artifacts.return_value = [artifact]

        restored = File._restore_all(session, {})

        assert restored.series == {}
        assert restored.statics["reports/2024"].name == "reports/2024"

    def test_explicit_series_is_restored_in_index_order(self):
        session = make_session()
        first = MagicMock(path=series_storage_name("frames", 0))
        second = MagicMock(path=series_storage_name("frames", 1))
        session.artifacts_api.list_experiment_artifacts.return_value = [second, first]

        restored = File._restore_all(session, {})

        assert [file.name for file in restored.series["frames"]] == ["frames", "frames"]


class TestMediaLog:
    """Image/Video/Text upload through the media API under the bare key."""

    def test_static_media_uploads_without_step(self):
        session = make_session()
        text = Text("hello")
        text.log(session, WritePlacement("notes"))

        session.media_api.upload_media.assert_called_once()
        kwargs = session.media_api.upload_media.call_args.kwargs
        assert kwargs["experiment_id"] == "ms-1"
        assert kwargs["name"] == "notes"
        assert kwargs["media_type"] == V1MediaType.TEXT
        assert kwargs["step"] is None
        assert text.name == "notes"
        assert session.stats.media_logged == 1

    def test_series_media_uploads_bare_key_with_step(self):
        session = make_session()
        text = Text("hello")
        text.log(session, WritePlacement("logs", index=2, x=7))

        kwargs = session.media_api.upload_media.call_args.kwargs
        assert kwargs["name"] == series_storage_name("logs")
        assert kwargs["step"] == 7

    def test_media_upload_cleans_up_rendered_temp(self):
        session = make_session()
        text = Text("hello")
        text.log(session, WritePlacement("notes"))

        assert text._temp_path is None or not os.path.exists(text._temp_path)

    def test_failed_media_upload_cleans_up_rendered_temp(self):
        session = make_session()
        session.media_api.upload_media.side_effect = RuntimeError("upload failed")
        text = Text("hello")

        with pytest.raises(RuntimeError, match="upload failed"):
            text.log(session)

        assert text._temp_path is None


class TestModelLog:
    """Model.log uploads through the registry and binds remote access."""

    def test_natural_version_sort_handles_mixed_prefixes(self):
        versions = ["v10", "2v", "v2", "10v"]

        assert sorted(versions, key=natural_sort_key) == ["2v", "10v", "v2", "v10"]

    @patch.object(Model, "_log_model", return_value="owner/team/checkpoint:latest")
    def test_static_model_upload(self, mock_log_model):
        session = make_session()
        model = Model("model.ckpt")
        model.log(session, WritePlacement("checkpoint"))

        mock_log_model.assert_called_once_with(
            experiment_name="exp",
            teamspace=session.teamspace,
            key="checkpoint",
            experiment=session.experiment_link,
            cloud_account="acc-1",
        )
        assert model.name == "checkpoint"
        assert model._model_name == "owner/team/checkpoint:latest"
        assert model._download_fn is not None
        assert session.stats.models_logged == 1

    @patch.object(Model, "_log_model", return_value="owner/team/models:v3")
    def test_series_model_auto_versions(self, mock_log_model):
        session = make_session()
        model = Model("model.ckpt")
        model.log(session, WritePlacement("models", index=2))

        assert model.version == "v3"
        assert model.name == "models"

    @patch.object(Model, "_log_model", return_value="owner/team/m:v9")
    def test_series_model_explicit_version_wins(self, mock_log_model):
        session = make_session()
        model = Model("model.ckpt", version="v9")
        model.log(session, WritePlacement("models", index=2))

        assert model.version == "v9"

    @patch.object(Model, "_log_model", return_value="owner/team/m:latest")
    def test_key_is_sanitized_for_registry(self, mock_log_model):
        session = make_session()
        model = Model("model.ckpt")
        model.log(session, WritePlacement("models/latest"))

        assert mock_log_model.call_args.kwargs["key"] == "models-latest"
        assert model.name == "models/latest"


class TestMediaTypeConversion:
    """_to_v1_media_type maps user types to wire types."""

    def test_maps_video(self):
        from litlogger.primitives import _to_v1_media_type
        from litlogger.types import MediaType

        assert _to_v1_media_type(MediaType.VIDEO) == V1MediaType.VIDEO

    def test_rejects_non_media_types(self):
        from litlogger.primitives import _to_v1_media_type
        from litlogger.types import MediaType

        with pytest.raises(ValueError, match="Unsupported media type"):
            _to_v1_media_type(MediaType.MODEL)


class TestWrapMediaFile:
    """Restored media records are wrapped by their wire type."""

    def test_wraps_text_with_path(self):
        from litlogger.primitives.file import _wrap_media_file

        wrapped = _wrap_media_file("logs/0", V1MediaType.TEXT)

        assert isinstance(wrapped, Text)
        assert wrapped.path == "logs/0"

    def test_wraps_video(self):
        from litlogger.primitives import Video
        from litlogger.primitives.file import _wrap_media_file

        wrapped = _wrap_media_file("clips/0", V1MediaType.VIDEO)

        assert isinstance(wrapped, Video)
        assert wrapped.path == "clips/0"


class TestReadBarriers:
    """Queued writes must land before dependent remote reads."""

    def test_enqueue_attaches_read_barrier(self, tmp_path):
        session = make_session()
        local = tmp_path / "config.yaml"
        local.write_text("lr: 0.1")
        f = File(str(local))
        f.enqueue(session, WritePlacement("config"))

        assert f._read_barrier is not None

    def test_save_flushes_queued_writes_first(self, tmp_path):
        session = make_session()
        local = tmp_path / "config.yaml"
        local.write_text("lr: 0.1")
        f = File(str(local))
        f.enqueue(session, WritePlacement("config"))

        # Simulate the worker: process the queued write, then the download
        # must observe the completed upload.
        events = []
        session.queue.join.side_effect = lambda: events.append("flush")
        session.artifacts_api.download_file.side_effect = (
            lambda teamspace, remote_path, local_path, cloud_account=None: (
                events.append("download"),
                local_path,
            )[1]
        )
        item = session.queue.put.call_args[0][0]
        item.execute(session)

        f.save(str(tmp_path / "out.yaml"))

        assert events == ["download"]

    def test_save_still_requires_remote_context(self):
        session = make_session()
        f = File("never-uploaded.txt")
        f._read_barrier = session.flush

        with pytest.raises(RuntimeError, match="no remote context"):
            f.save("out.txt")


class TestWriteBehindEndToEnd:
    """A real worker drains queued file writes and finalize-style joins cover them."""

    def test_worker_processes_enqueued_file(self, tmp_path):
        import queue as queue_module

        from litlogger.background import _BackgroundThread

        session = make_session(queue=queue_module.Queue())
        local = tmp_path / "report.txt"
        local.write_text("done")
        f = File(str(local))
        worker = _BackgroundThread(session=session)
        worker.start()
        session.background = worker

        f.enqueue(session, WritePlacement("report"))
        session.queue.join()  # what finalize()/read barriers do
        session.stop_event.set()
        worker.join(timeout=10)

        session.artifacts_api.upload_experiment_file_artifact.assert_called_once()
        assert f.name == "report"
        assert f._download_fn is not None
        assert session.stats.artifacts_logged == 1
