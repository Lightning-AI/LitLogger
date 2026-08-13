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
from litlogger.media import File, Model, Text
from litlogger.primitives import Metadata, Metric, Primitive, _QueuedWrite
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
    session = ExperimentSession(
        client=MagicMock(),
        metrics_api=MagicMock(),
        media_api=MagicMock(),
        artifacts_api=MagicMock(),
        teamspace=teamspace,
        experiment=experiment,
        queue=MagicMock(),
        stats=stats,
        store_step=True,
        store_created_at=False,
        last_steps={},
        background=background,
    )
    for key, value in overrides.items():
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
        assert callable(_p.log) and callable(_p.enqueue)
        assert callable(_q.log) and callable(_q.enqueue)


class TestMetricEnqueue:
    """Metric.enqueue feeds the background batching path."""

    def test_puts_single_value_batch(self):
        session = make_session()

        Metric("loss", 0.5, step=3).enqueue(session)

        session.queue.put.assert_called_once()
        batch = session.queue.put.call_args[0][0]
        assert list(batch.keys()) == ["loss"]
        metrics = batch["loss"]
        assert metrics.name == "loss"
        assert len(metrics.values) == 1
        assert metrics.values[0].value == 0.5
        assert metrics.values[0].step == 3
        assert metrics.values[0].created_at is None
        session.stats.record_metric.assert_called_once_with("loss", 0.5)

    def test_store_step_false_drops_step(self):
        session = make_session(store_step=False)

        Metric("loss", 0.5, step=3).enqueue(session)

        batch = session.queue.put.call_args[0][0]
        assert batch["loss"].values[0].step is None

    def test_store_created_at_sets_timestamp(self):
        session = make_session(store_created_at=True)

        Metric("loss", 0.5).enqueue(session)

        batch = session.queue.put.call_args[0][0]
        assert isinstance(batch["loss"].values[0].created_at, datetime)

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

    def test_explicit_step_does_not_update_sequence(self):
        # Mirrors the background worker: only auto-assigned steps advance last_steps.
        session = make_session(last_steps={"loss": 4})

        Metric("loss", 0.5, step=100).log(session)

        assert session.last_steps["loss"] == 4

    def test_store_step_false_still_auto_steps(self):
        # store_step=False means "ignore user-provided steps"; the worker then
        # auto-assigns, and the synchronous path mirrors that.
        session = make_session(store_step=False)

        Metric("loss", 0.5, step=100).log(session)

        kwargs = session.metrics_api.append_experiment_metrics.call_args.kwargs
        assert kwargs["metrics"][0].values[0].step == 0
        assert session.last_steps["loss"] == 0


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
        assert isinstance(item, _QueuedWrite)
        assert item.primitive is entry

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
        f._log_key = "config"

        f.log(session)

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
        f._log_key = "frames"
        f._series_index = 5

        f.log(session)

        kwargs = session.artifacts_api.upload_experiment_file_artifact.call_args.kwargs
        assert kwargs["remote_path"] == "frames/5"
        assert f.name == "frames/5"

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
        f._log_key = "config"

        f.log(session)

        assert f.save(str(tmp_path / "out.yaml")) == str(tmp_path / "out.yaml")
        kwargs = session.artifacts_api.download_file.call_args.kwargs
        assert kwargs["remote_path"] == "experiments/exp/config"
        assert kwargs["cloud_account"] == "acc-1"

    def test_enqueue_puts_queued_write(self):
        session = make_session()
        f = File("config.yaml")

        f.enqueue(session)

        item = session.queue.put.call_args[0][0]
        assert isinstance(item, _QueuedWrite)
        assert item.primitive is f


class TestMediaLog:
    """Image/Video/Text upload through the media API under the bare key."""

    def test_static_media_uploads_without_step(self):
        session = make_session()
        text = Text("hello")
        text._log_key = "notes"

        text.log(session)

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
        text._log_key = "logs"
        text._series_index = 2
        text._series_step = 7

        text.log(session)

        kwargs = session.media_api.upload_media.call_args.kwargs
        assert kwargs["name"] == "logs"  # never "logs/2"
        assert kwargs["step"] == 7

    def test_media_upload_cleans_up_rendered_temp(self):
        session = make_session()
        text = Text("hello")
        text._log_key = "notes"

        text.log(session)

        assert text._temp_path is None or not os.path.exists(text._temp_path)


class TestModelLog:
    """Model.log uploads through the registry and binds remote access."""

    @patch.object(Model, "_log_model", return_value="owner/team/checkpoint:latest")
    def test_static_model_upload(self, mock_log_model):
        session = make_session()
        model = Model("model.ckpt")
        model._log_key = "checkpoint"

        model.log(session)

        mock_log_model.assert_called_once_with(
            experiment_name="exp",
            teamspace=session.teamspace,
            key="checkpoint",
            experiment=session.experiment,
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
        model._log_key = "models"
        model._series_index = 2

        model.log(session)

        assert model.version == "v3"
        assert model.name == "models"

    @patch.object(Model, "_log_model", return_value="owner/team/m:v9")
    def test_series_model_explicit_version_wins(self, mock_log_model):
        session = make_session()
        model = Model("model.ckpt", version="v9")
        model._log_key = "models"
        model._series_index = 2

        model.log(session)

        assert model.version == "v9"

    @patch.object(Model, "_log_model", return_value="owner/team/m:latest")
    def test_key_is_sanitized_for_registry(self, mock_log_model):
        session = make_session()
        model = Model("model.ckpt")
        model._log_key = "models/latest"

        model.log(session)

        assert mock_log_model.call_args.kwargs["key"] == "models-latest"
        assert model.name == "models/latest"
