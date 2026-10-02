# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for offline mode: local SQLite storage, resume, sync, and disabled mode."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from litlogger.offline import (
    OfflineSession,
    _DEFAULT_MAX_BATCH_SIZE,
    _init_db,
    _send_metrics_batched,
    sync,
)
from litlogger.types import Metrics, MetricValue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_log_dir(tmp_path):
    """Return a fresh temporary directory for each test."""
    return str(tmp_path / "lightning_logs" / "test-run")


def _make_offline_session(tmp_log_dir: str, name: str = "test-exp", **kwargs):
    """Create an OfflineSession pointing at *tmp_log_dir*."""
    os.makedirs(tmp_log_dir, exist_ok=True)
    return OfflineSession(name=name, log_dir=tmp_log_dir, verbose=False, **kwargs)


# ---------------------------------------------------------------------------
# Schema / DB initialisation
# ---------------------------------------------------------------------------


class TestSchemaInit:
    def test_creates_db_file(self, tmp_log_dir):
        db_path = os.path.join(tmp_log_dir, "offline.db")
        os.makedirs(tmp_log_dir, exist_ok=True)
        conn = _init_db(db_path)
        assert os.path.exists(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"experiment", "metadata", "metrics", "artifacts"} <= tables
        conn.close()

    def test_created_at_uses_real_type(self, tmp_log_dir):
        """Verify the schema stores timestamps as REAL (Unix epoch)."""
        db_path = os.path.join(tmp_log_dir, "offline.db")
        os.makedirs(tmp_log_dir, exist_ok=True)
        conn = _init_db(db_path)
        # Insert a metric with a float timestamp
        conn.execute(
            "INSERT INTO metrics (key, value, x, created_at) VALUES (?, ?, ?, ?)",
            ("k", 1.0, 0.0, 1234567890.5),
        )
        row = conn.execute("SELECT created_at FROM metrics").fetchone()
        assert isinstance(row[0], float)
        conn.close()


# ---------------------------------------------------------------------------
# OfflineSession – writing
# ---------------------------------------------------------------------------


class TestOfflineSessionWrite:
    def test_write_and_read_metadata(self, tmp_log_dir):
        session = _make_offline_session(tmp_log_dir)
        session.write_metadata("optimizer", "adam")
        session.write_metadata("lr", "0.001")
        assert session.read_all_metadata() == {"optimizer": "adam", "lr": "0.001"}
        session.finalize()

    def test_write_and_read_metrics(self, tmp_log_dir):
        session = _make_offline_session(tmp_log_dir)
        session.write_metric("loss", 1.0, x=None, created_at=None)
        session.write_metric("loss", 0.5, x=None, created_at=None)
        session.write_metric("acc", 0.9, x=None, created_at=None)
        data = session.read_all_metrics()
        assert list(data.keys()) == ["loss", "acc"]
        assert [e["value"] for e in data["loss"]] == [1.0, 0.5]
        session.finalize()

    def test_write_artifact_copies_file(self, tmp_log_dir, tmp_path):
        session = _make_offline_session(tmp_log_dir)
        src = tmp_path / "model.pt"
        src.write_text("fake-weights")
        session.write_artifact("weights", str(src), kind="static")
        artifacts = session.read_all_artifacts()
        assert len(artifacts) == 1
        assert artifacts[0]["key"] == "weights"
        assert os.path.exists(artifacts[0]["local_path"])
        session.finalize()

    def test_coordinate_auto_increment(self, tmp_log_dir):
        session = _make_offline_session(tmp_log_dir)
        x0 = session.resolve_x("loss", None)
        x1 = session.resolve_x("loss", None)
        x2 = session.resolve_x("loss", None)
        assert (x0, x1, x2) == (0.0, 1.0, 2.0)
        session.finalize()

    def test_coordinate_explicit(self, tmp_log_dir):
        session = _make_offline_session(tmp_log_dir)
        x = session.resolve_x("loss", 42.0)
        assert x == 42.0
        session.finalize()


# ---------------------------------------------------------------------------
# OfflineSession – cached shim properties
# ---------------------------------------------------------------------------


class TestOfflineSessionShimCaching:
    def test_metrics_api_is_cached(self, tmp_log_dir):
        session = _make_offline_session(tmp_log_dir)
        api1 = session.metrics_api
        api2 = session.metrics_api
        assert api1 is api2
        session.finalize()

    def test_media_api_is_cached(self, tmp_log_dir):
        session = _make_offline_session(tmp_log_dir)
        assert session.media_api is session.media_api
        session.finalize()

    def test_artifacts_api_is_cached(self, tmp_log_dir):
        session = _make_offline_session(tmp_log_dir)
        assert session.artifacts_api is session.artifacts_api
        session.finalize()

    def test_teamspace_is_cached(self, tmp_log_dir):
        session = _make_offline_session(tmp_log_dir)
        assert session.teamspace is session.teamspace
        session.finalize()

    def test_teamspace_owner_is_proper_object(self, tmp_log_dir):
        session = _make_offline_session(tmp_log_dir)
        ts = session.teamspace
        assert hasattr(ts.owner, "name")
        assert hasattr(ts.owner, "id")
        assert ts.owner.name == "offline"
        assert ts.owner.id == "offline"
        session.finalize()


# ---------------------------------------------------------------------------
# OfflineSession – resume
# ---------------------------------------------------------------------------


class TestOfflineSessionResume:
    def test_resume_loads_last_x(self, tmp_log_dir):
        s1 = _make_offline_session(tmp_log_dir, name="resume-test")
        s1.write_metric("loss", 1.0, x=None, created_at=None)
        s1.write_metric("loss", 0.5, x=None, created_at=None)
        s1.finalize()

        s2 = _make_offline_session(tmp_log_dir, name="resume-test")
        assert s2.last_x["loss"] == 1.0  # 0-based: two writes → last is 1.0
        next_x = s2.resolve_x("loss", None)
        assert next_x == 2.0
        s2.finalize()


# ---------------------------------------------------------------------------
# Experiment with mode="offline"
# ---------------------------------------------------------------------------


class TestExperimentOfflineMode:
    def test_init_offline_creates_db(self, tmp_path):
        from litlogger.experiment import Experiment

        log_dir = str(tmp_path / "logs" / "run1")
        exp = Experiment(name="offline-test", log_dir=log_dir, mode="offline", verbose=False)
        assert exp.mode == "offline"
        assert os.path.exists(os.path.join(log_dir, "offline.db"))
        exp.finalize()

    def test_append_metrics_offline(self, tmp_path):
        from litlogger.experiment import Experiment

        log_dir = str(tmp_path / "logs" / "run2")
        exp = Experiment(name="metric-test", log_dir=log_dir, mode="offline", verbose=False)
        exp["loss"].append(1.0)
        exp["loss"].append(0.5)
        exp["loss"].append(0.1)
        assert list(exp["loss"]) == [1.0, 0.5, 0.1]
        exp.finalize()

        # Verify in SQLite
        conn = sqlite3.connect(os.path.join(log_dir, "offline.db"))
        rows = conn.execute("SELECT value FROM metrics WHERE key='loss' ORDER BY id").fetchall()
        assert [r[0] for r in rows] == [1.0, 0.5, 0.1]
        conn.close()

    def test_set_metadata_offline(self, tmp_path):
        from litlogger.experiment import Experiment

        log_dir = str(tmp_path / "logs" / "run3")
        exp = Experiment(name="meta-test", log_dir=log_dir, mode="offline", verbose=False)
        exp["optimizer"] = "sgd"
        exp["lr"] = "0.01"
        exp.finalize()

        conn = sqlite3.connect(os.path.join(log_dir, "offline.db"))
        rows = conn.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
        assert dict(rows) == {"lr": "0.01", "optimizer": "sgd"}
        conn.close()

    def test_extend_metrics_offline(self, tmp_path):
        from litlogger.experiment import Experiment

        log_dir = str(tmp_path / "logs" / "run4")
        exp = Experiment(name="extend-test", log_dir=log_dir, mode="offline", verbose=False)
        exp["acc"].extend([0.1, 0.5, 0.9])
        assert list(exp["acc"]) == [0.1, 0.5, 0.9]
        exp.finalize()

    def test_update_offline(self, tmp_path):
        from litlogger.experiment import Experiment

        log_dir = str(tmp_path / "logs" / "run5")
        exp = Experiment(name="update-test", log_dir=log_dir, mode="offline", verbose=False)
        exp.update({"loss": 0.5, "lr": "0.001"})
        assert list(exp["loss"]) == [0.5]
        exp.finalize()

        conn = sqlite3.connect(os.path.join(log_dir, "offline.db"))
        meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        assert meta == {"lr": "0.001"}
        conn.close()

    def test_invalid_mode_raises(self, tmp_path):
        from litlogger.experiment import Experiment

        with pytest.raises(ValueError, match="mode must be"):
            Experiment(name="bad", log_dir=str(tmp_path), mode="cloud", verbose=False)

    def test_finalize_is_idempotent(self, tmp_path):
        from litlogger.experiment import Experiment

        log_dir = str(tmp_path / "logs" / "idem")
        exp = Experiment(name="idem-test", log_dir=log_dir, mode="offline", verbose=False)
        exp["loss"].append(1.0)
        exp.finalize()
        exp.finalize()  # second call should not raise

    def test_metrics_property(self, tmp_path):
        from litlogger.experiment import Experiment

        log_dir = str(tmp_path / "logs" / "metrics-prop")
        exp = Experiment(name="mp-test", log_dir=log_dir, mode="offline", verbose=False)
        exp["loss"].append(0.5)
        exp["acc"].append(0.9)
        metrics = exp.metrics
        assert set(metrics.keys()) == {"loss", "acc"}
        exp.finalize()

    def test_metadata_property(self, tmp_path):
        from litlogger.experiment import Experiment

        log_dir = str(tmp_path / "logs" / "metadata-prop")
        exp = Experiment(name="mdp-test", log_dir=log_dir, mode="offline", verbose=False)
        exp["foo"] = "bar"
        md = exp.metadata
        assert md == {"foo": "bar"}
        exp.finalize()

    def test_print_url_offline(self, tmp_path, capsys):
        from litlogger.experiment import Experiment

        log_dir = str(tmp_path / "logs" / "print-url")
        exp = Experiment(name="url-test", log_dir=log_dir, mode="offline", verbose=True)
        exp.print_url()
        # Output goes to stderr via click.echo — just verify it doesn't crash
        exp.finalize()

    def test_artifact_restored_on_resume(self, tmp_path):
        """Artifacts written offline are restored when the experiment resumes."""
        from litlogger.experiment import Experiment
        from litlogger.primitives.file import File

        log_dir = str(tmp_path / "logs" / "art-resume")
        src = tmp_path / "weights.pt"
        src.write_text("weights-data")

        exp1 = Experiment(name="art-resume-test", log_dir=log_dir, mode="offline", verbose=False)
        exp1["model"] = File(str(src))
        exp1.finalize()

        exp2 = Experiment(name="art-resume-test", log_dir=log_dir, mode="offline", verbose=False)
        assert "model" in exp2._key_types
        assert exp2._key_types["model"] == "static_file"
        assert "model" in exp2._static_files
        exp2.finalize()


# ---------------------------------------------------------------------------
# init() with mode="offline"
# ---------------------------------------------------------------------------


class TestInitOffline:
    def test_init_offline_returns_experiment(self, tmp_path, monkeypatch):
        from litlogger.init import init

        log_dir = str(tmp_path / "logs")
        exp = init(
            name="init-offline",
            root_dir=log_dir,
            mode="offline",
            verbose=False,
            print_url=False,
        )
        assert exp.mode == "offline"
        exp["x"].append(42.0)
        exp.finalize()
        assert os.path.exists(os.path.join(log_dir, "init-offline", "offline.db"))


# ---------------------------------------------------------------------------
# sync() validation
# ---------------------------------------------------------------------------


class TestSyncValidation:
    def test_sync_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No offline database"):
            sync(str(tmp_path / "nonexistent"))

    def test_sync_empty_db_raises(self, tmp_path):
        db_path = str(tmp_path / "offline.db")
        conn = _init_db(db_path)
        conn.close()
        with pytest.raises(ValueError, match="no experiment"):
            sync(str(tmp_path))


# ---------------------------------------------------------------------------
# _send_metrics_batched — chunking logic
# ---------------------------------------------------------------------------


class TestSendMetricsBatched:
    def test_single_batch_no_chunking(self):
        """When total values < max_batch_size, sends in one call."""
        mock_session = MagicMock()
        mock_session.teamspace.id = "ts1"
        mock_session.metrics_store.id = "ms1"

        metrics = [Metrics(name="loss", values=[MetricValue(value=float(i)) for i in range(10)])]
        total = _send_metrics_batched(mock_session, metrics, max_batch_size=100)

        assert total == 10
        mock_session.metrics_api.append_experiment_metrics.assert_called_once()

    def test_chunks_across_batch_boundary(self):
        """When total values > max_batch_size, sends in multiple calls."""
        mock_session = MagicMock()
        mock_session.teamspace.id = "ts1"
        mock_session.metrics_store.id = "ms1"

        metrics = [Metrics(name="loss", values=[MetricValue(value=float(i)) for i in range(25)])]
        total = _send_metrics_batched(
            mock_session, metrics, max_batch_size=10, rate_limiting_interval=0,
        )

        assert total == 25
        # 10 + 10 + 5 = 3 calls
        assert mock_session.metrics_api.append_experiment_metrics.call_count == 3

    def test_multiple_series_chunked(self):
        """Multiple series are interleaved correctly in chunks."""
        mock_session = MagicMock()
        mock_session.teamspace.id = "ts1"
        mock_session.metrics_store.id = "ms1"

        metrics = [
            Metrics(name="loss", values=[MetricValue(value=float(i)) for i in range(8)]),
            Metrics(name="acc", values=[MetricValue(value=float(i)) for i in range(8)]),
        ]
        total = _send_metrics_batched(
            mock_session, metrics, max_batch_size=10, rate_limiting_interval=0,
        )

        assert total == 16
        assert mock_session.metrics_api.append_experiment_metrics.call_count == 2

    def test_empty_metrics(self):
        """Empty metrics list produces no API calls."""
        mock_session = MagicMock()
        total = _send_metrics_batched(mock_session, [], max_batch_size=10)
        assert total == 0
        mock_session.metrics_api.append_experiment_metrics.assert_not_called()


# ---------------------------------------------------------------------------
# sync() end-to-end with mocked ExperimentSession
# ---------------------------------------------------------------------------


class TestSyncEndToEnd:
    def _create_offline_experiment(self, log_dir: str) -> None:
        """Write offline data using the full Experiment stack."""
        from litlogger.experiment import Experiment
        from litlogger.primitives.file import File

        exp = Experiment(name="sync-e2e", log_dir=log_dir, mode="offline", verbose=False)
        exp["loss"].append(1.0)
        exp["loss"].append(0.5)
        exp["loss"].append(0.1)
        exp["acc"].extend([0.2, 0.6, 0.9])
        exp["optimizer"] = "adam"
        exp["lr"] = "0.001"
        exp.finalize()

    @patch("litlogger.session.ExperimentSession.__init__", return_value=None)
    def test_sync_replays_all_data(self, mock_init, tmp_path):
        """sync() replays metadata, metrics, and marks stream completed."""
        log_dir = str(tmp_path / "logs" / "sync-e2e")
        self._create_offline_experiment(log_dir)

        # Patch the session instance's attributes after __init__ is bypassed
        with patch("litlogger.session.ExperimentSession") as MockSession:
            mock_instance = MockSession.return_value
            mock_instance.teamspace.id = "ts1"
            mock_instance.metrics_store.id = "ms1"
            mock_instance.metrics_store.tags = []
            mock_instance.url = "https://example.com/exp"

            sync(log_dir, verbose=False)

            # Metadata was replayed
            mock_instance.refresh_metrics_store.assert_called()

            # Metrics were replayed via append_experiment_metrics
            append_calls = mock_instance.metrics_api.append_experiment_metrics.call_args_list
            assert len(append_calls) > 0
            total_values = 0
            for c in append_calls:
                for m in c.kwargs["metrics"]:
                    total_values += len(m.values)
            assert total_values == 6  # 3 loss + 3 acc

            # Stream was marked completed
            mock_instance.metrics_api.update_experiment_metrics.assert_called()
            final_update = mock_instance.metrics_api.update_experiment_metrics.call_args
            assert final_update.kwargs.get("persisted") is True

            # Session was finalized
            mock_instance.finalize.assert_called_once()

    def test_sync_batches_large_experiments(self, tmp_path):
        """sync() chunks metrics when they exceed max_batch_size."""
        log_dir = str(tmp_path / "logs" / "sync-large")

        # Create an experiment with many metrics directly in SQLite
        session = _make_offline_session(log_dir, name="sync-large")
        for i in range(150):
            session.write_metric("loss", float(i), x=None, created_at=None)
        session.finalize()

        with patch("litlogger.session.ExperimentSession") as MockSession:
            mock_instance = MockSession.return_value
            mock_instance.teamspace.id = "ts1"
            mock_instance.metrics_store.id = "ms1"
            mock_instance.url = "https://example.com"

            sync(log_dir, verbose=False, max_batch_size=50, rate_limiting_interval=0)

            # Should have been chunked: 150 / 50 = 3 calls
            assert mock_instance.metrics_api.append_experiment_metrics.call_count == 3


