# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline storage backend for litlogger.

Provides a local-only session that writes all experiment data to a SQLite
database instead of making network calls.  The ``sync()`` function replays
the stored data through the normal online session when connectivity is
available.
"""

from __future__ import annotations

import os
import queue
import shutil
import sqlite3
import threading
import weakref
from datetime import datetime
from time import sleep
from typing import TYPE_CHECKING, Any

from litlogger.primitives.primitive import MetricWrite, PrimitiveWrite, QueueItem
from litlogger.printer import Printer, RunStats
from litlogger.types import Metrics, MetricValue, PhaseType

if TYPE_CHECKING:
    from litlogger.experiment import Experiment

# Default batch size used by the online session and reused during sync.
_DEFAULT_MAX_BATCH_SIZE = 1000

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS experiment (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    store_step  INTEGER NOT NULL DEFAULT 1,
    store_created_at INTEGER NOT NULL DEFAULT 0,
    finalized   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT    NOT NULL,
    value      REAL    NOT NULL,
    x          REAL,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT    NOT NULL,
    local_path  TEXT    NOT NULL,
    remote_path TEXT,
    kind        TEXT    NOT NULL DEFAULT 'static',
    series_index INTEGER
);

CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(key);
CREATE INDEX IF NOT EXISTS idx_artifacts_key ON artifacts(key);
"""


def _init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the offline SQLite database and apply the schema."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript(_SCHEMA)
    return conn


def _ts_to_real(dt: datetime | None) -> float | None:
    """Convert a datetime to a Unix timestamp for REAL storage."""
    return dt.timestamp() if dt is not None else None


def _real_to_ts(value: float | None) -> datetime | None:
    """Convert a REAL Unix timestamp back to a datetime."""
    return datetime.fromtimestamp(value) if value is not None else None


# ---------------------------------------------------------------------------
# Shim API objects – allow primitives to call session.metrics_api.* etc.
# ---------------------------------------------------------------------------


class _OfflineTag:
    """Mimics a tag object with ``name``, ``value``, and ``from_code`` attributes."""

    __slots__ = ("name", "value", "from_code")

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value
        self.from_code = True


class _OfflineOwner:
    """Nested owner object for :class:`_OfflineTeamspace`."""

    __slots__ = ("name", "id")

    def __init__(self) -> None:
        self.name: str = "offline"
        self.id: str = "offline"


class _OfflineTeamspace:
    """Minimal teamspace stand-in so code that reads ``session.teamspace.id`` doesn't crash."""

    __slots__ = ("id", "name", "owner")

    def __init__(self) -> None:
        self.id: str = "offline"
        self.name: str = "offline"
        self.owner = _OfflineOwner()


class _OfflineMetricsStore:
    """Mimics the metrics store object that primitives read ``id`` and ``tags`` from."""

    __slots__ = ("id", "name", "_session", "cluster_id", "artifacts")

    def __init__(self, experiment_id: int, name: str, session: OfflineSession | None = None) -> None:
        self.id: int = experiment_id
        self.name: str = name
        self._session: OfflineSession | None = session
        self.cluster_id: str | None = None
        self.artifacts: list[Any] = []

    @property
    def tags(self) -> list[_OfflineTag]:
        """Read tags from the SQLite database."""
        if self._session is None:
            return []
        return [_OfflineTag(k, v) for k, v in self._session.read_all_metadata().items()]


class _OfflineMetricsApi:
    """Redirects metrics API calls to the SQLite backend."""

    __slots__ = ("_session",)

    def __init__(self, session: OfflineSession) -> None:
        self._session = session

    def append_experiment_metrics(
        self,
        *,
        teamspace_id: str,
        metrics_store_id: str | int,
        metrics: list[Metrics],
    ) -> None:
        for m in metrics:
            for v in m.values:
                self._session.write_metric(m.name, v.value, v.x, v.created_at)

    def update_experiment_metrics(
        self,
        *,
        teamspace_id: str | None = None,
        metrics_store_id: str | int | None = None,
        phase: PhaseType | None = None,
        metadata: dict[str, str] | None = None,
        persisted: bool | None = None,
    ) -> None:
        if metadata is not None:
            for k, v in metadata.items():
                self._session.write_metadata(k, v)

    def get_or_create_experiment_metrics(
        self,
        *,
        teamspace_id: str,
        name: str,
        metadata: dict[str, str] | None = None,
        light_color: str | None = None,
        dark_color: str | None = None,
        store_step: bool = True,
        store_created_at: bool = False,
    ) -> tuple[_OfflineMetricsStore, bool]:
        return self._session.metrics_store, self._session.created

    def get_experiment_metrics_by_name(self, teamspace_id: str, name: str) -> _OfflineMetricsStore:
        return self._session.metrics_store

    def get_last_steps(self, teamspace_id: str, metrics_store_id: str | int) -> dict[str, float] | None:
        return dict(self._session.last_x) or None

    def get_metric_values(self, teamspace_id: str, metrics_store_id: str | int) -> dict[str, list[float]]:
        raw = self._session.read_all_metrics()
        return {key: [e["value"] for e in entries] for key, entries in raw.items()}


class _OfflineMediaApi:
    """Redirects media API calls to local artifact storage."""

    __slots__ = ("_session",)

    def __init__(self, session: OfflineSession) -> None:
        self._session = session

    def upload_media(
        self,
        *,
        experiment_id: str | int,
        teamspace: Any,
        file_path: str,
        name: str,
        media_type: Any = None,
        step: float | None = None,
        epoch: int | None = None,
        caption: str | None = None,
    ) -> None:
        self._session.write_artifact(name, file_path, kind="media")

    def list_media(self, teamspace_id: str, metrics_store_id: str | int) -> list[Any]:
        return []


class _OfflineArtifactsApi:
    """Redirects artifact API calls to local file storage."""

    __slots__ = ("_session",)

    def __init__(self, session: OfflineSession) -> None:
        self._session = session

    def upload_experiment_file_artifact(
        self,
        *,
        teamspace: Any,
        metrics_store: Any,
        experiment_name: str,
        file_path: str,
        remote_path: str,
    ) -> None:
        self._session.write_artifact(remote_path, file_path, kind="static")

    def download_file(
        self,
        *,
        teamspace: Any,
        remote_path: str,
        local_path: str,
        cloud_account: str | None = None,
    ) -> str:
        artifacts = self._session.read_all_artifacts()
        for art in artifacts:
            if art["remote_path"] == remote_path or art["key"] == remote_path:
                src = art["local_path"]
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
                    shutil.copy2(src, local_path)
                    return local_path
        raise FileNotFoundError(f"Artifact {remote_path!r} not found in offline storage.")

    def list_experiment_artifacts(self, teamspace_id: str, metrics_store_id: str | int) -> list[Any] | None:
        return None


# ---------------------------------------------------------------------------
# Offline session – drop-in replacement for ExperimentSession
# ---------------------------------------------------------------------------


class OfflineSession:
    """A session that persists all writes to a local SQLite database.

    It exposes the same attributes and methods that ``Experiment`` and the
    primitive layer rely on (``submit``, ``flush``, ``finalize``,
    ``resolve_x``, ``stats``, ``printer``, etc.) but never touches the
    network.
    """

    def __init__(
        self,
        name: str,
        log_dir: str,
        *,
        store_step: bool = True,
        store_created_at: bool = False,
        verbose: bool = True,
        experiment: Experiment | None = None,
    ) -> None:
        self.name = name
        self.store_step = store_step
        self.store_created_at = store_created_at
        self.printer = Printer(verbose=verbose)
        self.stats = RunStats()
        self._experiment_ref = weakref.ref(experiment) if experiment is not None else None

        self._log_dir = log_dir
        self._db_path = os.path.join(log_dir, "offline.db")
        self._artifacts_dir = os.path.join(log_dir, "artifacts")
        os.makedirs(self._artifacts_dir, exist_ok=True)

        self._conn = _init_db(self._db_path)
        self._db_lock = threading.Lock()

        # Check if experiment row exists (resume)
        row = self._conn.execute("SELECT id, finalized FROM experiment WHERE name = ?", (name,)).fetchone()
        if row is not None:
            self._experiment_id: int = row[0]
            self.created = False
        else:
            cur = self._conn.execute(
                "INSERT INTO experiment (name, created_at, store_step, store_created_at) VALUES (?, ?, ?, ?)",
                (name, datetime.now().timestamp(), int(store_step), int(store_created_at)),
            )
            self._experiment_id = cur.lastrowid  # type: ignore[assignment]
            self._conn.commit()
            self.created = True

        # Coordinate tracking
        self.last_x: dict[str, float] = {}
        self._coordinate_lock = threading.Lock()
        self._load_last_x()

        # Cached shim API objects (avoid re-allocation on every property access)
        self._metrics_api = _OfflineMetricsApi(self)
        self._media_api = _OfflineMediaApi(self)
        self._artifacts_api = _OfflineArtifactsApi(self)
        self._teamspace = _OfflineTeamspace()

        # Queue / submission machinery
        self.queue: queue.Queue[QueueItem] = queue.Queue()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.done_event = threading.Event()
        self._submission_lock = threading.Lock()
        self._accepting = True
        self._failure: Exception | None = None
        self._finalized = False

        # Background thread for draining the queue
        self.background = _OfflineBackgroundThread(session=self)
        self.background.start()
        self.ready_event.wait()

        # Provide stub attributes that Experiment / legacy code may read
        self.url = f"(offline) {self._db_path}"
        self.accessible_url = self.url
        self.auth_api = None

    # ---- coordinate helpers ----

    def _load_last_x(self) -> None:
        """Restore per-series last-x from persisted metrics."""
        rows = self._conn.execute(
            "SELECT key, MAX(x) FROM metrics WHERE x IS NOT NULL GROUP BY key"
        ).fetchall()
        for key, max_x in rows:
            if max_x is not None:
                self.last_x[key] = max_x

    def resolve_x(self, key: str, x: float | None) -> float:
        """Resolve and record an explicit or auto-incremented x-coordinate."""
        with self._coordinate_lock:
            resolved = self.last_x.get(key, -1) + 1 if x is None else x
            self.last_x[key] = resolved
            return resolved

    # ---- write helpers (called by background thread) ----

    def write_metric(self, key: str, value: float, x: float | None, created_at: datetime | None) -> None:
        """Write a single metric observation to the local database."""
        resolved_x = self.resolve_x(key, x)
        ts = _ts_to_real(created_at)
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO metrics (key, value, x, created_at) VALUES (?, ?, ?, ?)",
                (key, value, resolved_x if self.store_step else None, ts),
            )
            self._conn.commit()

    def write_metadata(self, key: str, value: str) -> None:
        """Upsert one metadata entry."""
        with self._db_lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )
            self._conn.commit()

    def write_artifact(
        self,
        key: str,
        local_path: str,
        *,
        kind: str = "static",
        series_index: int | None = None,
    ) -> None:
        """Copy the file into the offline artifacts directory and record it."""
        dest_dir = os.path.join(self._artifacts_dir, key)
        os.makedirs(dest_dir, exist_ok=True)
        basename = os.path.basename(local_path)
        if series_index is not None:
            basename = f"{series_index}_{basename}"
        dest = os.path.join(dest_dir, basename)
        if os.path.isdir(local_path):
            shutil.copytree(local_path, dest, dirs_exist_ok=True)
        elif os.path.exists(local_path):
            shutil.copy2(local_path, dest)
        else:
            dest = local_path  # path doesn't exist yet – store reference
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO artifacts (key, local_path, remote_path, kind, series_index) VALUES (?, ?, ?, ?, ?)",
                (key, dest, key, kind, series_index),
            )
            self._conn.commit()

    # ---- Shim API properties (cached) ----

    @property
    def metrics_api(self) -> _OfflineMetricsApi:
        return self._metrics_api

    @property
    def media_api(self) -> _OfflineMediaApi:
        return self._media_api

    @property
    def artifacts_api(self) -> _OfflineArtifactsApi:
        return self._artifacts_api

    @property
    def metrics_store(self) -> _OfflineMetricsStore:
        return _OfflineMetricsStore(self._experiment_id, self.name, session=self)

    @property
    def teamspace(self) -> _OfflineTeamspace:
        return self._teamspace

    @property
    def client(self) -> None:
        return None

    # ---- Session protocol ----

    @property
    def experiment_link(self) -> Any:
        """Weak owning experiment reference for SDK model linkage."""
        if self._experiment_ref is None:
            return self
        experiment = self._experiment_ref()
        return experiment if experiment is not None else self

    @property
    def experiment_name(self) -> str:
        """Experiment name used for remote paths and model names."""
        return self.name

    @property
    def last_steps(self) -> dict[str, float]:
        """Legacy alias for the per-series last-x mapping."""
        return self.last_x

    @property
    def last_steps_lock(self) -> threading.Lock:
        """Legacy alias for the coordinate lock."""
        return self._coordinate_lock

    @property
    def metrics_store_id(self) -> str:
        """Identifier of the offline metrics stream."""
        return str(self._experiment_id)

    @property
    def teamspace_id(self) -> str:
        """Identifier of the owning teamspace (always ``'offline'``)."""
        return "offline"

    def submit(self, item: QueueItem) -> None:
        """Atomically reject failed/closed sessions or enqueue one command."""
        with self._submission_lock:
            self._raise_if_failed()
            if not self._accepting:
                raise RuntimeError("The offline session is no longer accepting writes.")
            self.queue.put(item)

    def _raise_if_failed(self) -> None:
        failure = self._failure or self.background.exception
        if failure is not None:
            raise failure

    def raise_if_background_failed(self) -> None:
        """Raise the background worker's captured exception, if any."""
        self._raise_if_failed()

    def flush(self) -> None:
        """Flush submitted commands, then surface failures."""
        self.queue.join()
        self._raise_if_failed()

    def refresh_metrics_store(self) -> None:
        """No-op: no remote store to refresh in offline mode."""

    def finalize(self) -> None:
        """Close submission and finish the background worker exactly once."""
        if self._finalized:
            return
        with self._submission_lock:
            self._raise_if_failed()
            self._accepting = False
        self.queue.join()
        self._raise_if_failed()
        self.stop_event.set()
        self.done_event.wait()
        self._raise_if_failed()
        with self._db_lock:
            self._conn.execute(
                "UPDATE experiment SET finalized = 1 WHERE id = ?", (self._experiment_id,)
            )
            self._conn.commit()
            self._conn.close()
        self._finalized = True

    # ---- Reading helpers (for state rebuild / sync) ----

    def read_all_metadata(self) -> dict[str, str]:
        """Return all metadata entries as a dict."""
        with self._db_lock:
            rows = self._conn.execute("SELECT key, value FROM metadata").fetchall()
        return dict(rows)

    def read_all_metrics(self) -> dict[str, list[dict[str, Any]]]:
        """Return all metric observations grouped by key, ordered by insertion."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT key, value, x, created_at FROM metrics ORDER BY id"
            ).fetchall()
        result: dict[str, list[dict[str, Any]]] = {}
        for key, value, x, created_at in rows:
            result.setdefault(key, []).append(
                {"value": value, "x": x, "created_at": created_at}
            )
        return result

    def read_all_artifacts(self) -> list[dict[str, Any]]:
        """Return all artifact records ordered by insertion."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT key, local_path, remote_path, kind, series_index FROM artifacts ORDER BY id"
            ).fetchall()
        return [
            {
                "key": key,
                "local_path": local_path,
                "remote_path": remote_path,
                "kind": kind,
                "series_index": series_index,
            }
            for key, local_path, remote_path, kind, series_index in rows
        ]


# ---------------------------------------------------------------------------
# Background thread for offline mode
# ---------------------------------------------------------------------------


class _OfflineBackgroundThread(threading.Thread):
    """Drains the offline session queue, writing to SQLite."""

    def __init__(self, session: OfflineSession) -> None:
        super().__init__(daemon=True)
        self.session = session
        self.exception: Exception | None = None

    def run(self) -> None:
        try:
            self._run()
        finally:
            self.session.done_event.set()

    def _run(self) -> None:
        self.session.ready_event.set()
        while not self.session.stop_event.is_set():
            self._step()
        # Drain remaining items
        while self._step():
            pass
        self._step()

    def _step(self) -> bool:
        read_any = False
        while True:
            try:
                item = self.session.queue.get(timeout=0.05)
                read_any = True
                try:
                    if isinstance(item, PrimitiveWrite):
                        item.execute(self.session)
                    elif isinstance(item, MetricWrite):
                        self.session.write_metric(
                            item.key, item.y, item.x, item.created_at,
                        )
                        self.session.stats.record_metric(item.key, item.y)
                    else:
                        raise TypeError(f"Unsupported queue item: {type(item).__name__}")
                finally:
                    self.session.queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                self.exception = e
                with self.session._submission_lock:
                    self.session._failure = e
                    self.session._accepting = False
                break
        return read_any

    def flush_metrics(self) -> None:
        """No-op: metrics are written immediately in offline mode."""


# ---------------------------------------------------------------------------
# Sync: replay offline data to the cloud (batched)
# ---------------------------------------------------------------------------


def _send_metrics_batched(
    session: Any,
    metrics: list[Metrics],
    *,
    max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    rate_limiting_interval: float = 1.0,
) -> int:
    """Chunk *metrics* into batches of at most *max_batch_size* values and send.

    Mirrors :meth:`_BackgroundThread._send_metrics` so that large offline
    experiments are replayed without hitting API size limits or OOM.

    Returns the total number of metric values sent.
    """
    current_chunk: list[Metrics] = []
    current_count = 0
    chunks_sent = 0
    total_values = 0

    for metric in metrics:
        values = metric.values
        idx = 0
        while idx < len(values):
            remaining_capacity = max_batch_size - current_count
            values_to_add = min(remaining_capacity, len(values) - idx)

            chunk_values = values[idx : idx + values_to_add]
            current_chunk.append(Metrics(name=metric.name, values=chunk_values))
            current_count += values_to_add
            total_values += values_to_add
            idx += values_to_add

            if current_count >= max_batch_size:
                if chunks_sent > 0:
                    sleep(rate_limiting_interval)
                session.metrics_api.append_experiment_metrics(
                    teamspace_id=session.teamspace.id,
                    metrics_store_id=session.metrics_store.id,
                    metrics=current_chunk,
                )
                current_chunk = []
                current_count = 0
                chunks_sent += 1

    if current_chunk:
        if chunks_sent > 0:
            sleep(rate_limiting_interval)
        session.metrics_api.append_experiment_metrics(
            teamspace_id=session.teamspace.id,
            metrics_store_id=session.metrics_store.id,
            metrics=current_chunk,
        )

    return total_values


def sync(
    path: str,
    *,
    teamspace: str | None = None,
    verbose: bool = True,
    max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    rate_limiting_interval: float = 1.0,
) -> None:
    """Upload a previously offline experiment to the Lightning.ai cloud.

    Reads all metrics, metadata, and artifacts from the offline SQLite
    database and replays them through the normal online session.  Metrics
    are chunked into batches of *max_batch_size* values per API request to
    avoid size limits and memory pressure on large experiments.

    Args:
        path: Path to the offline experiment directory (the ``log_dir``
            that was passed to ``init``).  Must contain an ``offline.db``
            file.
        teamspace: Optional teamspace override for the upload target.
        verbose: Whether to print progress information.
        max_batch_size: Maximum metric values per API request.
        rate_limiting_interval: Minimum seconds between chunked requests.

    Raises:
        FileNotFoundError: If no offline database is found at *path*.

    Example::

        import litlogger

        # After training offline:
        litlogger.sync("./lightning_logs/my-run")
    """
    from litlogger.session import ExperimentSession

    db_path = os.path.join(path, "offline.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No offline database found at {db_path}")

    conn = sqlite3.connect(db_path)
    printer = Printer(verbose=verbose)

    # Read experiment info
    exp_row = conn.execute(
        "SELECT name, store_step, store_created_at FROM experiment LIMIT 1"
    ).fetchone()
    if exp_row is None:
        conn.close()
        raise ValueError("The offline database contains no experiment.")

    exp_name, store_step, store_created_at = exp_row

    printer.log(f"Syncing offline experiment {printer.name(exp_name)} …")

    # Create an online session
    session = ExperimentSession(
        name=exp_name,
        teamspace=teamspace,
        store_step=bool(store_step),
        store_created_at=bool(store_created_at),
        verbose=verbose,
    )

    # 1) Replay metadata
    metadata_rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    if metadata_rows:
        metadata_dict = dict(metadata_rows)
        from litlogger.primitives.metadata import Metadata

        for key, value in metadata_dict.items():
            Metadata(key, value).log(session)
        printer.log(f"  Synced {len(metadata_dict)} metadata entries")

    # 2) Replay metrics (batched)
    metric_rows = conn.execute(
        "SELECT key, value, x, created_at FROM metrics ORDER BY id"
    ).fetchall()
    if metric_rows:
        batched: dict[str, Metrics] = {}
        for key, value, x, created_at in metric_rows:
            ts = _real_to_ts(created_at)
            mv = MetricValue(value=value, x=x, created_at=ts)
            if key in batched:
                batched[key].values.append(mv)
            else:
                batched[key] = Metrics(name=key, values=[mv])
        total_values = _send_metrics_batched(
            session,
            list(batched.values()),
            max_batch_size=max_batch_size,
            rate_limiting_interval=rate_limiting_interval,
        )
        printer.log(f"  Synced {total_values} metric values across {len(batched)} series")

    # 3) Replay artifacts
    artifact_rows = conn.execute(
        "SELECT key, local_path, remote_path, kind, series_index FROM artifacts ORDER BY id"
    ).fetchall()
    if artifact_rows:
        from litlogger.primitives.file import File

        synced = 0
        for key, local_path, remote_path, kind, series_index in artifact_rows:
            if not os.path.exists(local_path):
                printer.warn(f"  Skipping missing artifact: {local_path}")
                continue
            f = File(local_path)
            f._upload_artifact(session, remote_path=remote_path)
            synced += 1
        printer.log(f"  Synced {synced} artifacts")

    # 4) Mark stream completed
    session.metrics_api.update_experiment_metrics(
        teamspace_id=session.teamspace.id,
        metrics_store_id=session.metrics_store.id,
        persisted=True,
        phase=PhaseType.COMPLETED,
    )

    session.finalize()
    conn.close()

    printer.log(f"Sync complete. View at: {printer.link(session.url)}")
