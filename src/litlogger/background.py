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
"""Internal thread utilities for buffering and uploading metrics to Lightning Cloud.

This module defines the _ManagerThread which reads metrics from a multiprocessing queue,
persists them locally in a compact binary format, and periodically sends them to the backend.
"""

import queue
from multiprocessing import Queue
from threading import Event, Thread
from time import sleep, time
from typing import Dict, List

from lightning_sdk.lightning_cloud.openapi.rest import ApiException

from litlogger.api.metrics_api import MetricsApi
from litlogger.file_writer import BinaryFileWriter
from litlogger.types import Metrics, MetricsTracker, PhaseType


class _BackgroundThread(Thread):
    """Background worker that drains a queue of metrics and appends them to the metrics store.

    This thread batches values, writes a compact binary file for later upload, and pushes
    data to the Lightning Cloud API with basic rate limiting and batching.

    Args:
        teamspace_id: Project/teamspace identifier in Lightning Cloud.
        metrics_store_id: The metrics store id to append to.
        cloud_account: Cloud account/cluster identifier used for artifact uploads.
        metrics_api: MetricsApi instance used to communicate with the Lightning Cloud backend.
        metrics_queue: Source of metrics produced by the Experiment/Logger process.
        is_ready_event: Event set when the thread finished initialization.
        stop_event: Event that, when set, requests a graceful shutdown.
        done_event: Event set once all pending metrics have been flushed and the upload completed.
        log_dir: Local directory where temporary metric files are written.
        version: Experiment version used to name the archive on upload.
        store_step: Whether to persist the step field with each value.
        store_created_at: Whether to persist the timestamp for each value.
        rate_limiting_interval: Minimum seconds between consecutive network sends.
        max_batch_size: Number of metric values to accumulate before sending a batch.
    """

    def __init__(
        self,
        teamspace_id: str,
        metrics_store_id: str,
        cloud_account: str,
        metrics_api: MetricsApi,
        metrics_queue: Queue,
        is_ready_event: Event,
        stop_event: Event,
        done_event: Event,
        log_dir: str,
        version: str,
        store_step: bool,
        store_created_at: bool,
        rate_limiting_interval: int = 1,
        max_batch_size: int = 1000,
    ) -> None:
        super().__init__(daemon=True)
        self.teamspace_id = teamspace_id
        self.metrics_store_id = metrics_store_id
        self.metrics_api = metrics_api
        self.metrics_queue = metrics_queue
        self.last_time = time()
        self.rate_limiting_interval = rate_limiting_interval
        self.max_batch_size = max_batch_size
        self.is_ready_event = is_ready_event
        self.stop_event = stop_event
        self.done_event = done_event
        self.metrics: Dict[str, Metrics] = {}
        self.exception = None

        self.store_step = store_step
        self.store_created_at = store_created_at

        self.file_store = BinaryFileWriter(
            log_dir=log_dir,
            version=version,
            store_step=store_step,
            store_created_at=store_created_at,
            teamspace_id=teamspace_id,
            metrics_store_id=metrics_store_id,
            cloud_account=cloud_account,
            client=metrics_api.client,
        )

        self.trackers: Dict[str, MetricsTracker] = {}

    def run(self) -> None:
        self._run()
        self.done_event.set()

    def _run(self) -> None:
        """Drive the worker lifecycle: drain queue until stop, flush, upload, and mark stream completed."""
        try:
            self.is_ready_event.set()

            while not self.stop_event.is_set():
                self.step()

            while self.step():
                pass

            self.step()

            # Force send any remaining buffered metrics (regardless of rate limiting)
            self._send()

            self.file_store.upload()

            self.inform_done()

            self.done_event.set()
        except Exception as e:
            print(e)
            self.done_event.set()
            self.exception = e

    def step(self) -> bool:
        """Read all available metrics from queue, batch them, and send when ready.

        Batching strategy:
        - Drain all immediately available items from the queue
        - Send when we have >= 1000 values OR rate_limiting seconds have passed
        - This allows fast bulk loading while still respecting API limits
        """
        read_any = False

        # Drain all immediately available items from the queue
        while True:
            try:
                metrics = self.metrics_queue.get(timeout=0.1)
                read_any = True
                try:
                    for name, values in metrics.items():
                        self._update_tracker(name, values)
                        # Merge with existing metrics for this name
                        if name in self.metrics:
                            self.metrics[name].values.extend(values.values)
                        else:
                            self.metrics[name] = values
                finally:
                    if hasattr(self.metrics_queue, "task_done"):
                        self.metrics_queue.task_done()
            except queue.Empty:
                break

        # Check if we should send
        num_values = sum(len(m.values) for m in self.metrics.values())
        time_since_last = time() - self.last_time

        # Send if we have enough values OR enough time has passed (with some data)
        should_send = num_values >= self.max_batch_size or (
            num_values > 0 and time_since_last >= self.rate_limiting_interval
        )

        if should_send:
            self._send()

        return read_any

    def _send(self) -> None:
        """Persist buffered metrics to disk and send a batch to the backend; clears the buffer."""
        metrics = list(self.metrics.values())

        if not metrics:
            return

        try:
            self.file_store.store(self.metrics, self.trackers)
        except ApiException as ex:
            raise ex

        try:
            self._send_metrics(metrics)
        except ApiException as ex:
            if "not found" in str(ex):
                raise Exception("The metrics stream has been deleted.") from ex
            raise ex

        self.last_time = time()

        self.metrics = {}

    def _update_tracker(self, name: str, values: Metrics) -> None:
        """Create or update the series tracker and refresh stats/counters from incoming values."""
        # Create the tracker if it doesn't exist
        if name not in self.trackers:
            self.trackers[name] = MetricsTracker(name=name, num_rows=0)

        tracker = self.trackers[name]

        # Store the internal start step for this batch (used by BinaryFileWriter)
        if not hasattr(values, "internal_start_step"):
            values.internal_start_step = tracker.num_rows

        # Increment the number of rows
        for value_obj in values.values:
            value = float(value_obj.value)

            if tracker.started_at is None and self.store_created_at and value_obj.created_at:
                tracker.started_at = value_obj.created_at

            if tracker.min_value is None or (tracker.min_value is not None and value < tracker.min_value):
                tracker.min_value = value
                tracker.min_index = tracker.num_rows

            if tracker.max_value is None or (tracker.max_value is not None and value > tracker.max_value):
                tracker.max_value = value
                tracker.max_index = tracker.num_rows

            tracker.last_index = tracker.num_rows
            tracker.last_value = value

            if self.store_created_at and value_obj.created_at:
                tracker.updated_at = value_obj.created_at

            tracker.num_rows += 1

    def _send_metrics(self, metrics: List[Metrics]) -> None:
        """Send metrics to the API, chunking into batches of max_batch_size values per request.

        In normal operation, this should never receive more than max_batch_size values at a time.
        If it does however (for example when importing an offline metrics file),
        we will chunk the values into batches and sleep between requests to respect rate limiting.

        Args:
            metrics: List of metrics to send.
        """
        current_chunk: List[Metrics] = []
        current_count = 0
        chunks_sent = 0

        for metric in metrics:
            values = metric.values
            idx = 0

            while idx < len(values):
                remaining_capacity = self.max_batch_size - current_count
                values_to_add = min(remaining_capacity, len(values) - idx)

                chunk_values = values[idx : idx + values_to_add]
                chunk_metric = Metrics(name=metric.name, values=chunk_values)
                current_chunk.append(chunk_metric)
                current_count += values_to_add
                idx += values_to_add

                if current_count >= self.max_batch_size:
                    # Sleep between chunks to respect rate limiting (skip first chunk)
                    if chunks_sent > 0:
                        sleep(self.rate_limiting_interval)

                    self.metrics_api.append_experiment_metrics(
                        teamspace_id=self.teamspace_id,
                        metrics_store_id=self.metrics_store_id,
                        metrics=current_chunk,
                    )
                    current_chunk = []
                    current_count = 0
                    chunks_sent += 1

        if current_chunk:
            # Sleep before final chunk if we've already sent chunks
            if chunks_sent > 0:
                sleep(self.rate_limiting_interval)

            self.metrics_api.append_experiment_metrics(
                teamspace_id=self.teamspace_id,
                metrics_store_id=self.metrics_store_id,
                metrics=current_chunk,
            )

    def inform_done(self) -> None:
        """Inform the API that metrics collection is complete."""
        self.metrics_api.update_experiment_metrics(
            teamspace_id=self.teamspace_id,
            metrics_store_id=self.metrics_store_id,
            persisted=True,
            phase=PhaseType.COMPLETED,
            trackers=self.trackers,
        )
