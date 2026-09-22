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
"""Internal thread utilities for the experiment's write-behind pipeline.

This module defines the background worker which drains the experiment queue:
metric batches are merged, rate-limited, and appended in bulk, while queued
primitives (files, media, models, metadata) are executed one by one.
"""

import queue
from threading import Thread
from time import sleep, time
from typing import TYPE_CHECKING

from lightning_sdk.lightning_cloud.openapi.rest import ApiException

from litlogger.primitives import MetricWrite, PrimitiveWrite
from litlogger.types import Metrics, MetricValue, PhaseType

if TYPE_CHECKING:
    from litlogger.session import ExperimentSession


class _BackgroundThread(Thread):
    """Background worker draining the experiment queue.

    Metric values are batched and pushed to the Lightning Cloud API with basic
    rate limiting; queued primitives perform their own writes via the session.

    Args:
        session: Owning session containing the queue, APIs, events, and state.
        rate_limiting_interval: Minimum seconds between consecutive network sends.
        max_batch_size: Number of metric values to accumulate before sending a batch.
    """

    def __init__(
        self,
        session: "ExperimentSession",
        rate_limiting_interval: int = 1,
        max_batch_size: int = 1000,
    ) -> None:
        super().__init__(daemon=True)
        self.session = session
        self.teamspace_id = str(session.teamspace.id)
        self.metrics_store_id = str(session.metrics_store.id)
        self.metrics_api = session.metrics_api
        self.metrics_queue = session.queue
        self.last_time = time()
        self.rate_limiting_interval = rate_limiting_interval
        self.max_batch_size = max_batch_size
        self.is_ready_event = session.ready_event
        self.stop_event = session.stop_event
        self.done_event = session.done_event
        self.metrics: dict[str, Metrics] = {}
        self.exception: Exception | None = None
        self.last_x = session.last_x

        self.store_step = session.store_step
        self.store_created_at = session.store_created_at

    @property
    def last_steps(self) -> dict[str, float]:
        """Legacy alias for the session-owned last-x mapping."""
        return self.last_x

    def run(self) -> None:
        try:
            self._run()
        finally:
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

            self.inform_done()

        except Exception as e:
            print(e)
            self.exception = e
            self.session._record_background_failure(e)

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
                item = self.metrics_queue.get(timeout=0.1)
                read_any = True
                try:
                    if isinstance(item, PrimitiveWrite):
                        item.execute(self.session)
                    elif isinstance(item, MetricWrite):
                        self._collect_metric(item)
                    else:
                        raise TypeError(f"Unsupported queue command: {type(item).__name__}")
                finally:
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

    def _collect_metric(self, item: MetricWrite) -> None:
        """Resolve one coordinate and add its observation to the current batch."""
        x = self.session.resolve_x(item.key, item.x)
        value = MetricValue(
            value=item.y,
            x=x if self.store_step else None,
            created_at=item.created_at,
        )
        if item.key in self.metrics:
            self.metrics[item.key].values.append(value)
        else:
            self.metrics[item.key] = Metrics(name=item.key, values=[value])

    def _send(self) -> None:
        """Persist buffered metrics to disk and send a batch to the backend; clears the buffer."""
        metrics = list(self.metrics.values())

        if not metrics:
            return

        try:
            self._send_metrics(metrics)
        except ApiException as ex:
            if "not found" in str(ex):
                raise Exception("The metrics stream has been deleted.") from ex
            raise ex

        self.last_time = time()

        self.metrics = {}

    def flush_metrics(self) -> None:
        """Force-send the metric observations currently buffered by the worker."""
        self._send()

    def _send_metrics(self, metrics: list[Metrics]) -> None:
        """Send metrics to the API, chunking into batches of max_batch_size values per request.

        In normal operation, this should never receive more than max_batch_size values at a time.
        If it does however (for example when importing an offline metrics file),
        we will chunk the values into batches and sleep between requests to respect rate limiting.

        Args:
            metrics: List of metrics to send.
        """
        current_chunk: list[Metrics] = []
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
        """Inform the API that metrics collection is complete.

        Sends the final update with phase=COMPLETED.
        Metadata is intentionally omitted (None) so that tags set via
        ``Experiment.log_metadata`` are preserved.
        """
        self.metrics_api.update_experiment_metrics(
            teamspace_id=self.teamspace_id,
            metrics_store_id=self.metrics_store_id,
            persisted=True,
            phase=PhaseType.COMPLETED,
        )
