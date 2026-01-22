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
"""API layer for metrics and experiment operations."""

import os
from typing import Any, Dict, List

from google.protobuf import timestamp_pb2
from google.protobuf.json_format import MessageToDict
from lightning_sdk.lightning_cloud.openapi import (
    LitLoggerServiceAppendLoggerMetricsBody,
    LitLoggerServiceCreateMetricsStreamBody,
    LitLoggerServiceUpdateMetricsStreamBody,
    V1Metrics,
    V1MetricsTags,
    V1MetricsTracker,
    V1MetricValue,
    V1PhaseType,
    V1SystemInfo,
)

from litlogger.api.client import LitRestClient
from litlogger.colors import _create_colors
from litlogger.diagnostics import collect_system_info
from litlogger.types import Metrics, MetricsTracker, MetricValue, PhaseType


# Translation functions between user-facing models and V1 models
def _to_v1_metric_value(value: MetricValue) -> V1MetricValue:
    """Convert user-facing MetricValue to V1MetricValue."""
    created_at_str = None
    if value.created_at:
        # Convert datetime to ISO format string with timezone
        created_at_str = value.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00"

    # Build kwargs, excluding None values that the API might not accept
    kwargs = {"value": value.value}
    if value.step is not None:
        kwargs["step"] = value.step
    if created_at_str is not None:
        kwargs["created_at"] = created_at_str

    return V1MetricValue(**kwargs)


def _to_v1_metrics(metrics: Metrics) -> V1Metrics:
    """Convert user-facing Metrics to V1Metrics."""
    v1_values = [_to_v1_metric_value(v) for v in metrics.values]
    return V1Metrics(name=metrics.name, values=v1_values)


def _to_v1_metrics_tracker(tracker: MetricsTracker) -> V1MetricsTracker:
    """Convert user-facing MetricsTracker to V1MetricsTracker."""
    started_at_val = None
    if tracker.started_at:
        # Convert datetime to protobuf Timestamp format
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(tracker.started_at)
        started_at_val = MessageToDict(timestamp)

    updated_at_val = None
    if tracker.updated_at:
        # Convert datetime to protobuf Timestamp format
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(tracker.updated_at)
        updated_at_val = MessageToDict(timestamp)

    # Build kwargs, excluding None values that the API might not accept
    kwargs = {
        "name": tracker.name,
        "num_rows": tracker.num_rows,
    }
    if tracker.min_value is not None:
        kwargs["min_value"] = tracker.min_value
    if tracker.max_value is not None:
        kwargs["max_value"] = tracker.max_value
    if tracker.min_index is not None:
        kwargs["min_index"] = tracker.min_index
    if tracker.max_index is not None:
        kwargs["max_index"] = tracker.max_index
    if tracker.last_value is not None:
        kwargs["last_value"] = tracker.last_value
    if tracker.last_index is not None:
        kwargs["last_index"] = tracker.last_index
    if started_at_val is not None:
        kwargs["started_at"] = started_at_val
    if updated_at_val is not None:
        kwargs["updated_at"] = updated_at_val
    if tracker.max_user_step is not None:
        kwargs["max_user_step"] = tracker.max_user_step

    return V1MetricsTracker(**kwargs)


def _from_v1_metrics_tracker(v1_tracker: V1MetricsTracker) -> MetricsTracker:
    """Convert V1MetricsTracker from API response to user-facing MetricsTracker."""
    return MetricsTracker(
        name=v1_tracker.name,
        num_rows=v1_tracker.num_rows or 0,
        min_value=v1_tracker.min_value,
        max_value=v1_tracker.max_value,
        min_index=v1_tracker.min_index,
        max_index=v1_tracker.max_index,
        last_value=v1_tracker.last_value,
        last_index=v1_tracker.last_index,
        max_user_step=v1_tracker.max_user_step,
    )


def _to_v1_phase_type(phase: PhaseType) -> str:
    """Convert user-facing PhaseType to V1PhaseType string.

    Note: V1PhaseType only has RUNNING, COMPLETED, and FAILED.
    We map PENDING and STOPPED to RUNNING and COMPLETED respectively.
    """
    phase_map = {
        PhaseType.PENDING: V1PhaseType.RUNNING,
        PhaseType.RUNNING: V1PhaseType.RUNNING,
        PhaseType.COMPLETED: V1PhaseType.COMPLETED,
        PhaseType.FAILED: V1PhaseType.FAILED,
        PhaseType.STOPPED: V1PhaseType.COMPLETED,
    }
    return phase_map[phase]


class MetricsApi:
    """API layer for metrics and experiment backend operations.

    Handles project resolution, metrics stream creation, and artifact uploads.
    Follows the lightning_sdk pattern of separating API operations from user-facing classes.
    """

    def __init__(self, client: LitRestClient | None = None) -> None:
        """Initialize the MetricsApi.

        Args:
            client: Optional pre-configured LitRestClient. If None, creates a new one.
        """
        self.client = client or LitRestClient(max_retries=5)

    def get_experiment_metrics_by_name(
        self,
        teamspace_id: str,
        name: str,
        version_number: int | None = None,
    ) -> Any | None:
        """Fetch an experiment (metrics stream) by name.

        Args:
            teamspace_id: The teamspace ID.
            name: The experiment name.
            version_number: Optional version number. If not specified, returns the latest version.

        Returns:
            The metrics stream object for the experiment, or None if not found.
        """
        response = self.client.lit_logger_service_list_metrics_streams(project_id=teamspace_id)

        if not response.metrics_streams:
            return None

        # Filter by name
        matching = [ms for ms in response.metrics_streams if ms.name == name]

        if not matching:
            return None

        # If version_number specified, find that specific version
        if version_number is not None:
            for ms in matching:
                if ms.version_number == version_number:
                    return ms
            return None

        # Return the latest version (highest version_number)
        return max(matching, key=lambda ms: ms.version_number)

    def get_or_create_experiment_metrics(
        self,
        teamspace_id: str,
        name: str,
        version: str,
        metadata: Dict[str, str] | None = None,
        light_color: str | None = None,
        dark_color: str | None = None,
        store_step: bool = True,
        store_created_at: bool = False,
    ) -> tuple[Any, bool]:
        """Get an existing experiment by name, or create a new one if it doesn't exist.

        Args:
            teamspace_id: The teamspace ID.
            name: Experiment name.
            version: Experiment version (used only when creating).
            metadata: Optional metadata tags (used only when creating).
            light_color: Light mode color (used only when creating).
            dark_color: Dark mode color (used only when creating).
            store_step: Whether to store step values (used only when creating).
            store_created_at: Whether to store timestamps (used only when creating).

        Returns:
            A tuple of (metrics_store, created) where created is True if a new
            experiment was created, False if an existing one was returned.
        """
        existing = self.get_experiment_metrics_by_name(teamspace_id=teamspace_id, name=name)
        if existing is not None:
            return existing, False

        new_experiment = self.create_experiment_metrics(
            teamspace_id=teamspace_id,
            name=name,
            version=version,
            metadata=metadata,
            light_color=light_color,
            dark_color=dark_color,
            store_step=store_step,
            store_created_at=store_created_at,
        )
        return new_experiment, True

    def create_experiment_metrics(
        self,
        teamspace_id: str,
        name: str,
        version: str,
        metadata: Dict[str, str] | None = None,
        light_color: str | None = None,
        dark_color: str | None = None,
        store_step: bool = True,
        store_created_at: bool = False,
    ) -> Any:
        """Create a metrics store for an experiment.

        Args:
            teamspace_id: The teamspace ID.
            name: Experiment name.
            version: Experiment version.
            metadata: Optional metadata tags.
            light_color: Light mode color.
            dark_color: Dark mode color.
            store_step: Whether to store step values.
            store_created_at: Whether to store timestamps.

        Returns:
            The created metrics store object.
        """
        # Generate colors based on name + version for consistent unique colors per version
        random_light_color, random_dark_color = _create_colors(name, version)

        # Convert metadata to tags
        tags = []
        if metadata:
            tags = [V1MetricsTags(name=str(k), value=str(v), from_code=True, active=False) for k, v in metadata.items()]

        # Create the stream
        cloudspace_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID")
        app_id = os.getenv("LIGHTNING_CLOUD_APP_ID")
        work_id = os.getenv("LIGHTNING_CLOUD_WORK_ID")

        # we're logging to a different teamspace than the one we're logged in to, so we cannot cross-reference
        if teamspace_id != os.getenv("LIGHTNING_CLOUD_PROJECT_ID"):
            app_id = None
            work_id = None
            cloudspace_id = None

        return self.client.lit_logger_service_create_metrics_stream(
            project_id=teamspace_id,
            body=LitLoggerServiceCreateMetricsStreamBody(
                name=name,
                version=version,
                cloudspace_id=cloudspace_id,
                app_id=app_id,
                work_id=work_id,
                light_color=light_color or random_light_color,
                dark_color=dark_color or random_dark_color,
                tags=tags,
                store_step=store_step,
                store_created_at=store_created_at,
                system_info=V1SystemInfo(**collect_system_info()),
            ),
        )

    def append_experiment_metrics(
        self,
        teamspace_id: str,
        metrics_store_id: str,
        metrics: List[Metrics],
    ) -> None:
        """Append metrics to an existing experiment metrics store.

        Args:
            teamspace_id: The teamspace ID.
            metrics_store_id: The metrics store ID.
            metrics: List of metrics to append.
        """
        # Convert user-facing metrics to V1 metrics
        v1_metrics = [_to_v1_metrics(m) for m in metrics]

        self.client.lit_logger_service_append_logger_metrics(
            project_id=teamspace_id,
            id=metrics_store_id,
            body=LitLoggerServiceAppendLoggerMetricsBody(
                metrics=v1_metrics,
            ),
        )

    def update_experiment_metrics(
        self,
        teamspace_id: str,
        metrics_store_id: str,
        persisted: bool = True,
        phase: PhaseType = PhaseType.COMPLETED,
        trackers: Dict[str, MetricsTracker] | None = None,
    ) -> None:
        """Update an experiment metrics store with completion status and trackers.

        Args:
            teamspace_id: The teamspace ID.
            metrics_store_id: The metrics store ID.
            persisted: Whether the metrics have been persisted.
            phase: The phase of the metrics store (e.g., COMPLETED).
            trackers: Optional dictionary of metric trackers.
        """
        # Convert user-facing phase and trackers to V1 types
        v1_phase = _to_v1_phase_type(phase)
        v1_trackers = None
        if trackers:
            v1_trackers = {name: _to_v1_metrics_tracker(tracker) for name, tracker in trackers.items()}

        self.client.lit_logger_service_update_metrics_stream(
            project_id=teamspace_id,
            id=metrics_store_id,
            body=LitLoggerServiceUpdateMetricsStreamBody(
                persisted=persisted,
                phase=v1_phase,
                trackers=v1_trackers,
            ),
        )

    def get_trackers_from_metrics_store(self, metrics_store: Any) -> Dict[str, MetricsTracker] | None:
        """Extract and convert trackers from a metrics store object.

        Args:
            metrics_store: The metrics store object from the API.

        Returns:
            Dictionary of MetricsTracker objects, or None if no trackers exist.
        """
        if not hasattr(metrics_store, "trackers") or not metrics_store.trackers:
            return None

        return {name: _from_v1_metrics_tracker(v1_tracker) for name, v1_tracker in metrics_store.trackers.items()}
