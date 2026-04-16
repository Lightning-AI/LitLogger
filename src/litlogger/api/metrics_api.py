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

import contextlib
import os
from typing import Any

from lightning_sdk.lightning_cloud.openapi import (
    LitLoggerServiceAppendLoggerMetricsBody,
    LitLoggerServiceCreateMetricsStreamBody,
    LitLoggerServiceUpdateMetricsStreamBody,
    V1Metrics,
    V1MetricsTags,
    V1MetricValue,
    V1PhaseType,
    V1SystemInfo,
)
from lightning_sdk.lightning_cloud.openapi.rest import ApiException

from litlogger.api.client import LitRestClient
from litlogger.colors import _create_colors
from litlogger.diagnostics import collect_system_info
from litlogger.types import Metrics, MetricValue, PhaseType


# Translation functions between user-facing models and V1 models
def _to_v1_metric_value(value: MetricValue) -> V1MetricValue:
    """Convert user-facing MetricValue to V1MetricValue."""
    created_at_str = None
    if value.created_at:
        # Convert datetime to ISO format string with timezone
        created_at_str = value.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00"

    # Build kwargs, excluding None values that the API might not accept
    kwargs: dict[str, float | int | str] = {"value": value.value}
    if value.step is not None:
        kwargs["step"] = value.step
    if created_at_str is not None:
        kwargs["created_at"] = created_at_str

    return V1MetricValue(**kwargs)


def _to_v1_metrics(metrics: Metrics) -> V1Metrics:
    """Convert user-facing Metrics to V1Metrics."""
    v1_values = [_to_v1_metric_value(v) for v in metrics.values]
    return V1Metrics(name=metrics.name, values=v1_values)


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
    return str(phase_map[phase])


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
    ) -> Any | None:
        """Fetch an experiment (metrics stream) by name.

        Args:
            teamspace_id: The teamspace ID.
            name: The experiment name.

        Returns:
            The metrics stream object for the experiment, or None if not found.
        """
        try:
            return self.client.lit_logger_service_get_metrics_stream(project_id=teamspace_id, name=name)
        except ApiException as ex:
            if ex.status == 404:
                return None
            raise

    def get_or_create_experiment_metrics(
        self,
        teamspace_id: str,
        name: str,
        metadata: dict[str, str] | None = None,
        light_color: str | None = None,
        dark_color: str | None = None,
        store_step: bool = True,
        store_created_at: bool = False,
    ) -> tuple[Any, bool]:
        """Get an existing experiment by name, or create a new one if it doesn't exist.

        Args:
            teamspace_id: The teamspace ID.
            name: Experiment name.
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
        metadata: dict[str, str] | None = None,
        light_color: str | None = None,
        dark_color: str | None = None,
        store_step: bool = True,
        store_created_at: bool = False,
    ) -> Any:
        """Create a metrics store for an experiment.

        Args:
            teamspace_id: The teamspace ID.
            name: Experiment name.
            metadata: Optional metadata tags.
            light_color: Light mode color.
            dark_color: Dark mode color.
            store_step: Whether to store step values.
            store_created_at: Whether to store timestamps.

        Returns:
            The created metrics store object.
        """
        # Generate colors based on name for consistent unique colors
        random_light_color, random_dark_color = _create_colors(name)

        # Convert metadata to tags
        tags = self._metadata_to_tags(metadata=metadata)

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
        metrics: list[Metrics],
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
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Update an experiment metrics store with completion status, and/or metadata.

        When ``metadata`` is ``None`` (the default), existing tags on the server are
        left untouched.  Pass an explicit dict to replace the tags.

        Args:
            teamspace_id: The teamspace ID.
            metrics_store_id: The metrics store ID.
            persisted: Whether the metrics have been persisted.
            phase: The phase of the metrics store (e.g., COMPLETED).
            metadata: Optional metadata to attach to the experiment. If None,
                existing tags are preserved.
        """
        # Convert user-facing phase to V1 types
        v1_phase = _to_v1_phase_type(phase)

        self.client.lit_logger_service_update_metrics_stream(
            project_id=teamspace_id,
            id=metrics_store_id,
            body=LitLoggerServiceUpdateMetricsStreamBody(
                persisted=persisted,
                phase=v1_phase,
                tags=self._metadata_to_tags(metadata=metadata) if metadata is not None else None,
            ),
        )

    def get_last_steps(self, teamspace_id: str, metrics_stream_id: str) -> dict[str, int] | None:
        """Get the last logged step for each metric in the metrics store.

        Args:
            teamspace_id: The teamspace ID.
            metrics_stream_id: The metrics stream ID.

        Returns:
            A dictionary mapping metric names to their last logged step, or None if no metrics are found
        """
        response = self.client.lit_logger_service_get_logger_metrics_summary(
            project_id=teamspace_id,
            ids=[metrics_stream_id],
        )

        if not response.summaries_per_name:
            return {}

        result = {}
        for name, s in response.summaries_per_name.items():
            last_step = s.summaries_per_id[metrics_stream_id].last_step
            with contextlib.suppress(TypeError, ValueError):
                result[name] = int(last_step)
        return result

    @staticmethod
    def _metadata_to_tags(metadata: dict[str, Any] | None) -> list[V1MetricsTags]:
        """Convert a metadata dictionary to a list of V1MetricsTags.

        Args:
            metadata: Dictionary of metadata key-value pairs, or None.

        Returns:
            List of V1MetricsTags with ``from_code=True`` and ``active=True``,
            or an empty list if metadata is None/empty.
        """
        tags = []
        if metadata:
            tags = [V1MetricsTags(name=str(k), value=str(v), from_code=True, active=True) for k, v in metadata.items()]

        return tags
