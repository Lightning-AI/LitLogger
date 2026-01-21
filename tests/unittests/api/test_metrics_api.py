# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Unit tests for metrics API."""

from unittest.mock import MagicMock, patch

from lightning_sdk.lightning_cloud.openapi import (
    V1Metrics,
    V1MetricValue,
    V1PhaseType,
)
from litlogger.api.metrics_api import MetricsApi
from litlogger.types import MetricsTracker, PhaseType


class TestMetricsApi:
    """Test the MetricsApi class."""

    def test_init_with_client(self):
        """Test initialization with a custom client."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)
        assert api.client is mock_client

    def test_get_experiment_metrics_by_name(self):
        """Test fetching an experiment by name."""
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.id = "ms-123"
        mock_stream.name = "my-experiment"
        mock_stream.version_number = 1
        mock_response = MagicMock()
        mock_response.metrics_streams = [mock_stream]
        mock_client.lit_logger_service_list_metrics_streams.return_value = mock_response
        api = MetricsApi(client=mock_client)

        result = api.get_experiment_metrics_by_name(
            teamspace_id="ts-123",
            name="my-experiment",
        )

        mock_client.lit_logger_service_list_metrics_streams.assert_called_once_with(
            project_id="ts-123",
        )
        assert result.id == "ms-123"
        assert result.name == "my-experiment"

    def test_get_experiment_metrics_by_name_returns_latest_version(self):
        """Test that get_experiment_metrics_by_name returns the latest version when multiple exist."""
        mock_client = MagicMock()
        mock_stream_v1 = MagicMock()
        mock_stream_v1.id = "ms-123"
        mock_stream_v1.name = "my-experiment"
        mock_stream_v1.version_number = 1
        mock_stream_v2 = MagicMock()
        mock_stream_v2.id = "ms-456"
        mock_stream_v2.name = "my-experiment"
        mock_stream_v2.version_number = 2
        mock_response = MagicMock()
        mock_response.metrics_streams = [mock_stream_v1, mock_stream_v2]
        mock_client.lit_logger_service_list_metrics_streams.return_value = mock_response
        api = MetricsApi(client=mock_client)

        result = api.get_experiment_metrics_by_name(
            teamspace_id="ts-123",
            name="my-experiment",
        )

        assert result.id == "ms-456"
        assert result.version_number == 2

    def test_get_experiment_metrics_by_name_with_version(self):
        """Test fetching a specific version of an experiment by name."""
        mock_client = MagicMock()
        mock_stream_v1 = MagicMock()
        mock_stream_v1.id = "ms-123"
        mock_stream_v1.name = "my-experiment"
        mock_stream_v1.version_number = 1
        mock_stream_v2 = MagicMock()
        mock_stream_v2.id = "ms-456"
        mock_stream_v2.name = "my-experiment"
        mock_stream_v2.version_number = 2
        mock_response = MagicMock()
        mock_response.metrics_streams = [mock_stream_v1, mock_stream_v2]
        mock_client.lit_logger_service_list_metrics_streams.return_value = mock_response
        api = MetricsApi(client=mock_client)

        result = api.get_experiment_metrics_by_name(
            teamspace_id="ts-123",
            name="my-experiment",
            version_number=1,
        )

        assert result.id == "ms-123"
        assert result.version_number == 1

    def test_get_experiment_metrics_by_name_not_found(self):
        """Test that get_experiment_metrics_by_name returns None when experiment not found."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.metrics_streams = []
        mock_client.lit_logger_service_list_metrics_streams.return_value = mock_response
        api = MetricsApi(client=mock_client)

        result = api.get_experiment_metrics_by_name(
            teamspace_id="ts-123",
            name="nonexistent",
        )

        assert result is None

    def test_get_experiment_metrics_by_name_version_not_found(self):
        """Test that get_experiment_metrics_by_name returns None when specific version not found."""
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.id = "ms-123"
        mock_stream.name = "my-experiment"
        mock_stream.version_number = 1
        mock_response = MagicMock()
        mock_response.metrics_streams = [mock_stream]
        mock_client.lit_logger_service_list_metrics_streams.return_value = mock_response
        api = MetricsApi(client=mock_client)

        result = api.get_experiment_metrics_by_name(
            teamspace_id="ts-123",
            name="my-experiment",
            version_number=99,
        )

        assert result is None

    def test_get_or_create_experiment_metrics_creates_new(self):
        """Test get_or_create_experiment_metrics creates a new experiment when none exists."""
        mock_client = MagicMock()
        # First call to list returns empty (no existing experiment)
        mock_list_response = MagicMock()
        mock_list_response.metrics_streams = []
        mock_client.lit_logger_service_list_metrics_streams.return_value = mock_list_response
        # Create returns a new experiment
        mock_created = MagicMock()
        mock_created.id = "ms-new"
        mock_created.name = "my-experiment"
        mock_client.lit_logger_service_create_metrics_stream.return_value = mock_created
        api = MetricsApi(client=mock_client)

        with (
            patch("litlogger.api.metrics_api._create_colors", return_value=("#abc", "#def")),
            patch("litlogger.api.metrics_api.collect_system_info", return_value={}),
        ):
            result, created = api.get_or_create_experiment_metrics(
                teamspace_id="ts-123",
                name="my-experiment",
                version="v1",
            )

        assert created is True
        assert result.id == "ms-new"
        mock_client.lit_logger_service_create_metrics_stream.assert_called_once()

    def test_get_or_create_experiment_metrics_returns_existing(self):
        """Test get_or_create_experiment_metrics returns existing experiment without creating."""
        mock_client = MagicMock()
        # List returns an existing experiment
        mock_existing = MagicMock()
        mock_existing.id = "ms-existing"
        mock_existing.name = "my-experiment"
        mock_existing.version_number = 1
        mock_list_response = MagicMock()
        mock_list_response.metrics_streams = [mock_existing]
        mock_client.lit_logger_service_list_metrics_streams.return_value = mock_list_response
        api = MetricsApi(client=mock_client)

        result, created = api.get_or_create_experiment_metrics(
            teamspace_id="ts-123",
            name="my-experiment",
            version="v1",
        )

        assert created is False
        assert result.id == "ms-existing"
        mock_client.lit_logger_service_create_metrics_stream.assert_not_called()

    def test_init_without_client(self):
        """Test initialization creates a default client."""
        with patch("litlogger.api.metrics_api.LitRestClient") as mock_client_class:
            api = MetricsApi()
            mock_client_class.assert_called_once_with(max_retries=5)
            assert api.client == mock_client_class.return_value

    def test_create_experiment_metrics_minimal(self):
        """Test creating experiment metrics with minimal parameters."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)

        with (
            patch("litlogger.api.metrics_api._create_colors") as mock_colors,
            patch("litlogger.api.metrics_api.collect_system_info") as mock_sysinfo,
            patch("litlogger.api.metrics_api.V1SystemInfo") as mock_system_info,
        ):
            mock_colors.return_value = ("#abc123", "#def456")
            mock_sysinfo.return_value = {"cpu": "x86_64", "memory": "16GB"}
            mock_system_info.return_value = MagicMock()

            api.create_experiment_metrics(
                teamspace_id="ts-123",
                name="my-experiment",
                version="v1",
            )

            # Verify _create_colors was called with name and version
            mock_colors.assert_called_once_with("my-experiment", "v1")

            # Check that client method was called
            mock_client.lit_logger_service_create_metrics_stream.assert_called_once()
            call_args = mock_client.lit_logger_service_create_metrics_stream.call_args

            # Verify parameters
            assert call_args[1]["body"].name == "my-experiment"
            assert call_args[1]["body"].version == "v1"
            assert call_args[1]["body"].store_step is True
            assert call_args[1]["body"].store_created_at is False

    def test_create_experiment_metrics_with_metadata(self):
        """Test creating experiment metrics with metadata tags."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)

        with (
            patch("litlogger.api.metrics_api._create_colors") as mock_colors,
            patch("litlogger.api.metrics_api.collect_system_info") as mock_sysinfo,
            patch("litlogger.api.metrics_api.V1SystemInfo") as mock_system_info,
        ):
            mock_colors.return_value = ("#abc123", "#def456")
            mock_sysinfo.return_value = {"cpu": "x86_64"}
            mock_system_info.return_value = MagicMock()

            metadata = {"learning_rate": "0.001", "batch_size": "32"}

            api.create_experiment_metrics(
                teamspace_id="ts-123",
                name="my-experiment",
                version="v1",
                metadata=metadata,
            )

            # Check tags were created
            call_args = mock_client.lit_logger_service_create_metrics_stream.call_args
            tags = call_args[1]["body"].tags
            assert len(tags) == 2
            # Tags should be V1MetricsTags objects with the metadata

    def test_create_experiment_metrics_with_colors(self):
        """Test creating experiment metrics with custom colors."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)

        with (
            patch("litlogger.api.metrics_api._create_colors") as mock_colors,
            patch("litlogger.api.metrics_api.collect_system_info") as mock_sysinfo,
        ):
            mock_colors.return_value = ("#random1", "#random2")
            mock_sysinfo.return_value = {}

            api.create_experiment_metrics(
                teamspace_id="ts-123",
                name="my-experiment",
                version="v1",
                light_color="#custom-light",
                dark_color="#custom-dark",
            )

            # Check that custom colors were used
            call_args = mock_client.lit_logger_service_create_metrics_stream.call_args
            assert call_args[1]["body"].light_color == "#custom-light"
            assert call_args[1]["body"].dark_color == "#custom-dark"

    def test_create_experiment_metrics_store_flags(self):
        """Test creating experiment metrics with store_step and store_created_at flags."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)

        with (
            patch("litlogger.api.metrics_api._create_colors") as mock_colors,
            patch("litlogger.api.metrics_api.collect_system_info") as mock_sysinfo,
        ):
            mock_colors.return_value = ("#abc123", "#def456")
            mock_sysinfo.return_value = {}

            api.create_experiment_metrics(
                teamspace_id="ts-123",
                name="my-experiment",
                version="v1",
                store_step=False,
                store_created_at=True,
            )

            # Check that flags were passed correctly
            call_args = mock_client.lit_logger_service_create_metrics_stream.call_args
            assert call_args[1]["body"].store_step is False
            assert call_args[1]["body"].store_created_at is True

    def test_append_experiment_metrics(self):
        """Test appending metrics to an experiment metrics store."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)

        metrics = [
            V1Metrics(
                name="loss",
                values=[
                    V1MetricValue(value=0.5, step=0),
                    V1MetricValue(value=0.3, step=1),
                ],
            ),
            V1Metrics(
                name="accuracy",
                values=[V1MetricValue(value=0.95, step=0)],
            ),
        ]

        api.append_experiment_metrics(
            teamspace_id="ts-123",
            metrics_store_id="ms-456",
            metrics=metrics,
        )

        # Verify client was called correctly
        mock_client.lit_logger_service_append_logger_metrics.assert_called_once()
        call_args = mock_client.lit_logger_service_append_logger_metrics.call_args

        assert call_args.kwargs["project_id"] == "ts-123"
        assert call_args.kwargs["id"] == "ms-456"
        assert call_args.kwargs["body"].metrics == metrics

    def test_append_experiment_metrics_empty_list(self):
        """Test appending empty metrics list."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)

        api.append_experiment_metrics(
            teamspace_id="ts-123",
            metrics_store_id="ms-456",
            metrics=[],
        )

        # Should still call the client
        mock_client.lit_logger_service_append_logger_metrics.assert_called_once()

    def test_update_experiment_metrics_default_params(self):
        """Test updating experiment metrics with default parameters."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)

        api.update_experiment_metrics(
            teamspace_id="ts-123",
            metrics_store_id="ms-456",
        )

        # Verify client was called with defaults
        mock_client.lit_logger_service_update_metrics_stream.assert_called_once()
        call_args = mock_client.lit_logger_service_update_metrics_stream.call_args

        assert call_args.kwargs["project_id"] == "ts-123"
        assert call_args.kwargs["id"] == "ms-456"
        assert call_args.kwargs["body"].persisted is True
        assert call_args.kwargs["body"].phase == V1PhaseType.COMPLETED
        assert call_args.kwargs["body"].trackers is None

    def test_update_experiment_metrics_with_trackers(self):
        """Test updating experiment metrics with trackers."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)

        # Use user-facing types
        trackers = {
            "loss": MetricsTracker(
                name="loss",
                num_rows=100,
                min_value=0.1,
                max_value=1.0,
                last_value=0.3,
            ),
            "accuracy": MetricsTracker(
                name="accuracy",
                num_rows=100,
                min_value=0.8,
                max_value=0.99,
                last_value=0.95,
            ),
        }

        api.update_experiment_metrics(
            teamspace_id="ts-123",
            metrics_store_id="ms-456",
            persisted=True,
            phase=PhaseType.COMPLETED,
            trackers=trackers,
        )

        # Verify client was called correctly
        mock_client.lit_logger_service_update_metrics_stream.assert_called_once()
        call_args = mock_client.lit_logger_service_update_metrics_stream.call_args

        assert call_args.kwargs["project_id"] == "ts-123"
        assert call_args.kwargs["id"] == "ms-456"
        assert call_args.kwargs["body"].persisted is True
        # The API translates PhaseType.COMPLETED to V1PhaseType.COMPLETED
        assert call_args.kwargs["body"].phase == V1PhaseType.COMPLETED
        # Trackers should be translated to V1MetricsTracker
        assert "loss" in call_args.kwargs["body"].trackers
        assert "accuracy" in call_args.kwargs["body"].trackers

    def test_update_experiment_metrics_custom_phase(self):
        """Test updating experiment metrics with custom phase."""
        mock_client = MagicMock()
        api = MetricsApi(client=mock_client)

        api.update_experiment_metrics(
            teamspace_id="ts-123",
            metrics_store_id="ms-456",
            persisted=False,
            phase=PhaseType.RUNNING,
        )

        # Verify custom phase was used (translated to V1PhaseType.RUNNING)
        call_args = mock_client.lit_logger_service_update_metrics_stream.call_args
        assert call_args.kwargs["body"].persisted is False
        assert call_args.kwargs["body"].phase == V1PhaseType.RUNNING
