"""Integration tests for metadata via the standalone API.

Covers both the legacy litlogger.get_metadata() and the new dict-like API.
"""

import tempfile
import uuid
from time import sleep

import litlogger
import pytest
from lightning_sdk.lightning_cloud.openapi.models import LitLoggerServiceDeleteMetricsStreamBody
from litlogger.api.client import LitRestClient
from litlogger.experiment import Experiment

# Suppress deprecation warnings from legacy API usage in integration tests
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.cloud()
def test_metadata_and_tags():
    """Test logging experiments with metadata/tags."""
    experiment_name = f"standalone_metadata_test-{uuid.uuid4().hex}"
    metadata = {
        "model": "ResNet50",
        "dataset": "CIFAR10",
        "optimizer": "Adam",
        "learning_rate": "0.001",
    }

    exp = litlogger.init(
        name=experiment_name,
        teamspace="oss-litlogger",
        metadata=metadata,
    )

    # Log some metrics
    for i in range(5):
        litlogger.log_metrics({"train_loss": i * 0.1}, step=i)

    litlogger.finalize()

    # Verify metadata was stored
    project_id = exp._teamspace.id
    stream_id = exp._metrics_store.id

    client = LitRestClient()
    for _ in range(30):
        response = client.lit_logger_service_list_metrics_streams(project_id=project_id)
        if response.metrics_streams:
            break
        sleep(1)

    # Find our stream
    metrics_stream = next((ms for ms in response.metrics_streams if ms.id == stream_id), None)
    assert metrics_stream is not None

    # Verify metadata
    if hasattr(metrics_stream, "tags") and metrics_stream.tags:
        # Convert tags list to dict
        stream_metadata = {tag.name: tag.value for tag in metrics_stream.tags}
        for key, value in metadata.items():
            assert key in stream_metadata
            assert stream_metadata[key] == value

    # Cleanup
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )


@pytest.mark.cloud()
def test_get_metadata():
    """Test get_metadata() function and experiment.metadata property."""
    experiment_name = f"meta_test-{uuid.uuid4().hex}"
    metadata = {
        "model": "GPT-2",
        "dataset": "WikiText",
        "learning_rate": "0.0001",
        "batch_size": "16",
    }

    exp = litlogger.init(
        name=experiment_name,
        teamspace="oss-litlogger",
        metadata=metadata,
    )

    # Test experiment.metadata property returns the metadata from the metrics stream
    retrieved_metadata = exp.metadata
    assert isinstance(retrieved_metadata, dict)
    for key, value in metadata.items():
        assert key in retrieved_metadata, f"Expected key '{key}' in metadata"
        assert (
            retrieved_metadata[key] == value
        ), f"Expected metadata['{key}'] == '{value}', got '{retrieved_metadata[key]}'"

    # Test litlogger.get_metadata() returns the same metadata
    global_metadata = litlogger.get_metadata()
    assert isinstance(global_metadata, dict)
    for key, value in metadata.items():
        assert key in global_metadata, f"Expected key '{key}' in global metadata"
        assert global_metadata[key] == value

    # Verify both return the same data
    assert retrieved_metadata == global_metadata

    litlogger.finalize()

    # Cleanup
    project_id = exp._teamspace.id
    stream_id = exp._metrics_store.id

    client = LitRestClient()
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )


@pytest.mark.cloud()
def test_get_metadata_empty():
    """Test get_metadata() when no metadata is provided."""
    experiment_name = f"meta_empty_test-{uuid.uuid4().hex}"

    exp = litlogger.init(
        name=experiment_name,
        teamspace="oss-litlogger",
    )

    # Test that metadata is empty dict when none provided
    retrieved_metadata = exp.metadata
    assert isinstance(retrieved_metadata, dict)
    assert len(retrieved_metadata) == 0

    # Test litlogger.get_metadata() also returns empty dict
    global_metadata = litlogger.get_metadata()
    assert isinstance(global_metadata, dict)
    assert len(global_metadata) == 0

    litlogger.finalize()

    # Cleanup
    project_id = exp._teamspace.id
    stream_id = exp._metrics_store.id

    client = LitRestClient()
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )


@pytest.mark.cloud()
def test_get_metadata_direct_experiment():
    """Test experiment.metadata property when using Experiment class directly."""
    experiment_name = f"meta_direct-{uuid.uuid4().hex}"

    metadata = {
        "framework": "PyTorch",
        "version": "2.0",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            name=experiment_name,
            teamspace="oss-litlogger",
            log_dir=tmpdir,
            metadata=metadata,
        )

        # Test experiment.metadata property
        retrieved_metadata = exp.metadata
        assert isinstance(retrieved_metadata, dict)
        for key, value in metadata.items():
            assert key in retrieved_metadata
            assert retrieved_metadata[key] == value

        exp.finalize()

        # Cleanup
        project_id = exp._teamspace.id
        stream_id = exp._metrics_store.id

        client = LitRestClient()
        client.lit_logger_service_delete_metrics_stream(
            project_id=project_id,
            body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
        )


@pytest.mark.cloud()
def test_new_dict_api_metadata():
    """Test the new dict-like API for setting metadata."""
    experiment_name = f"standalone_dict_metadata_test-{uuid.uuid4().hex}"
    exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

    # Set metadata using the new dict API
    exp["model"] = "ResNet50"
    exp["dataset"] = "CIFAR10"

    # Verify local tracking
    assert exp["model"] == "ResNet50"
    assert exp["dataset"] == "CIFAR10"

    # Log a metric to ensure it doesn't conflict
    exp["loss"].append(0.5, step=0)

    litlogger.finalize()

    # Verify metadata was stored in the cloud
    project_id = exp._teamspace.id
    stream_id = exp._metrics_store.id

    client = LitRestClient()
    for _ in range(30):
        response = client.lit_logger_service_list_metrics_streams(project_id=project_id)
        if response.metrics_streams:
            break
        sleep(1)

    metrics_stream = next((ms for ms in response.metrics_streams if ms.id == stream_id), None)
    assert metrics_stream is not None

    if hasattr(metrics_stream, "tags") and metrics_stream.tags:
        stream_metadata = {tag.name: tag.value for tag in metrics_stream.tags}
        assert stream_metadata.get("model") == "ResNet50"
        assert stream_metadata.get("dataset") == "CIFAR10"

    # Cleanup
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )
