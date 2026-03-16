"""Integration tests for metrics logging via the standalone API.

Covers both the legacy litlogger.log_metrics() and the new dict-like API.
"""

import os
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


def _wait_for_metrics(
    client: LitRestClient,
    *,
    project_id: str,
    stream_id: str,
    expected_counts: dict[str, int],
    attempts: int = 30,
) -> object:
    response = None
    for _ in range(attempts):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        named_metrics = response.named_metrics or {}
        if all(
            len(named_metrics.get(name, {}).ids_metrics.get(stream_id, {}).metrics_values or []) == count
            for name, count in expected_counts.items()
        ):
            return response
        sleep(1)
    return response


@pytest.mark.cloud()
def test_module_level_api_basic():
    """Test basic litlogger.init() and module-level logging."""
    experiment_name = f"standalone_basic_test-{uuid.uuid4().hex}"
    exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

    # Verify global state is set
    assert litlogger.experiment is not None
    assert litlogger.experiment.name == experiment_name
    assert exp is litlogger.experiment

    # Test module-level log functions
    for i in range(10):
        litlogger.log_metrics({"loss": i * 0.1, "accuracy": 1.0 - i * 0.05}, step=i)

    # Alternative logging method
    for i in range(5):
        litlogger.log({"train_loss": i * 0.2}, step=i)

    litlogger.finalize()

    # Verify metrics were logged
    project_id = exp._teamspace.id
    stream_id = exp._metrics_store.id

    client = LitRestClient()
    response = _wait_for_metrics(
        client,
        project_id=project_id,
        stream_id=stream_id,
        expected_counts={"loss": 10, "accuracy": 10, "train_loss": 5},
    )

    # Cleanup
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )

    # Verify all metrics were logged
    assert "loss" in response.named_metrics
    assert "accuracy" in response.named_metrics
    assert "train_loss" in response.named_metrics
    assert len(response.named_metrics["loss"].ids_metrics[stream_id].metrics_values) == 10


@pytest.mark.cloud()
def test_storage_configurations():
    """Test different storage configurations with standalone API."""
    # Test with store_step=False
    exp1 = litlogger.init(
        name=f"standalone_no_step-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
        store_step=False,
    )

    for i in range(5):
        litlogger.log_metrics({"loss": i * 0.1})

    litlogger.finalize()

    # Test with store_created_at=True
    exp2 = litlogger.init(
        name=f"standalone_with_timestamp-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
        store_created_at=True,
    )

    for i in range(5):
        litlogger.log_metrics({"accuracy": i * 0.2}, step=i)

    litlogger.finalize()

    # Cleanup
    client = LitRestClient()
    for exp in [exp1, exp2]:
        project_id = exp._teamspace.id
        stream_id = exp._metrics_store.id

        for _ in range(30):
            response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
            if response.named_metrics != {}:
                break
            sleep(1)

        client.lit_logger_service_delete_metrics_stream(
            project_id=project_id,
            body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
        )

        # Verify metrics exist
        assert response.named_metrics != {}


@pytest.mark.cloud()
def test_new_dict_api_metrics():
    """Test the new dict-like API for logging metrics."""
    experiment_name = f"standalone_dict_api_test-{uuid.uuid4().hex}"
    exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

    # Log metrics using the new dict API with step
    for i in range(10):
        exp["loss"].append(1.0 / (i + 1), step=i)
        exp["accuracy"].append(i / 10.0, step=i)

    # Verify local series tracking
    assert len(exp["loss"]) == 10
    assert len(exp["accuracy"]) == 10
    assert list(exp["loss"])[0] == 1.0
    assert list(exp["accuracy"])[9] == 0.9

    # Test extend with start_step
    exp["val_loss"].extend([0.5, 0.4, 0.3], start_step=0)
    assert len(exp["val_loss"]) == 3

    # Verify metrics property
    assert set(exp.metrics.keys()) == {"loss", "accuracy", "val_loss"}

    litlogger.finalize()

    # Verify metrics were uploaded
    project_id = exp._teamspace.id
    stream_id = exp._metrics_store.id

    client = LitRestClient()
    response = _wait_for_metrics(
        client,
        project_id=project_id,
        stream_id=stream_id,
        expected_counts={"loss": 10, "accuracy": 10, "val_loss": 3},
    )

    # Cleanup
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )

    # Verify all metrics were logged
    assert "loss" in response.named_metrics
    assert "accuracy" in response.named_metrics
    assert "val_loss" in response.named_metrics
    assert len(response.named_metrics["loss"].ids_metrics[stream_id].metrics_values) == 10


@pytest.mark.cloud()
def test_direct_experiment_usage():
    """Test using the Experiment class directly without module-level API."""
    experiment_name = f"standalone_direct_exp-{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            name=experiment_name,
            teamspace="oss-litlogger",
            log_dir=tmpdir,
            metadata={"direct": "true"},
        )

        # Log metrics directly
        for i in range(5):
            exp.log_metrics({"direct_metric": i * 0.5}, step=i)

        # Create and log a file
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        exp.log_file(test_file)

        exp.finalize()

        # Verify metrics
        project_id = exp._teamspace.id
        stream_id = exp._metrics_store.id

        client = LitRestClient()
        for _ in range(30):
            response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
            if response.named_metrics != {}:
                break
            sleep(1)

        # Cleanup
        client.lit_logger_service_delete_metrics_stream(
            project_id=project_id,
            body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
        )

        # Verify
        assert response.named_metrics != {}
        metrics = response.named_metrics["direct_metric"].ids_metrics[stream_id].metrics_values
        assert len(metrics) == 5
