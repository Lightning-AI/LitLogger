"""Integration tests for experiment lifecycle: init, finalize, resume, errors.

Covers error handling, sequential experiments, resume, console output, and API internals.
"""

import uuid
from contextlib import redirect_stderr
from io import StringIO
from time import sleep

import litlogger
import pytest
from lightning_sdk.lightning_cloud.openapi.models import LitLoggerServiceDeleteMetricsStreamBody
from lightning_sdk.utils.resolve import _get_cloud_url
from litlogger.api.client import LitRestClient

# Suppress deprecation warnings from legacy API usage in integration tests
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.cloud()
def test_error_handling_before_init():
    """Test that calling functions before init() raises proper errors."""
    # Reset global state
    from litlogger._module import unset_globals

    unset_globals()

    # Verify functions raise errors before init
    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.log_metrics({"loss": 0.5})

    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.log({"accuracy": 0.9})

    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.log_file("test.txt")

    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.get_file("test.txt")

    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.log_model(None)

    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.get_model()

    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.log_model_artifact("model.pt")

    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.get_model_artifact("model.pt")

    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.finalize()

    with pytest.raises(RuntimeError, match="You must call litlogger.init\\(\\)"):
        litlogger.get_metadata()


@pytest.mark.cloud()
def test_multiple_experiments_in_sequence():
    """Test running multiple experiments one after another."""
    experiments = []

    for i in range(3):
        experiment_name = f"standalone_sequence_test_{i}-{uuid.uuid4().hex}"
        exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

        # Log some metrics
        for j in range(5):
            litlogger.log_metrics({"metric": j * 0.1 * (i + 1)}, step=j)

        litlogger.finalize()
        experiments.append(exp)

        # Verify state is finalized
        assert exp._finalized

    # Cleanup all experiments
    client = LitRestClient()
    for exp in experiments:
        project_id = exp._teamspace.id
        stream_id = exp._metrics_store.id

        # Verify metrics exist
        for _ in range(30):
            response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
            if response.named_metrics != {}:
                break
            sleep(1)

        client.lit_logger_service_delete_metrics_stream(
            project_id=project_id,
            body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
        )


@pytest.mark.cloud()
def test_finish_with_status():
    """Test litlogger.finish() with different status values."""
    for status in ["completed", "failed", "interrupted"]:
        experiment_name = f"standalone_status_{status}-{uuid.uuid4().hex}"
        exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

        litlogger.log_metrics({"metric": 1.0}, step=0)
        litlogger.finish(status=status)

        # Cleanup
        project_id = exp._teamspace.id
        stream_id = exp._metrics_store.id

        client = LitRestClient()
        for _ in range(30):
            response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
            if response.named_metrics != {}:
                break
            sleep(1)

        client.lit_logger_service_delete_metrics_stream(
            project_id=project_id,
            body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
        )


@pytest.mark.cloud()
def test_console_output():
    """Test that console output contains the expected URL."""
    experiment_name = f"standalone_console_test-{uuid.uuid4().hex}"

    err = StringIO()

    with redirect_stderr(err):
        exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

        for i in range(5):
            litlogger.log_metrics({"metric": i}, step=i)

        litlogger.finalize()

    output = err.getvalue()
    # Printer outputs to stderr with new format
    assert "Experiment initialized" in output or _get_cloud_url() in output

    # Cleanup
    project_id = exp._teamspace.id
    stream_id = exp._metrics_store.id

    client = LitRestClient()
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )


@pytest.mark.cloud()
def test_custom_colors():
    """Test experiments with custom colors."""
    experiment_name = f"standalone_colors_test-{uuid.uuid4().hex}"

    # Note: Colors are typically specified in hex format
    exp = litlogger.init(
        name=experiment_name,
        teamspace="oss-litlogger",
    )

    # Log metrics
    for i in range(5):
        litlogger.log_metrics({"metric": i * 0.1}, step=i)

    litlogger.finalize()

    # Cleanup
    project_id = exp._teamspace.id
    stream_id = exp._metrics_store.id

    client = LitRestClient()
    for _ in range(30):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        if response.named_metrics != {}:
            break
        sleep(0.1)

    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )


@pytest.mark.cloud()
def test_resume_experiment():
    """Test that resuming an experiment augments steps correctly."""
    experiment_name = f"standalone_resume_tracker-{uuid.uuid4().hex}"

    # First experiment run - log metrics with explicit steps
    exp1 = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

    for i in range(10):
        litlogger.log_metrics({"loss": 1.0 - i * 0.1}, step=i)

    litlogger.finalize()

    # Store info for verification
    project_id = exp1._teamspace.id
    stream_id = exp1._metrics_store.id

    # Wait for metrics to be available
    client = LitRestClient()
    for _ in range(30):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        if response.named_metrics != {}:
            metrics = response.named_metrics
            if len(metrics.get("loss", {}).ids_metrics.get(stream_id, {}).metrics_values or []) == 10:
                break
        sleep(1)

    # Second experiment run (resume) - log metrics WITHOUT explicit steps
    # The steps should be augmented from tracker's num_rows (which should be 10)
    exp2 = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

    # Verify that the experiment resumed (same stream ID means same experiment)
    assert exp2._metrics_store.id == stream_id, "Expected to resume the same experiment"

    # Log 5 more metrics WITHOUT explicit steps - they should get steps 10-14
    for i in range(5):
        litlogger.log_metrics({"loss": 0.05 - i * 0.01})  # No step parameter

    litlogger.finalize()

    # Wait for all metrics to be available
    for _ in range(30):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        if response.named_metrics != {}:
            metrics = response.named_metrics
            loss_values = metrics.get("loss", {}).ids_metrics.get(stream_id, {}).metrics_values or []
            if len(loss_values) == 15:  # 10 from first run + 5 from second
                break
        sleep(1)

    # Verify we have all 15 metrics
    loss_metrics = response.named_metrics["loss"].ids_metrics[stream_id].metrics_values
    assert len(loss_metrics) == 15, f"Expected 15 loss metrics, got {len(loss_metrics)}"

    # Verify the steps are sequential (0-9 from first run, 10-14 from second run)
    # Steps may come back as strings from the API, so convert to int for comparison
    steps = sorted([int(m.step) for m in loss_metrics])
    expected_steps = list(range(15))
    assert steps == expected_steps, f"Expected steps {expected_steps}, got {steps}"

    # Cleanup
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )


@pytest.mark.cloud()
def test_resume_old_experiment():
    """Resume an experiment that has many newer experiments in the same teamspace."""
    from litlogger.api.metrics_api import MetricsApi

    target_name = f"standalone_resume_old-{uuid.uuid4().hex}"
    filler_prefix = f"standalone_resume_old_filler-{uuid.uuid4().hex}"

    exp1 = litlogger.init(name=target_name, teamspace="oss-litlogger")
    litlogger.log_metrics({"loss": 0.5}, step=0)
    litlogger.finalize()

    project_id = exp1._teamspace.id
    target_stream_id = exp1._metrics_store.id

    api = MetricsApi()
    client = LitRestClient()

    filler_ids: list[str] = []
    try:
        for i in range(51):
            stream = api.create_experiment_metrics(
                teamspace_id=project_id,
                name=f"{filler_prefix}-{i}",
            )
            filler_ids.append(stream.id)

        exp2 = litlogger.init(name=target_name, teamspace="oss-litlogger")
        try:
            assert (
                exp2._metrics_store.id == target_stream_id
            ), "Expected to resume the original experiment, got a new stream"
        finally:
            litlogger.finalize()
    finally:
        client.lit_logger_service_delete_metrics_stream(
            project_id=project_id,
            body=LitLoggerServiceDeleteMetricsStreamBody(ids=[target_stream_id, *filler_ids]),
        )


@pytest.mark.cloud()
def test_new_dict_api_resume():
    """Test resuming an experiment with the new dict-like API."""
    experiment_name = f"standalone_dict_resume_test-{uuid.uuid4().hex}"

    # First run - log metrics and metadata
    exp1 = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

    exp1["model"] = "ResNet50"
    for i in range(10):
        exp1["loss"].append(1.0 - i * 0.1, step=i)

    litlogger.finalize()

    # Store info for verification
    project_id = exp1._teamspace.id
    stream_id = exp1._metrics_store.id

    # Wait for metrics to be available
    client = LitRestClient()
    for _ in range(30):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        if response.named_metrics != {}:
            metrics = response.named_metrics
            if len(metrics.get("loss", {}).ids_metrics.get(stream_id, {}).metrics_values or []) == 10:
                break
        sleep(1)

    # Second run (resume) - log more metrics
    exp2 = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

    assert exp2._metrics_store.id == stream_id, "Expected to resume the same experiment"

    # Log 5 more metrics without explicit steps (should auto-increment from 10)
    for i in range(5):
        exp2["loss"].append(0.05 - i * 0.01)

    litlogger.finalize()

    # Wait for all metrics
    for _ in range(30):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        if response.named_metrics != {}:
            metrics = response.named_metrics
            loss_values = metrics.get("loss", {}).ids_metrics.get(stream_id, {}).metrics_values or []
            if len(loss_values) == 15:
                break
        sleep(1)

    loss_metrics = response.named_metrics["loss"].ids_metrics[stream_id].metrics_values
    assert len(loss_metrics) == 15, f"Expected 15 loss metrics, got {len(loss_metrics)}"

    steps = sorted([int(m.step) for m in loss_metrics])
    expected_steps = list(range(15))
    assert steps == expected_steps, f"Expected steps {expected_steps}, got {steps}"

    # Cleanup
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )


@pytest.mark.cloud()
def test_new_dict_api_key_uniqueness():
    """Test that key uniqueness is enforced across the new dict API."""
    experiment_name = f"standalone_dict_uniqueness_test-{uuid.uuid4().hex}"
    exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger")

    # Set a metadata key
    exp["tag"] = "value"

    # Trying to use the same key for metrics should raise
    with pytest.raises(KeyError, match="already used"):
        exp["tag"].append(0.5)

    # Set a metric key
    exp["loss"].append(0.5, step=0)

    # Trying to use the same key for metadata should raise
    with pytest.raises(KeyError, match="already used"):
        exp["loss"] = "not-a-metric"

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
def test_get_or_create_experiment_metrics():
    """Test get_or_create_experiment_metrics returns existing experiment on second call."""
    from litlogger.api.metrics_api import MetricsApi

    experiment_name = f"standalone_get_or_create-{uuid.uuid4().hex}"

    # Use litlogger.init to create the first experiment and get teamspace_id
    exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger")
    litlogger.finalize()

    teamspace_id = exp._teamspace.id
    original_id = exp._metrics_store.id

    # Now use the API to test get_or_create
    api = MetricsApi()
    client = LitRestClient()

    # get_or_create should return the existing experiment (created by litlogger.init)
    experiment, created = api.get_or_create_experiment_metrics(
        teamspace_id=teamspace_id,
        name=experiment_name,
    )

    assert created is False
    assert experiment.id == original_id
    assert experiment.name == experiment_name

    # Also test get_experiment_metrics_by_name directly
    fetched = api.get_experiment_metrics_by_name(
        teamspace_id=teamspace_id,
        name=experiment_name,
    )
    assert fetched is not None
    assert fetched.id == original_id

    # Now test creating a new experiment with a different name
    new_experiment_name = f"standalone_get_or_create_new-{uuid.uuid4().hex}"
    new_experiment, new_created = api.get_or_create_experiment_metrics(
        teamspace_id=teamspace_id,
        name=new_experiment_name,
    )

    assert new_created is True
    assert new_experiment.name == new_experiment_name

    # Cleanup both experiments
    client.lit_logger_service_delete_metrics_stream(
        project_id=teamspace_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[original_id, new_experiment.id]),
    )
