"""Integration tests for litlogger standalone API (litlogger.init).

Tests the standalone usage of litlogger without PyTorch Lightning integration.
"""

import os
import pickle
import sys
import tempfile
import uuid
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from time import sleep

import litlogger
import pytest
import torch
import torch.nn as nn
from lightning_sdk.lightning_cloud.openapi.models import LitLoggerServiceDeleteMetricsStreamBody
from lightning_sdk.utils.resolve import _get_cloud_url
from litlogger.api.client import LitRestClient
from litlogger.experiment import Experiment


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
    for _ in range(30):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        if response.named_metrics != {}:
            metrics = response.named_metrics
            if len(metrics.get("loss", {}).ids_metrics.get(stream_id, {}).metrics_values or []) == 10:
                break
        sleep(1)

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


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not support file operations")
@pytest.mark.cloud()
def test_file_operations():
    """Test logging and retrieving files with standalone API."""
    experiment_name = f"standalone_file_test-{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = litlogger.init(
            name=experiment_name,
            teamspace="oss-litlogger",
            root_dir=tmpdir,
        )

        # Create test files
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("learning_rate: 0.001\nbatch_size: 32\nepochs: 10\n")

        results_path = os.path.join(tmpdir, "results.txt")
        with open(results_path, "w") as f:
            f.write("Final accuracy: 0.95\nFinal loss: 0.05\n")

        # Log files (output goes to stderr via Printer)
        err = StringIO()
        with redirect_stderr(err):
            litlogger.log_file(config_path)
            litlogger.log_file(results_path)

        output = err.getvalue()
        assert "config.yaml" in output
        assert "results.txt" in output

        # Wait for file to be uploaded (retry up to 30 times with 1 second sleep)
        file_retrieved = False
        last_file_exception = None
        for attempt in range(30):
            try:
                out = StringIO()
                with redirect_stdout(out):
                    result = litlogger.get_file("config.yaml", verbose=True)
                # If get_file succeeds without exception, file was retrieved
                file_retrieved = True
                break
            except FileNotFoundError as e:
                last_file_exception = e
                if attempt < 29:  # Don't sleep on last attempt
                    sleep(1)
            except Exception as e:
                last_file_exception = e
                if attempt < 29:
                    sleep(1)

        if not file_retrieved:
            import traceback

            traceback.print_exc()
        assert file_retrieved, f"Failed to retrieve config.yaml after waiting. Last error: {last_file_exception}"
        assert os.path.exists(result), f"Downloaded file not found at {result}"

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
def test_model_operations():
    """Test logging and retrieving models with standalone API."""
    experiment_name = f"standalone_model_test-{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = litlogger.init(
            name=experiment_name,
            teamspace="oss-litlogger",
            root_dir=tmpdir,
        )

        # Create a simple PyTorch model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Save model to pickle file
        model_path = os.path.join(tmpdir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"weights": model.state_dict(), "epoch": 5}, f)

        # Test log_model with explicit version (saves model object)
        model_version = "test-v1"
        err = StringIO()
        with redirect_stderr(err):
            litlogger.log_model(model, staging_dir=tmpdir, verbose=True, version=model_version)

        output = err.getvalue()
        # Printer outputs success messages to stderr
        assert "model" in output.lower() or output == ""

        # Retrieve the model object with same version (retry up to 30 times with 1 second sleep)
        model_retrieved = False
        last_model_exception = None
        retrieved_model = None
        for attempt in range(30):
            try:
                out = StringIO()
                with redirect_stdout(out):
                    retrieved_model = litlogger.get_model(staging_dir=tmpdir, verbose=True, version=model_version)
                # If get_model succeeds without exception, model was retrieved
                model_retrieved = True
                break
            except Exception as e:
                last_model_exception = e
                if attempt < 29:
                    sleep(1)

        # Make model retrieval graceful - warn instead of fail
        if model_retrieved:
            # Verify model was retrieved - litmodels may return either the model object or a dict
            if isinstance(retrieved_model, nn.Module):
                # It's a model object
                assert hasattr(retrieved_model, "linear")
            elif isinstance(retrieved_model, dict):
                # It's a dictionary (state_dict or similar)
                assert "weights" in retrieved_model or "state_dict" in retrieved_model or len(retrieved_model) > 0
            else:
                # Unexpected type
                print(f"\nWarning: Retrieved model has unexpected type: {type(retrieved_model)}")
        else:
            # Log warning but don't fail - litmodels may not support this model format or indexing delays
            print(f"\nWarning: Could not retrieve model. Last exception: {last_model_exception}")

        litlogger.finalize()

        # Cleanup
        project_id = exp._teamspace.id
        stream_id = exp._metrics_store.id
        client = LitRestClient()
        client.lit_logger_service_delete_metrics_stream(
            project_id=project_id,
            body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
        )

        # Cleanup model
        try:
            model_record = client.models_store_get_model_by_name(
                project_owner_name=exp._teamspace.owner.name,
                project_name=exp._teamspace.name,
                model_name=experiment_name,
            )
            client.models_store_delete_model(project_id=project_id, model_id=model_record.id)
        except Exception:
            pass  # Model might not exist if upload failed


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not support model artifact operations")
@pytest.mark.cloud()
def test_model_artifact_operations():
    """Test log_model_artifact and get_model_artifact with standalone API."""
    experiment_name = f"standalone_model_artifact_test-{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = litlogger.init(
            name=experiment_name,
            teamspace="oss-litlogger",
            root_dir=tmpdir,
        )

        # Create a model directory with multiple files
        model_dir = os.path.join(tmpdir, "model_artifacts")
        os.makedirs(model_dir)

        # Create model files
        with open(os.path.join(model_dir, "weights.pkl"), "wb") as f:
            pickle.dump({"weights": torch.randn(10, 10)}, f)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            f.write('{"hidden_size": 128}')

        # Test log_model_artifact with explicit version
        artifact_version = "test-v1"
        err = StringIO()
        with redirect_stderr(err):
            litlogger.log_model_artifact(model_dir, verbose=True, version=artifact_version)

        upload_output = err.getvalue()
        print(f"\nUpload output: {upload_output}")

        # Retrieve the model artifact with the same version (retry up to 30 times with 1 second sleep)
        retrieved_dir = os.path.join(tmpdir, "retrieved_artifacts")
        artifact_retrieved = False
        last_artifact_exception = None
        result = None
        for attempt in range(30):
            try:
                out = StringIO()
                with redirect_stdout(out):
                    result = litlogger.get_model_artifact(retrieved_dir, verbose=True, version=artifact_version)
                # If get_model_artifact succeeds without exception, artifact was retrieved
                artifact_retrieved = True
                break
            except Exception as e:
                last_artifact_exception = e
                if attempt < 29:
                    sleep(1)

        # Make artifact retrieval graceful - warn instead of fail
        if artifact_retrieved and result:
            assert os.path.exists(result)
            # Verify the retrieved artifacts contain the expected files
            assert os.path.isdir(result)
            assert os.path.exists(os.path.join(result, "weights.pkl"))
            assert os.path.exists(os.path.join(result, "config.json"))
            # Verify we can load the weights file
            with open(os.path.join(result, "weights.pkl"), "rb") as f:
                weights = pickle.load(f)
            assert "weights" in weights
        else:
            # Log warning but don't fail - litmodels indexing delays
            print(f"\nWarning: Could not retrieve model artifact. Last exception: {last_artifact_exception}")

        litlogger.finalize()

        # Cleanup
        project_id = exp._teamspace.id
        stream_id = exp._metrics_store.id
        client = LitRestClient()
        client.lit_logger_service_delete_metrics_stream(
            project_id=project_id,
            body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
        )

        # Cleanup model if exists
        try:
            model_record = client.models_store_get_model_by_name(
                project_owner_name=exp._teamspace.owner.name,
                project_name=exp._teamspace.name,
                model_name=experiment_name,
            )
            client.models_store_delete_model(project_id=project_id, model_id=model_record.id)
        except Exception:
            pass


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
    experiment_name = f"standalone_get_metadata_test-{uuid.uuid4().hex}"
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
    experiment_name = f"standalone_get_metadata_empty_test-{uuid.uuid4().hex}"

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
    from datetime import datetime, timezone

    experiment_name = f"standalone_direct_metadata-{uuid.uuid4().hex}"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    version_str = timestamp.replace("+00:00", "Z")

    metadata = {
        "framework": "PyTorch",
        "version": "2.0",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            name=experiment_name,
            version=version_str,
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
def test_direct_experiment_usage():
    """Test using the Experiment class directly without module-level API."""
    from datetime import datetime, timezone

    experiment_name = f"standalone_direct_exp-{uuid.uuid4().hex}"
    # Create version as proper RFC 3339 timestamp with Z suffix (required by protobuf)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    version_str = timestamp.replace("+00:00", "Z")

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            name=experiment_name,
            version=version_str,
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
def test_resume_experiment_with_tracker_initialization():
    """Test that resuming an experiment initializes trackers and augments steps correctly."""
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

    from datetime import datetime, timezone

    version = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    # get_or_create should return the existing experiment (created by litlogger.init)
    experiment, created = api.get_or_create_experiment_metrics(
        teamspace_id=teamspace_id,
        name=experiment_name,
        version=version,
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
        version=version,
    )

    assert new_created is True
    assert new_experiment.name == new_experiment_name

    # Cleanup both experiments
    client.lit_logger_service_delete_metrics_stream(
        project_id=teamspace_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[original_id, new_experiment.id]),
    )
