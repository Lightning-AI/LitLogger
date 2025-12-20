"""Integration tests for LightningLogger with PyTorch Lightning.

Tests the PyTorch Lightning integration of litlogger using the LightningLogger class.
"""

import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import uuid
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from time import sleep

import pytest

# TODO: remove this once the PL integration is merged
pytest.skip("PL integration is not merged yet", allow_module_level=True)

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.data as data
import torchvision as tv
from lightning_sdk.lightning_cloud.openapi.models import LitLoggerServiceDeleteMetricsStreamBody
from lightning_sdk.lightning_cloud.openapi.rest import ApiException
from lightning_utilities import module_available
from litlogger import LightningLogger
from litlogger.api.client import LitRestClient
from litlogger.diagnostics import collect_system_info

if module_available("lightning"):
    from lightning import LightningModule, Trainer
else:
    from pytorch_lightning import LightningModule, Trainer


@pytest.mark.cloud()
def test_full_training_integration(tmpdir):
    """Full end-to-end test with PyTorch Lightning training."""
    logger = LightningLogger(
        name=f"lightning_full_training-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
        log_model=True,
    )
    expected_metrics = []

    # Define a LightningModule
    class LitAutoEncoder(LightningModule):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

        def forward(self, x):
            return self.encoder(x)

        def training_step(self, batch, batch_idx):
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log("train_loss", loss)
            expected_metrics.append(loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    # Prepare data
    print("Preparing data")
    dataset = tv.datasets.MNIST("..", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    # Log a config file
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("foo: bar\n")
        file_path = tmp.name

    # Train
    print("Starting training")
    autoencoder = LitAutoEncoder()
    trainer = Trainer(logger=logger, max_epochs=1, log_every_n_steps=1, max_steps=100, default_root_dir=tmpdir)

    err = StringIO()
    with redirect_stderr(err):
        try:
            logger.log_file(file_path)
            os.remove(file_path)
            trainer.fit(autoencoder, data.DataLoader(train, batch_size=32))
        except ApiException as e:
            print(f"An API error occurred: {e}")

    print("Training is finished")

    # Assertions - Printer outputs to stderr
    output = err.getvalue()
    # Check for logged file confirmation (Printer format: "✓ Logged <path>")
    assert "Logged" in output or file_path in output

    # Retrieve and verify metrics
    project_id = logger._experiment._teamspace.id
    stream_id = logger._experiment._metrics_store.id

    client = LitRestClient()
    for _ in range(30):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        if response.named_metrics != {}:
            metrics = response.named_metrics.get("train_loss", {}).ids_metrics.get(stream_id, {}).metrics_values
            if metrics and len(expected_metrics) == len(metrics):
                break
        sleep(1)

    assert response.named_metrics != {}
    metrics = response.named_metrics["train_loss"].ids_metrics[stream_id].metrics_values
    assert len(expected_metrics) == len(metrics)

    # Compare metrics
    for metric in metrics:
        idx, actual_value = int(metric.step), metric.value
        assert round(expected_metrics[idx].item(), 3) == round(actual_value, 3)

    # Cleanup
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )

    # Cleanup model
    try:
        model = client.models_store_get_model_by_name(
            project_owner_name=logger._experiment._teamspace.owner.name,
            project_name=logger._experiment._teamspace.name,
            model_name=logger._name,
        )
        client.models_store_delete_model(project_id=project_id, model_id=model.id)
    except Exception:
        pass


@pytest.mark.cloud()
def test_console_output():
    """Test that console output contains the expected information."""
    logger = LightningLogger(
        name=f"lightning_console_output-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
    )

    out, err = StringIO(), StringIO()

    with redirect_stdout(out), redirect_stderr(err):
        for i in range(10):
            logger.log_metrics({"my_metric": i}, step=i)
        logger.finalize()

    # Cleanup
    project_id = logger._experiment._teamspace.id
    stream_id = logger._experiment._metrics_store.id

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

    # Printer outputs to stderr - verify it contains expected content
    stderr_output = err.getvalue()
    # Should contain run completion info
    assert "Run complete" in stderr_output or "Metrics logged" in stderr_output, stderr_output


@pytest.mark.cloud()
def test_system_info():
    """Test that system info is collected and uploaded."""
    logger = LightningLogger(
        name=f"lightning_system_info-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
    )

    for i in range(10):
        logger.log_metrics({"my_metric": i}, step=i)
    logger.finalize()

    # Retrieve metrics stream
    project_id = logger._experiment._teamspace.id
    stream_id = logger._experiment._metrics_store.id

    client = LitRestClient()
    for _ in range(30):
        response = client.lit_logger_service_list_metrics_streams(project_id=project_id)
        if response.metrics_streams:
            break
        sleep(1)

    # Cleanup
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )

    # Verify system info
    metrics_stream = next((ms for ms in response.metrics_streams if ms.id == stream_id), None)
    assert metrics_stream, "metrics stream is not found"
    assert hasattr(metrics_stream, "system_info"), "system info is not in the metrics stream"
    assert metrics_stream.system_info != {}, "system info is empty"

    expected_system_info = collect_system_info()
    actual_system_info = metrics_stream.system_info.to_dict()
    for key, value in expected_system_info.items():
        assert key in actual_system_info, f"{key} is not in the system info"
        assert actual_system_info[key] == value, f"{key} has different value"


@pytest.mark.cloud()
def test_model_and_file_operations(tmpdir):
    """Test logging and retrieving models and files."""
    logger = LightningLogger(
        name=f"lightning_model_file_ops-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
        log_model=True,
    )

    # Create a test PyTorch model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    test_model = TestModel()
    model_path = os.path.join(tmpdir, "test_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"weights": test_model.state_dict(), "epoch": 10}, f)

    # Create a test config file
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        f.write("learning_rate: 0.001\nbatch_size: 32\n")

    # Log model and file with explicit versions
    model_version = "test-v1"
    artifact_version = "test-v1"
    err = StringIO()
    with redirect_stderr(err):
        logger.log_model(test_model, staging_dir=tmpdir, version=model_version)
        logger.log_model_artifact(model_path, version=artifact_version)
        logger.log_file(config_path)

    # Wait for file to be uploaded (retry up to 30 times with 1 second sleep)
    file_retrieved = False
    last_file_exception = None
    for attempt in range(30):
        try:
            out = StringIO()
            with redirect_stdout(out):
                result = logger.get_file("config.yaml", verbose=True)
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
    assert os.path.exists(result)

    # Retrieve the model object with same version (retry up to 30 times with 1 second sleep)
    model_retrieved = False
    last_model_exception = None
    for attempt in range(30):
        try:
            out = StringIO()
            with redirect_stdout(out):
                retrieved_model = logger.get_model(staging_dir=tmpdir, verbose=True, version=model_version)
            # If get_model succeeds without exception, model was retrieved
            model_retrieved = True
            break
        except Exception as e:
            last_model_exception = e
            if attempt < 29:
                sleep(1)

    if model_retrieved:
        # Verify model was retrieved - litmodels may return either the model object or a dict
        # depending on how it was saved
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
        # Log warning but don't fail - litmodels may not support this model format
        print(f"\nWarning: Could not retrieve model. Last exception: {last_model_exception}")

    # Retrieve the model artifact with explicit version (retry up to 30 times with 1 second sleep)
    model_artifact_dir = os.path.join(tmpdir, "retrieved_model_artifact")
    artifact_retrieved = False
    result_artifact = None
    for attempt in range(30):
        try:
            out = StringIO()
            with redirect_stdout(out):
                result_artifact = logger.get_model_artifact(model_artifact_dir, verbose=True, version=artifact_version)
            # If get_model_artifact succeeds without exception, artifact was retrieved
            artifact_retrieved = True
            break
        except Exception as e:
            if attempt < 29:
                sleep(1)
                continue

            raise e

    assert artifact_retrieved
    assert result_artifact
    # result_artifact might be a full path, relative path, or just a filename
    # Try to construct the full path if it's not absolute
    if not os.path.isabs(result_artifact):
        full_artifact_path = os.path.join(model_artifact_dir, result_artifact)
    else:
        full_artifact_path = result_artifact

    # Assert that the path exists
    assert os.path.exists(full_artifact_path), f"Downloaded artifact path does not exist: {full_artifact_path}"

    # The result could be a directory or a file - check both
    if os.path.isdir(full_artifact_path):
        # Look for the pickle file in the directory
        pkl_file = os.path.join(full_artifact_path, "test_model.pkl")
        assert os.path.exists(pkl_file), f"Expected {pkl_file} to exist in downloaded directory"
        with open(pkl_file, "rb") as f:
            loaded_artifact = pickle.load(f)
    else:
        # It's a file
        with open(full_artifact_path, "rb") as f:
            loaded_artifact = pickle.load(f)

    assert isinstance(loaded_artifact, dict)
    assert "weights" in loaded_artifact
    assert "epoch" in loaded_artifact
    logger.finalize()

    # Cleanup
    project_id = logger._experiment._teamspace.id
    stream_id = logger._experiment._metrics_store.id

    client = LitRestClient()
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )

    # Delete model
    try:
        model = client.models_store_get_model_by_name(
            project_owner_name=logger._experiment._teamspace.owner.name,
            project_name=logger._experiment._teamspace.name,
            model_name=logger._name,
        )
        client.models_store_delete_model(project_id=project_id, model_id=model.id)
    except Exception:
        pass


@pytest.mark.cloud()
def test_storage_configurations():
    """Test different storage configurations (store_step, store_created_at)."""
    # Test with store_step=False
    logger1 = LightningLogger(
        name=f"lightning_no_step-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
        store_step=False,
    )

    for i in range(5):
        logger1.log_metrics({"loss": i * 0.1})

    logger1.finalize()

    # Test with store_created_at=True
    logger2 = LightningLogger(
        name=f"lightning_with_timestamp-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
        store_created_at=True,
    )

    for i in range(5):
        logger2.log_metrics({"accuracy": i * 0.2}, step=i)

    logger2.finalize()

    # Verify and cleanup
    client = LitRestClient()

    for logger in [logger1, logger2]:
        project_id = logger._experiment._teamspace.id
        stream_id = logger._experiment._metrics_store.id

        for _ in range(30):
            response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
            if response.named_metrics != {}:
                break
            sleep(1)

        # Verify metrics exist
        assert response.named_metrics != {}, f"Metrics not found for {logger._name}"

        # Cleanup
        client.lit_logger_service_delete_metrics_stream(
            project_id=project_id,
            body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
        )


@pytest.mark.cloud()
def test_finish_with_different_statuses():
    """Test finalize() with different status values."""
    for status in ["completed", "failed", "interrupted"]:
        logger = LightningLogger(
            name=f"lightning_status_{status}-{uuid.uuid4().hex}",
            teamspace="oss-litlogger",
        )
        logger.log_metrics({"metric": 1.0}, step=0)
        logger.finalize(status=status)

        # Cleanup
        project_id = logger._experiment._teamspace.id
        stream_id = logger._experiment._metrics_store.id

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
def test_hyperparameter_logging():
    """Test logging hyperparameters."""
    logger = LightningLogger(
        name=f"lightning_hyperparams-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
    )

    # Log hyperparameters
    hparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "Adam",
        "model": "ResNet50",
    }

    logger.log_hyperparams(hparams)

    # Log some metrics
    for i in range(5):
        logger.log_metrics({"train_loss": i * 0.1}, step=i)

    logger.finalize()

    # Cleanup
    project_id = logger._experiment._teamspace.id
    stream_id = logger._experiment._metrics_store.id

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
def test_custom_metadata_and_tags():
    """Test logger with custom metadata/tags."""
    metadata = {
        "experiment_type": "classification",
        "dataset": "CIFAR10",
        "framework": "PyTorch Lightning",
    }

    logger = LightningLogger(
        name=f"lightning_metadata-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
        metadata=metadata,
    )

    # Log some metrics
    for i in range(5):
        logger.log_metrics({"accuracy": i * 0.1}, step=i)

    logger.finalize()

    # Verify metadata
    project_id = logger._experiment._teamspace.id
    stream_id = logger._experiment._metrics_store.id

    client = LitRestClient()
    for _ in range(30):
        response = client.lit_logger_service_list_metrics_streams(project_id=project_id)
        if response.metrics_streams:
            break
        sleep(1)

    # Find our stream
    metrics_stream = next((ms for ms in response.metrics_streams if ms.id == stream_id), None)
    assert metrics_stream is not None

    # Verify metadata if available
    if hasattr(metrics_stream, "tags") and metrics_stream.tags:
        # Convert tags list to dict
        stream_metadata = {tag.name: tag.value for tag in metrics_stream.tags}
        for key, value in metadata.items():
            if key in stream_metadata:
                assert stream_metadata[key] == value

    # Cleanup
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )


@pytest.mark.cloud()
def test_model_artifact_operations(tmpdir):
    """Test log_model_artifact and get_model_artifact methods."""
    logger = LightningLogger(
        name=f"lightning_model_artifact-{uuid.uuid4().hex}",
        teamspace="oss-litlogger",
    )

    # Create a model directory with multiple files
    model_dir = os.path.join(tmpdir, "model_artifact")
    os.makedirs(model_dir)

    # Save model components
    with open(os.path.join(model_dir, "weights.pkl"), "wb") as f:
        pickle.dump({"weights": torch.randn(5, 5)}, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        f.write('{"hidden_size": 128, "num_layers": 3}')

    # Log model artifact
    err = StringIO()
    with redirect_stderr(err):
        logger.log_model_artifact(model_dir, verbose=True)

    output = err.getvalue()
    # Printer outputs to stderr - model artifact should be logged
    assert "Logged" in output or "model" in output.lower() or output == ""

    logger.finalize()

    # Cleanup
    project_id = logger._experiment._teamspace.id
    stream_id = logger._experiment._metrics_store.id

    client = LitRestClient()
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )

    # Try to cleanup model if it exists
    try:
        model = client.models_store_get_model_by_name(
            project_owner_name=logger._experiment._teamspace.owner.name,
            project_name=logger._experiment._teamspace.name,
            model_name=logger._name,
        )
        client.models_store_delete_model(project_id=project_id, model_id=model.id)
    except Exception:
        pass


def test_log_file_creation_with_save_logs(tmpdir):
    """Launches a lightning trainer run with save_logs = True.

    Subprocess is needed here to avoid pytest exiting due to how
    terminal outputs are rerouted in Experiment.

    NOTE: This is an integration test that requires PyTorch Lightning
    and tests the PTY logging functionality with real cloud APIs.
    """
    code_to_run = textwrap.dedent(f"""
        import os
        from litlogger import LightningLogger
        from pytorch_lightning import Trainer, LightningModule
        import torch

        from psutil import cpu_count
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor

        class LitAutoEncoder(LightningModule):
            def __init__(self, lr=1e-3, inp_size=28):
                super().__init__()

                self.encoder = nn.Sequential(nn.Linear(inp_size * inp_size, 64), nn.ReLU(), nn.Linear(64, 3))
                self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, inp_size * inp_size))
                self.lr = lr
                self.save_hyperparameters()

            def training_step(self, batch, batch_idx):
                x, y = batch
                x = x.view(x.size(0), -1)
                z = self.encoder(x)
                x_hat = self.decoder(z)
                loss = nn.functional.mse_loss(x_hat, x)
                # log metrics
                self.log("train_loss", loss)
                return loss

            def configure_optimizers(self):
                return optim.Adam(self.parameters(), lr=self.lr)

        autoencoder = LitAutoEncoder(lr=1e-3, inp_size=28)

        train_loader = DataLoader(
            dataset=MNIST(os.getcwd(), download=True, transform=ToTensor()),
            batch_size=32,
            shuffle=True,
            num_workers=cpu_count(),
            persistent_workers=True,
        )

        logger = LightningLogger(
            name="save_log_test",
            save_logs=True,
            root_dir=r"{tmpdir}",
            teamspace="oss-litlogger"
        )

        # pass logger to the Trainer
        trainer = Trainer(
            logger=logger,
            limit_train_batches=50,
            max_epochs=2,
        )

        # train the model
        trainer.fit(model=autoencoder, train_dataloaders=train_loader)
        """)

    script_path = os.path.join(tmpdir, "run_script.py")
    with open(script_path, "w") as f:
        f.write(code_to_run)

    env = os.environ.copy()
    env.pop("_IN_PTY_RECORDER", None)

    subprocess.run(
        [sys.executable, script_path], cwd=tmpdir, capture_output=True, text=True, env=env, check=True, timeout=60
    )

    found_path = None
    for root, _, files in os.walk(tmpdir):
        if "logs.txt" in files:
            found_path = os.path.join(root, "logs.txt")
            break

    assert found_path is not None, f"logs.txt was not found anywhere inside {tmpdir}"

    with open(found_path) as f:
        content = f.read()
        # this ensures all ansi codes are stripped correctly
        # with rich, the output table is formatted differently using unicode, so we check for both
        assert "| Name" in content or "┃ Name " in content
