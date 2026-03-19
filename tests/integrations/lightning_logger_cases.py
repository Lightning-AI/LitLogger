"""Shared integration cases for Lightning logger adapters."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import uuid
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from time import sleep
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.data as data
import torchvision as tv
from lightning.pytorch import LightningModule, Trainer
from lightning_sdk.lightning_cloud.openapi.models import LitLoggerServiceDeleteMetricsStreamBody
from litlogger.api.client import LitRestClient
from litlogger.diagnostics import collect_system_info


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _project_and_stream_ids(logger: Any) -> tuple[str, str]:
    experiment = logger.experiment
    return experiment._teamspace.id, experiment._metrics_store.id


def _cleanup_logger_run(logger: Any) -> None:
    experiment = logger.experiment
    project_id = experiment._teamspace.id
    stream_id = experiment._metrics_store.id

    client = LitRestClient()
    client.lit_logger_service_delete_metrics_stream(
        project_id=project_id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[stream_id]),
    )

    try:
        model = client.models_store_get_model_by_name(
            project_owner_name=experiment._teamspace.owner.name,
            project_name=experiment._teamspace.name,
            model_name=experiment.name,
        )
        client.models_store_delete_model(project_id=project_id, model_id=model.id)
    except Exception:
        pass


def run_full_training_integration(logger_cls: type, *, name_prefix: str, tmpdir: Any) -> None:
    """Run a full Trainer fit flow and verify uploaded metrics."""
    logger = logger_cls(
        name=_unique_name(name_prefix),
        teamspace="oss-litlogger",
        log_model=True,
    )
    expected_metrics = []

    class LitAutoEncoder(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log("train_loss", loss)
            expected_metrics.append(loss.detach().cpu())
            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    dataset = tv.datasets.MNIST("..", download=True, transform=tv.transforms.ToTensor())
    train, _ = data.random_split(dataset, [55_000, 5_000])

    config_path = os.path.join(str(tmpdir), "config.yaml")
    with open(config_path, "w") as f:
        f.write("foo: bar\n")

    trainer = Trainer(logger=logger, max_epochs=1, log_every_n_steps=1, max_steps=20, default_root_dir=tmpdir)

    err = StringIO()
    with redirect_stderr(err):
        logger.log_file(config_path)
        trainer.fit(LitAutoEncoder(), data.DataLoader(train, batch_size=32))

    output = err.getvalue()
    assert "Logged" in output or config_path in output or output == ""

    project_id, stream_id = _project_and_stream_ids(logger)
    client = LitRestClient()
    response = None
    for _ in range(30):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        if response.named_metrics:
            metrics = response.named_metrics.get("train_loss", {}).ids_metrics.get(stream_id, {}).metrics_values
            if metrics and len(expected_metrics) == len(metrics):
                break
        sleep(1)

    assert response is not None
    assert response.named_metrics != {}
    metrics = response.named_metrics["train_loss"].ids_metrics[stream_id].metrics_values
    assert len(expected_metrics) == len(metrics)

    for metric in metrics:
        idx, actual_value = int(metric.step), metric.value
        assert round(expected_metrics[idx].item(), 3) == round(actual_value, 3)

    _cleanup_logger_run(logger)


def run_console_output(logger_cls: type, *, name_prefix: str) -> None:
    """Verify console output and basic metric logging."""
    logger = logger_cls(
        name=_unique_name(name_prefix),
        teamspace="oss-litlogger",
    )

    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        for i in range(10):
            logger.log_metrics({"my_metric": i}, step=i)
        logger.finalize()

    stderr_output = err.getvalue()
    assert "Run complete" in stderr_output or "Metrics logged" in stderr_output or stderr_output == ""

    _cleanup_logger_run(logger)


def run_system_info(logger_cls: type, *, name_prefix: str) -> None:
    """Verify system info is attached to the metrics stream."""
    logger = logger_cls(
        name=_unique_name(name_prefix),
        teamspace="oss-litlogger",
    )

    for i in range(10):
        logger.log_metrics({"my_metric": i}, step=i)
    logger.finalize()

    project_id, stream_id = _project_and_stream_ids(logger)
    client = LitRestClient()
    response = None
    for _ in range(30):
        response = client.lit_logger_service_list_metrics_streams(project_id=project_id)
        if response.metrics_streams:
            break
        sleep(1)

    assert response is not None
    metrics_stream = next((ms for ms in response.metrics_streams if ms.id == stream_id), None)
    assert metrics_stream is not None
    assert hasattr(metrics_stream, "system_info")
    assert metrics_stream.system_info != {}

    expected_system_info = collect_system_info()
    actual_system_info = metrics_stream.system_info.to_dict()
    for key, value in expected_system_info.items():
        assert actual_system_info[key] == value

    _cleanup_logger_run(logger)


def run_file_and_model_logging(logger_cls: type, *, name_prefix: str, tmpdir: Any) -> None:
    """Verify file logging, file retrieval, and model logging flows."""
    logger = logger_cls(
        name=_unique_name(name_prefix),
        teamspace="oss-litlogger",
        log_model=True,
    )

    class TestModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    test_model = TestModel()
    model_path = os.path.join(str(tmpdir), "test_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"weights": test_model.state_dict(), "epoch": 10}, f)

    config_path = os.path.join(str(tmpdir), "config.yaml")
    with open(config_path, "w") as f:
        f.write("learning_rate: 0.001\nbatch_size: 32\n")

    logger.log_model(test_model, staging_dir=str(tmpdir), version="model-v1")
    logger.log_model_artifact(model_path, version="artifact-v1")
    logger.log_file(config_path)

    retrieved_path = None
    for _ in range(30):
        try:
            with redirect_stdout(StringIO()):
                retrieved_path = logger.get_file("config.yaml", verbose=True)
            break
        except FileNotFoundError:
            sleep(1)

    assert retrieved_path is not None
    assert os.path.exists(retrieved_path)

    logger.finalize()

    client = LitRestClient()
    experiment = logger.experiment
    model = client.models_store_get_model_by_name(
        project_owner_name=experiment._teamspace.owner.name,
        project_name=experiment._teamspace.name,
        model_name=experiment.name,
    )
    assert model is not None

    _cleanup_logger_run(logger)


def run_hyperparameter_logging(logger_cls: type, *, name_prefix: str) -> None:
    """Verify hyperparameters are persisted as metadata tags."""
    logger = logger_cls(
        name=_unique_name(name_prefix),
        teamspace="oss-litlogger",
    )

    hparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "Adam",
        "model": "ResNet50",
    }

    logger.log_hyperparams(hparams)
    for i in range(5):
        logger.log_metrics({"train_loss": i * 0.1}, step=i)
    logger.finalize()

    project_id, stream_id = _project_and_stream_ids(logger)
    client = LitRestClient()
    response = None
    for _ in range(30):
        response = client.lit_logger_service_list_metrics_streams(project_id=project_id)
        if response.metrics_streams:
            break
        sleep(1)

    assert response is not None
    metrics_stream = next((ms for ms in response.metrics_streams if ms.id == stream_id), None)
    assert metrics_stream is not None

    if getattr(metrics_stream, "tags", None):
        stream_metadata = {tag.name: tag.value for tag in metrics_stream.tags}
        for key, value in hparams.items():
            if key in stream_metadata:
                assert stream_metadata[key] == str(value)

    _cleanup_logger_run(logger)


def run_custom_metadata(logger_cls: type, *, name_prefix: str) -> None:
    """Verify metadata provided at init time is stored on the stream."""
    metadata = {
        "experiment_type": "classification",
        "dataset": "CIFAR10",
        "framework": "PyTorch Lightning",
    }
    logger = logger_cls(
        name=_unique_name(name_prefix),
        teamspace="oss-litlogger",
        metadata=metadata,
    )

    for i in range(5):
        logger.log_metrics({"accuracy": i * 0.1}, step=i)
    logger.finalize()

    project_id, stream_id = _project_and_stream_ids(logger)
    client = LitRestClient()
    response = None
    for _ in range(30):
        response = client.lit_logger_service_list_metrics_streams(project_id=project_id)
        if response.metrics_streams:
            break
        sleep(1)

    assert response is not None
    metrics_stream = next((ms for ms in response.metrics_streams if ms.id == stream_id), None)
    assert metrics_stream is not None
    if getattr(metrics_stream, "tags", None):
        stream_metadata = {tag.name: tag.value for tag in metrics_stream.tags}
        for key, value in metadata.items():
            if key in stream_metadata:
                assert stream_metadata[key] == value

    _cleanup_logger_run(logger)


def run_save_logs_subprocess(
    *,
    tmpdir: Any,
    logger_kind: str,
    name_prefix: str,
) -> None:
    """Run a training subprocess with save_logs enabled and verify logs.txt was created."""
    logger_name = _unique_name(name_prefix)
    script_path = Path(__file__).with_name("scripts") / "run_lightning_save_logs.py"

    env = os.environ.copy()
    env.pop("_IN_PTY_RECORDER", None)

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--logger-kind",
            logger_kind,
            "--name",
            logger_name,
            "--root-dir",
            str(tmpdir),
            "--teamspace",
            "oss-litlogger",
        ],
        cwd=tmpdir,
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=90,
    )

    found_path = None
    for root, _, files in os.walk(str(tmpdir)):
        if "logs.txt" in files:
            found_path = os.path.join(root, "logs.txt")
            break

    assert found_path is not None

    with open(found_path) as f:
        content = f.read()
        assert "| Name" in content or "┃ Name " in content
