"""Shared integration cases for Lightning logger adapters."""

import os
import subprocess
import sys
import uuid
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from time import sleep
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.data as data
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning_sdk.lightning_cloud.openapi.models import LitLoggerServiceDeleteMetricsStreamBody
from litlogger.api.client import LitRestClient


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

    for model_name in {experiment.name, getattr(logger, "_checkpoint_name", None)}:
        if not model_name:
            continue
        try:
            model = client.models_store_get_model_by_name(
                project_owner_name=experiment._teamspace.owner.name,
                project_name=experiment._teamspace.name,
                model_name=model_name,
            )
            client.models_store_delete_model(project_id=project_id, model_id=model.id)
        except Exception:
            pass


def _wait_for_model(
    logger: Any,
    *,
    model_name: str,
    attempts: int = 30,
) -> Any:
    experiment = logger.experiment
    for _ in range(attempts):
        try:
            models = experiment.teamspace.list_models()
            model = next((item for item in models if getattr(item, "name", None) == model_name), None)
            if model is not None:
                return model
        except Exception:
            pass
        sleep(1)
    return None


def _metric_values(named_metrics: dict[str, Any], metric_name: str, stream_id: str) -> list[Any]:
    metric = named_metrics.get(metric_name, {})
    ids_metrics = metric.get("ids_metrics", {}) if isinstance(metric, dict) else getattr(metric, "ids_metrics", {})
    stream_metrics = ids_metrics.get(stream_id, {})
    values = (
        stream_metrics.get("metrics_values", {})
        if isinstance(stream_metrics, dict)
        else getattr(stream_metrics, "metrics_values", {})
    )
    return values or []


def _wait_for_metric_count(
    *,
    project_id: str,
    stream_id: str,
    metric_name: str,
    minimum_count: int = 1,
    attempts: int = 30,
) -> Any:
    client = LitRestClient()
    response = None
    for _ in range(attempts):
        response = client.lit_logger_service_get_logger_metrics(project_id=project_id, ids=[stream_id])
        if len(_metric_values(response.named_metrics or {}, metric_name, stream_id)) >= minimum_count:
            return response
        sleep(1)
    return response


def _wait_for_metadata(logger: Any, expected: dict[str, str], attempts: int = 30) -> dict[str, str]:
    metadata = {}
    for _ in range(attempts):
        metadata = logger.experiment.metadata
        if all(metadata.get(key) == value for key, value in expected.items()):
            return metadata
        sleep(1)
    return metadata


def run_full_training_integration(logger_cls: type, *, name_prefix: str, tmpdir: Any) -> None:
    """Run a Trainer fit flow and verify train metrics reach the backend."""
    logger = logger_cls(
        name=_unique_name(name_prefix),
        teamspace="oss-litlogger",
        root_dir=str(tmpdir),
        log_model=False,
    )

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
            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    torch.manual_seed(1234)
    inputs = torch.rand(1_920, 1, 28, 28)
    targets = torch.zeros(1_920, dtype=torch.long)
    train = data.TensorDataset(inputs, targets)
    trainer = Trainer(
        logger=logger,
        max_epochs=1,
        log_every_n_steps=1,
        max_steps=20,
        default_root_dir=tmpdir,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
            category=FutureWarning,
        )
        trainer.fit(LitAutoEncoder(), data.DataLoader(train, batch_size=32))

    project_id, stream_id = _project_and_stream_ids(logger)
    response = _wait_for_metric_count(project_id=project_id, stream_id=stream_id, metric_name="train_loss")

    assert response is not None
    metrics = _metric_values(response.named_metrics or {}, "train_loss", stream_id)
    assert len(metrics) > 0

    _cleanup_logger_run(logger)


def run_checkpoint_upload_smoke(logger_cls: type, *, name_prefix: str, tmpdir: Any) -> None:
    """Verify local checkpointing and basic checkpoint upload."""
    checkpoint_name = f"{_unique_name(name_prefix)}-ckpt"
    logger = logger_cls(
        name=_unique_name(name_prefix),
        teamspace="oss-litlogger",
        root_dir=str(tmpdir),
        log_model=True,
        checkpoint_name=checkpoint_name,
    )

    class LitAutoEncoder(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    torch.manual_seed(1234)
    inputs = torch.rand(640, 1, 28, 28)
    targets = torch.zeros(640, dtype=torch.long)
    train = data.TensorDataset(inputs, targets)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(tmpdir),
        filename="basic-{step}",
        save_top_k=-1,
        every_n_train_steps=5,
        save_last=False,
    )
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=1,
        log_every_n_steps=1,
        max_steps=5,
        default_root_dir=tmpdir,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
            category=FutureWarning,
        )
        trainer.fit(LitAutoEncoder(), data.DataLoader(train, batch_size=32))

    checkpoint_paths = list(Path(str(tmpdir)).rglob("*.ckpt"))
    assert checkpoint_paths

    uploaded_model = _wait_for_model(logger, model_name=checkpoint_name)
    assert uploaded_model is not None

    _cleanup_logger_run(logger)


def run_file_logging(logger_cls: type, *, name_prefix: str, tmpdir: Any) -> None:
    """Verify file logging and file retrieval."""
    logger = logger_cls(
        name=_unique_name(name_prefix),
        teamspace="oss-litlogger",
        root_dir=str(tmpdir),
    )

    config_path = os.path.join(str(tmpdir), "config.yaml")
    with open(config_path, "w") as f:
        f.write("learning_rate: 0.001\nbatch_size: 32\n")

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

    _cleanup_logger_run(logger)


def run_hyperparameter_logging(logger_cls: type, *, name_prefix: str) -> None:
    """Verify hyperparameters are persisted as experiment metadata."""
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
    metadata = _wait_for_metadata(logger, {key: str(value) for key, value in hparams.items()})
    for key, value in hparams.items():
        assert metadata.get(key) == str(value)

    logger.finalize()

    _cleanup_logger_run(logger)


def run_custom_metadata(logger_cls: type, *, name_prefix: str) -> None:
    """Verify metadata provided at init time is stored on the experiment."""
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

    actual_metadata = _wait_for_metadata(logger, metadata)
    for key, value in metadata.items():
        assert actual_metadata.get(key) == value

    logger.finalize()

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
