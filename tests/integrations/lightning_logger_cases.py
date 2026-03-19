"""Shared integration cases for Lightning logger adapters."""

import os
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
    model_names: set[str],
    attempts: int = 30,
) -> Any:
    experiment = logger.experiment

    def _matches(candidate: str | None) -> bool:
        if not candidate:
            return False
        base = candidate.split(":")[0]
        tail = base.rsplit("/", 1)[-1]
        return candidate in model_names or base in model_names or tail in model_names

    for _ in range(attempts):
        try:
            models = experiment.teamspace.list_models()
            model = next((item for item in models if _matches(getattr(item, "name", None))), None)
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


def run_end_to_end_smoke(logger_cls: type, *, name_prefix: str, tmpdir: Any) -> None:
    """Run one end-to-end flow covering the main Lightning adapter behaviors."""
    run_id = _unique_name(name_prefix)
    checkpoint_name = f"{run_id}-ckpt"
    init_metadata = {
        "experiment_type": "classification",
        "framework": "PyTorch Lightning",
    }
    hparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
    }
    logger = logger_cls(
        name=run_id,
        teamspace="oss-litlogger",
        root_dir=str(tmpdir),
        metadata=init_metadata,
        log_model=True,
        save_logs=True,
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
    config_path = os.path.join(str(tmpdir), "config.yaml")
    with open(config_path, "w") as f:
        f.write("learning_rate: 0.001\nbatch_size: 32\n")

    logger.log_hyperparams(hparams)
    logger.log_file(config_path)

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
    logger.finalize("success")

    project_id, stream_id = _project_and_stream_ids(logger)
    response = _wait_for_metric_count(project_id=project_id, stream_id=stream_id, metric_name="train_loss")
    assert response is not None
    assert len(_metric_values(response.named_metrics or {}, "train_loss", stream_id)) > 0

    metadata = _wait_for_metadata(
        logger,
        {
            **init_metadata,
            **{key: str(value) for key, value in hparams.items()},
        },
    )
    for key, value in init_metadata.items():
        assert metadata.get(key) == value
    for key, value in hparams.items():
        assert metadata.get(key) == str(value)

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

    checkpoint_paths = list(Path(str(tmpdir)).rglob("*.ckpt"))
    assert checkpoint_paths

    uploaded_model = _wait_for_model(
        logger,
        model_names={
            checkpoint_name,
            logger.experiment.name,
            logger.name,
        },
    )
    assert uploaded_model is not None

    logs_path = logger.experiment.terminal_logs_path
    assert os.path.basename(logs_path) == "logs.txt"
    assert logs_path == os.path.join(logger.log_dir, "logs.txt")
    if os.environ.get("_IN_PTY_RECORDER") != "1":
        for _ in range(30):
            if os.path.exists(logs_path):
                break
            sleep(1)
        assert os.path.exists(logs_path)

    _cleanup_logger_run(logger)
