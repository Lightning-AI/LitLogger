# simple test with Lighting training experiment using Boring model and mocking all calls to SDK
from typing import Any
from unittest import mock

import pytest

# TODO: remove this once the PL integration is merged
pytest.skip("PL integration is not merged yet", allow_module_level=True)

from lightning_utilities import module_available
from litlogger import LightningLogger

if module_available("lightning"):
    from lightning import Trainer
    from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
else:
    from pytorch_lightning import Trainer
    from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel


class LoggingModel(BoringModel):
    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        # logging the computed loss
        self.log("train_loss", loss)
        return {"loss": loss}


@mock.patch("litlogger.logger.Experiment")
def test_boring_model_training(mock_experiment_class, monkeypatch, tmpdir):
    """Test a simple training experiment with the LightningLogger and a BoringModel."""
    # Create a mock experiment instance
    mock_experiment = mock.MagicMock()
    mock_experiment.log_metrics = mock.MagicMock()
    mock_experiment.finalize = mock.MagicMock()
    mock_experiment.teamspace = mock.MagicMock()
    mock_experiment.teamspace.owner.name = "test-owner"
    mock_experiment.teamspace.name = "test-teamspace"
    mock_experiment_class.return_value = mock_experiment

    # Instantiate the logger
    lit_logger = LightningLogger(log_model=True)

    # Create the trainer
    trainer = Trainer(max_epochs=3, logger=lit_logger, default_root_dir=tmpdir)

    # Train the model
    trainer.fit(LoggingModel(), BoringDataModule())

    # Assertions to ensure the training ran
    # Experiment should be created
    assert mock_experiment_class.call_count == 1
    # Metrics should be logged
    assert mock_experiment.log_metrics.call_count >= 1
    # it can be called multiple times since it is training on random data
    # Changed from log_model to log_model_artifact since checkpoints are logged as artifacts
    assert mock_experiment.log_model_artifact.call_count >= 1
