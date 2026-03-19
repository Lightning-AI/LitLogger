"""Launch a small lightning.pytorch training run with save_logs enabled."""

from __future__ import annotations

import argparse
import os

from lightning.pytorch import LightningModule, Trainer
from psutil import cpu_count
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def _build_logger(logger_kind: str, *, name: str, root_dir: str, teamspace: str):
    if logger_kind == "deprecated-wrapper":
        from litlogger import LightningLogger

        logger_cls = LightningLogger
    elif logger_kind == "pytorch-litlogger":
        from lightning.pytorch.loggers import LitLogger

        logger_cls = LitLogger
    else:
        raise ValueError(f"Unsupported logger kind: {logger_kind}")

    return logger_cls(
        name=name,
        save_logs=True,
        root_dir=root_dir,
        teamspace=teamspace,
    )


class LitAutoEncoder(LightningModule):
    def __init__(self, lr: float = 1e-3, inp_size: int = 28) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inp_size * inp_size, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, inp_size * inp_size))
        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logger-kind", required=True, choices=["deprecated-wrapper", "pytorch-litlogger"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--teamspace", required=True)
    args = parser.parse_args()

    logger = _build_logger(
        args.logger_kind,
        name=args.name,
        root_dir=args.root_dir,
        teamspace=args.teamspace,
    )
    train_loader = DataLoader(
        dataset=MNIST(os.getcwd(), download=True, transform=ToTensor()),
        batch_size=32,
        shuffle=True,
        num_workers=cpu_count(),
        persistent_workers=True,
    )

    trainer = Trainer(logger=logger, limit_train_batches=20, max_epochs=1)
    trainer.fit(model=LitAutoEncoder(), train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
