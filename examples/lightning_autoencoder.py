# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import lightning as L
from litlogger import LightningLogger
from psutil import cpu_count
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class LitAutoEncoder(L.LightningModule):
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


if __name__ == "__main__":
    # init the autoencoder
    autoencoder = LitAutoEncoder(lr=1e-3, inp_size=28)

    # setup data
    train_loader = DataLoader(
        dataset=MNIST(os.getcwd(), download=True, transform=ToTensor()),
        batch_size=32,
        shuffle=True,
        num_workers=cpu_count(),
        persistent_workers=True,
    )

    # configure the logger
    lit_logger = LightningLogger()

    # pass logger to the Trainer
    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=3,
        logger=lit_logger,
    )

    # train the model
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)
