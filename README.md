<div align="center">

# ⚡ Lightning Logger

**Log your metrics from anywhere and view them on Lightning.AI**

✅ fast      ✅ cloud persistent      ✅ accessible from everywhere      ✅ easy to share      ✅ easy to use

______________________________________________________________________

<p align="center">
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="https://lightning.ai/docs">Docs</a> •
  <a href="#quick-start">Quick start</a>
</p>

</div>

# Log & view your metrics on Lightning.AI

The Lightning Logger is a python library to log your metrics to the Lightning.AI platform.

### Install Lightning Logger

Install Lightning Logger with pip.

```bash
pip install litlogger
```

### Quick start

Once installed, you can dive right in with our quick start guide. This example shows how to log metadata and metrics using the standalone API.
It's designed to be simple enough for beginners yet robust enough to handle more advanced use cases.

```python
import litlogger

# Initialize experiment with name and metadata
litlogger.init(
    name="my-experiment",
    metadata={
        "learning_rate": "0.001",
        "batch_size": "32",
        "model": "resnet50",
    },
)

# Simulate a training loop
for step in range(100):
    loss = 1.0 / (step + 1)
    accuracy = min(0.95, step / 100.0)

    # Log metrics
    litlogger.log(
        {"train/loss": loss, "train/accuracy": accuracy},
        step=step,
    )

# Optionally log a file
litlogger.log_file("/path/to/config.txt")

# Finalize the experiment
litlogger.finalize()
```

### Lightning AI Boring Model

Integrate Lightning Logger into your model training process with ease.
The example below extends a basic model to log both training and validation metrics, giving you clear insights into model performance with every step.

```python
from lightning import Trainer
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from litlogger import LightningLogger


class LoggingBoringModel(BoringModel):

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        self.log("train_loss", out["loss"])
        return out

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        self.log("val_loss", out["x"])
        return out


trainer = Trainer(max_epochs=10, logger=LightningLogger(), log_every_n_steps=10)
trainer.fit(LoggingBoringModel(), BoringDataModule())
```

### Loss simulator

This detailed example illustrates how to simulate a changing loss value during training.
It logs metric updates at intervals, so you can see how your loss fluctuates over time.
This practical demonstration shows you the flexibility of Lightning Logger in handling real-world training scenarios.

<details>
<summary>Python code</summary>

```python
import random
from time import sleep

import litlogger

litlogger.init(name="loss-simulator")

# Initial loss value
current_loss = 0.09

# Total number of steps
total_steps = 1000000

for i in range(total_steps):
    if (i + 1) % 5 == 0:
        sleep(0.02)  # Simulate some delay

    if i % 100 == 0:
        # Apply small random fluctuation within 5% of the current loss
        fluctuation = random.uniform(-0.05 * current_loss, 0.05 * current_loss)
        current_loss += fluctuation

        # Determine direction of the major adjustment with weighted probabilities
        direction = random.choices(
            population=['decrease', 'increase'],
            weights=[0.6, 0.4],
            k=1
        )[0]

        # Apply major adjustment within 20% of the current loss
        if direction == 'decrease':
            adjustment = random.uniform(-0.2 * current_loss, 0)
        else:
            adjustment = random.uniform(0, 0.2 * current_loss)

        current_loss += adjustment

    # Ensure loss does not go negative
    current_loss = max(0, current_loss)

    # Log metrics less frequently to save resources
    if i % 1000 == 0:
        litlogger.log({"step": i, "loss": current_loss}, step=i)

litlogger.finalize()
```

</details>

Enjoy exploring Lightning Logger and watch your project's metrics come to life on [lightning.ai](lightning.ai)!
This library is designed to simplify your workflow while providing you with clear, insightful data that drives smarter decisions in your development process.
