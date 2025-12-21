<div align="center">

# ⚡ LitLogger - The lightweight AI experiment manager

**Log, track, compare, and share AI model experiments**

<pre>
✅ Lightweight      ✅ Zero-setup           ✅ Any Python code  
✅ Artifacts        ✅ Machine metadata     ✅ Cloud or on-prem 
✅ Fine-grain RBAC  ✅ Training, inference  ✅ Free tier        
</pre>

______________________________________________________________________

<p align="center">
  <a href="#quick-start">Quick start</a> •
  <a href="https://lightning.ai/docs/overview/experiment-management">Docs</a>
</p>

</div>

# Why LitLogger?
Reproducible model building is hard. As teams iterate on models, data, or prompts, it quickly becomes difficult to track what changed and why results improved or regressed. LitLogger is a lightweight, minimal experiment logger that tracks every run, including inputs, metrics, prompts, and model outputs, so teams can trace changes, compare results, and audit decisions over time without feature bloat or re-running everything from scratch.

LitLogger is free for developers and built into [Lightning AI](https://lightning.ai/), an independent platform trusted by enterprises. It runs in the cloud or fully on-prem, giving teams long-term stability, clear auditability, and control over their experiment history.

<img width="2323" height="1072" alt="image" src="https://github.com/user-attachments/assets/669e8f8e-9d68-473d-8625-6f80d7733cba" />

#  Quick start

Install LitLogger with pip.

```bash
pip install litlogger
```

### Hello world example
LitLogger works with any Python code, not just model training. Use it with PyTorch, vLLM, LangChain, custom scripts, batch jobs, or live services to track metrics and results consistently.

LitLogger provides a simple functional API for scripts and services, as well as an object-based logger for framework integrations.

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

### PyTorch Lightning example
PyTorch Lightning now comes with LitLogger natively built in. It's also built by the PyTorch Lightning team for guaranteed fast performance at multi-node GPU scale.

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

### Example 3: long-running experiment simulator
This is a fun example that simulates a long model training run.

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

# Community
LitLogger is a community project accepting contributions - Let's make the world's most advanced AI experiment manager.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litlogger/blob/main/LICENSE)    

