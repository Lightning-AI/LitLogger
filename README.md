<div align="center">

# ⚡ LitLogger - The lightweight AI experiment manager

**Log, track, compare, and share AI model experiments**

<pre>
✅ Lightweight      ✅ Zero-setup           ✅ Any Python code    
✅ Artifacts        ✅ Machine metadata     ✅ Cloud or on-prem   
✅ Training         ✅ Inference            ✅ Agents, multi-modal
✅ Fine-grain RBAC  ✅ Share experiments    ✅ Free tier          
</pre>

______________________________________________________________________

<p align="center">
  <a href="#quick-start">Quick start</a> •
  <a href="#examples">Examples</a> •
  <a href="https://lightning.ai/docs/overview/experiment-management">Docs</a>
</p>

</div>

# Why LitLogger?
Reproducible model building is hard. As teams iterate on models, data, or prompts, it quickly becomes difficult to track what changed and why results improved or regressed. LitLogger is a ***lightweight, minimal*** experiment logger that tracks every run, including inputs, metrics, prompts, and model outputs, so teams can trace changes, compare results, and audit decisions over time without feature bloat or re-running everything from scratch.

LitLogger is free for developers and built into [Lightning AI](https://lightning.ai/), an independent platform trusted by enterprises. It runs in the cloud or on-prem, giving teams long-term stability, clear auditability, and control over experiment history.

<img width="2323" height="1072" alt="image" src="https://github.com/user-attachments/assets/669e8f8e-9d68-473d-8625-6f80d7733cba" />

#  Quick start

Install LitLogger with pip.

```bash
pip install litlogger
```

### Hello world example
Use LitLogger with any Python code (PyTorch, vLLM, LangChain, etc).

```python
from litlogger import LightningLogger

logger = LightningLogger(metadata={"my_metadata": "anything"})

for i in range(10):
    logger.log_metrics({"my_metric": i}, step=i)

# log more than just metrics (files, text, artifacts, model weights)
# logger.log_file("/path/to/config.txt")
# logger.log_model(torch.nnModule)
# logger.log_model_artifact('/path/to/artifact')
logger.finalize()
```

# Examples
Use LitLogger for any usecase (training, inference, agents, etc).

<details>
<summary>Model training</summary>
    
Add LitLogger to any training framework, PyTorch, Jax, TensorFlow, Numpy, SKLearn, etc...

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
</details>

<details>
<summary>Model inference</summary>
    
Add LitLogger to any inference engine, LitServe, vLLM, FastAPI, etc...

<div align='center'>

<img alt="LitServe" src="https://github.com/user-attachments/assets/ac454da2-0825-4fcf-b422-c6d3a1526cf0" width="800px" style="max-width: 100%;">

&nbsp; 
</div>

```python
import time
import litserve as ls
from litlogger import LightningLogger

class InferenceEngine(ls.LitAPI):
    def setup(self, device):
        # initialize your models here
        self.text_model = lambda x: x**2
        self.vision_model = lambda x: x**3
        # initialize LightningLogger
        self.logger = LightningLogger(metadata={"service_name": "InferenceEngine", "device": device})

    def predict(self, request):
        start_time = time.time()
        x = request["input"]    
        
        # perform calculations using both models
        a = self.text_model(x)
        b = self.vision_model(x)
        c = a + b
        output = {"output": c}

        end_time = time.time()
        latency = end_time - start_time

        # log inference metrics
        self.logger.log_metrics({
            "input_value": x,
            "output_value": c,
            "prediction_latency_ms": latency * 1000,
        })
        
        return output

    def teardown(self):
        # ensure the logger is finalized when the service shuts down
        self.logger.finalize()

if __name__ == "__main__":
    server = ls.LitServer(InferenceEngine(max_batch_size=1), accelerator="auto")
    server.run(port=8000)
```
</details>

<details>
<summary>PyTorch Lightning</summary> 
     
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

</details>

<details>
<summary>Example: Long-running experiment simulator</summary>
This is a fun example that simulates a long model training run.

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
LitLogger accepts community contributions - Let's make the world's most advanced AI experiment manager.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litlogger/blob/main/LICENSE)    

