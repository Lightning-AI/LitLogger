<div align="center">

# ⚡ LitLogger - A minimal logger for AI experiments

**Log, track, and share inputs, outputs, metrics, prompts, and artifacts from any Python code so you can see what changed and why.**

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
When building AI, you change many things at once: code, data, prompts, models. After a few runs, it becomes unclear what actually caused results to improve or regress. LitLogger records every run as it happens. It logs inputs, metrics, prompts, model outputs, and files, without requiring a framework, config files, or a workflow change. You can compare and share runs later instead of rerunning everything from scratch.

LitLogger runs locally (coming soon), in the cloud, or on-prem. It is free for developers and integrates with [Lightning AI](https://lightning.ai/), but works without logging in.

<img width="3024" height="1716" alt="image" src="https://github.com/user-attachments/assets/27f9d8f1-2a13-4080-a64f-374d957712fa" />

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

<div align='center'>

<img alt="LitServe" src="https://github.com/user-attachments/assets/50d9a2f7-17d0-4448-ad21-6be600ab53fc" width="800px" style="max-width: 100%;">

&nbsp; 
</div>

```python
import torch
import torch.nn as nn
import torch.optim as optim
from litlogger import LightningLogger
import os

# define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def train():
    # initialize LightningLogger
    logger = LightningLogger(metadata={"task": "model_training", "model_name": "SimpleModel"})

    # hyperparameters
    num_epochs = 10
    learning_rate = 0.01

    # model, loss, and optimizer
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # dummy data
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)

    # training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # log training loss
        logger.log_metrics({"train_loss": loss.item()}, step=epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # log the trained model
    logger.log_model(model)
    print("model logged.")

    # create a dummy artifact file and log it
    with open("model_config.txt", "w") as f:
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"num_epochs: {num_epochs}\n")
    logger.log_model_artifact("model_config.txt")
    print("model config artifact logged.")
    
    # Clean up the dummy artifact file after logging
    os.remove("model_config.txt")

    # finalize the logger when training is done
    logger.finalize()
    print("training complete and logger finalized.")

if __name__ == "__main__":
    train()
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

Ping the server from the terminal to have it generate some metrics
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"input": 4.0}'
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"input": 5.5}'
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"input": 2.1}'
```

</details>

<details>
<summary>PyTorch Lightning</summary> 
     
PyTorch Lightning now comes with LitLogger natively built in. It's also built by the PyTorch Lightning team for guaranteed fast performance at multi-node GPU scale.

<div align='center'>

<img alt="LitServe" src="https://github.com/user-attachments/assets/43071433-c319-4fc1-ac5a-03a5c5598a88" width="800px" style="max-width: 100%;">

&nbsp; 
</div>

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
<summary>Long-running experiment simulator</summary>
This is a fun example that simulates a long model training run.

<div align='center'>

<img alt="LitServe" src="https://github.com/user-attachments/assets/fd15aa32-2b56-4324-81b6-c87c86db8a3b" width="800px" style="max-width: 100%;">

&nbsp; 
</div>

```python
import random
from time import sleep

import litlogger

litlogger.init(name="loss-simulator")

# Initial loss value
current_loss = 0.09

# Total number of steps
total_steps = 10000

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

