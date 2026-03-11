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
"""Example of using litlogger's new dict-like API.

This example demonstrates the pythonic dict-of-lists API where:
- experiment["key"].append(value) logs time-series data
- experiment["key"] = value sets static metadata or files
"""

import random

import litlogger
from litlogger import File, Text

# Initialize experiment
experiment = litlogger.init(name="new-api-demo")

# --- Static metadata (dict of strings) ---
experiment["model"] = "resnet50"
experiment["dataset"] = "cifar10"
experiment["optimizer"] = "adam"

# --- Static file ---
config_file = "/tmp/config.txt"
with open(config_file, "w") as f:
    f.write("learning_rate: 0.001\nbatch_size: 32\n")
experiment["config"] = File(config_file)

# --- Time series metrics (dict of lists) ---
for epoch in range(10):
    loss = 2.0 / (epoch + 1) + random.uniform(-0.1, 0.1)
    accuracy = min(0.95, epoch / 10.0 + random.uniform(-0.05, 0.05))

    # Append with optional step
    experiment["train/loss"].append(loss, step=epoch)
    experiment["train/accuracy"].append(accuracy, step=epoch)

# Extend with start_step for batch logging
val_losses = [0.5, 0.4, 0.3, 0.25, 0.2]
experiment["val/loss"].extend(val_losses, start_step=0)

# --- File time series ---
for epoch in range(3):
    log_content = f"Epoch {epoch}: loss={2.0 / (epoch + 1):.4f}"
    experiment["training-logs"].append(Text(log_content))

# --- Reading data back ---
print(f"Model: {experiment['model']}")
print(f"Train losses: {list(experiment['train/loss'])}")

# --- Convenience properties ---
print(f"All metrics: {list(experiment.metrics.keys())}")
print(f"All artifacts: {list(experiment.artifacts.keys())}")

# Finalize
experiment.finalize()
