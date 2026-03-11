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
"""Example of using litlogger in a standalone script.

This example demonstrates the dict-like API:
- experiment["key"].append(value) to log time-series metrics
- experiment["key"] = "value" to set metadata
- experiment["key"] = File(path) to log static files
"""

import random
import time

import litlogger
from litlogger import File

# Initialize experiment
experiment = litlogger.init(
    name="standalone-demo",
    # teamspace="my-team",  # Optional: specify teamspace
)

# Set metadata
experiment["learning_rate"] = "0.001"
experiment["batch_size"] = "32"
experiment["model"] = "resnet50"

print("Training started...")

# Simulate a training loop
for epoch in range(10):
    for step in range(20):
        # Simulate training metrics
        loss = 2.0 * (1.0 / (epoch * 20 + step + 1))
        accuracy = min(0.95, (epoch * 20 + step) / 200.0)

        global_step = epoch * 20 + step
        experiment["train/loss"].append(loss + random.uniform(-0.1, 0.1), step=global_step)
        experiment["train/accuracy"].append(accuracy + random.uniform(-0.05, 0.05), step=global_step)

        time.sleep(0.01)  # Simulate training time

    # Log validation metrics at the end of each epoch
    val_loss = loss * 0.9
    val_accuracy = accuracy + 0.05

    val_step = (epoch + 1) * 20
    experiment["val/loss"].append(val_loss, step=val_step)
    experiment["val/accuracy"].append(val_accuracy, step=val_step)
    experiment["epoch"].append(epoch, step=val_step)

    print(f"Epoch {epoch + 1}/10 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Log a file (e.g., final config)
config_file = "/tmp/config.txt"
with open(config_file, "w") as f:
    f.write("learning_rate: 0.001\nbatch_size: 32\nmodel: resnet50\n")

experiment["config"] = File(config_file)

# Read back data
print(f"Model: {experiment['model']}")
print(f"Train losses logged: {len(experiment['train/loss'])}")
print(f"All metrics: {list(experiment.metrics.keys())}")

# Finalize the experiment
experiment.finalize()
