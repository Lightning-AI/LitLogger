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

This example demonstrates the standalone where you can simply call:
- litlogger.init() to start tracking
- litlogger.log() to log metrics
- litlogger.finalize() to finish tracking
"""

import random
import time

import litlogger

# Initialize experiment
litlogger.init(
    name="standalone-demo",
    # teamspace="my-team",  # Optional: specify teamspace
    metadata={
        "learning_rate": "0.001",
        "batch_size": "32",
        "model": "resnet50",
    },
)

print("Training started...")

# Simulate a training loop
for epoch in range(10):
    for step in range(20):
        # Simulate training metrics
        loss = 2.0 * (1.0 / (epoch * 20 + step + 1))
        accuracy = min(0.95, (epoch * 20 + step) / 200.0)

        # Log metrics
        litlogger.log(
            {
                "train/loss": loss + random.uniform(-0.1, 0.1),
                "train/accuracy": accuracy + random.uniform(-0.05, 0.05),
            },
            step=epoch * 20 + step,
        )

        time.sleep(0.01)  # Simulate training time

    # Log validation metrics at the end of each epoch
    val_loss = loss * 0.9
    val_accuracy = accuracy + 0.05

    litlogger.log(
        {
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
            "epoch": epoch,
        },
        step=(epoch + 1) * 20,
    )

    print(f"Epoch {epoch + 1}/10 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Log a file (e.g., final model weights)
config_file = "/tmp/config.txt"
with open(config_file, "w") as f:
    f.write("learning_rate: 0.001\nbatch_size: 32\nmodel: resnet50\n")

litlogger.log_file(config_file)

# Finalize the experiment
litlogger.finalize()
