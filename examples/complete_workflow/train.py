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
import json

import litlogger
from litlogger import File

config = {"model": "ResNet50", "lr": 0.001}
with open("config.json", "w") as f:
    json.dump(config, f)

experiment = litlogger.init(name="image-classifier")

# Set metadata
experiment["model"] = "ResNet50"
experiment["dataset"] = "CIFAR10"

# Log a static file
experiment["config"] = File("config.json")

# Log training metrics
for epoch in range(10):
    experiment["loss"].append(1.0 / (epoch + 1), step=epoch)

experiment.finalize()
