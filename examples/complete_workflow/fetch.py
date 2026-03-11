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
"""Fetch data from a previously logged experiment.

This script resumes the 'image-classifier' experiment and reads back
metadata and artifacts using the dict-like API.
"""

import json

import litlogger

experiment = litlogger.init(name="image-classifier")

# Read metadata via property
metadata = experiment.metadata
print(f"Metadata: {metadata}")

# Read artifacts and download
artifacts = experiment.artifacts
if "config" in artifacts:
    artifacts["config"].save("/tmp/config.json")
    with open("/tmp/config.json") as f:
        print(f"Config: {json.load(f)}")

experiment.finalize()
