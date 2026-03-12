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
"""Example of logging files, media, and models with the dict API."""

import base64
import tempfile
import uuid
from pathlib import Path

import litlogger
from litlogger import File, Image, Model, Text

tmpdir = Path(tempfile.mkdtemp(prefix="litlogger-file-media-model-"))
experiment = litlogger.init(name=f"file-media-model-demo-{uuid.uuid4().hex[:8]}")

experiment["workflow"] = "new-api"
experiment["framework"] = "litlogger"
experiment["notes"] = Text("Static text, files, images, and models can all be logged from the dict API.")

readme_path = tmpdir / "readme.txt"
readme_path.write_text("dict api example\n", encoding="utf-8")
experiment["artifacts/readme"] = File(str(readme_path))

png_path = tmpdir / "preview.png"
png_path.write_bytes(
    base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wn4nWQAAAAASUVORK5CYII=")
)
experiment["media/preview"] = Image(str(png_path))

model_path = tmpdir / "checkpoint.ckpt"
model_path.write_text("fake checkpoint contents\n", encoding="utf-8")
experiment["models/latest"] = Model(str(model_path), version="latest")

for step in range(3):
    loss = 1.0 / (step + 1)
    experiment["train/loss"].append(loss, step=step)

    report_path = tmpdir / f"report-{step}.txt"
    report_path.write_text(f"step={step}\nloss={loss:.4f}\n", encoding="utf-8")
    experiment["reports"].append(File(str(report_path)))

    experiment["captions"].append(Text(f"checkpoint at step {step}"), step=step)

    step_model_path = tmpdir / f"checkpoint-{step}.ckpt"
    step_model_path.write_text(f"checkpoint step {step}\n", encoding="utf-8")
    experiment["checkpoints"].append(Model(str(step_model_path), version=f"step-{step}"))

print("metadata:", experiment.metadata)
print("artifacts:", sorted(experiment.artifacts))
print("metrics:", sorted(experiment.metrics))

experiment.finalize()
