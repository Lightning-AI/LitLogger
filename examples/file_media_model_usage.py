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

experiment_name = f"file-media-model-demo-{uuid.uuid4().hex[:8]}"
workdir = Path(tempfile.mkdtemp(prefix="litlogger-file-media-model-"))


experiment = litlogger.init(name=experiment_name)
(workdir / "staging").mkdir(parents=True, exist_ok=True)

experiment["workflow"] = "new-api"
experiment["framework"] = "litlogger"
experiment["notes"] = "files, media, and models"
experiment["summary"] = Text("Static files, media series, metrics, and the new Model API.")

readme_path = workdir / "readme.txt"
readme_path.write_text("dict api example\n", encoding="utf-8")
experiment["artifacts/readme"] = File(str(readme_path))

png_path = workdir / "preview.png"
png_path.write_bytes(
    base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wn4nWQAAAAASUVORK5CYII=")
)
experiment["media/preview"] = Image(str(png_path))

for step in range(3):
    loss = 1.0 / (step + 1)
    experiment["train/loss"].append(loss, step=step)

    report_path = workdir / f"report-{step}.txt"
    report_path.write_text(f"step={step}\nloss={loss:.4f}\n", encoding="utf-8")
    experiment["reports"].append(File(str(report_path)), step=step)

    experiment["captions"].append(Text(f"checkpoint at step {step}"), step=step)

(workdir / "model_artifact").mkdir(parents=True, exist_ok=True)
(workdir / "model_artifact" / "weights.bin").write_bytes(b"fake-weights")
(workdir / "model_artifact" / "config.json").write_text('{"layers": 2}\n', encoding="utf-8")
experiment[f"{experiment_name}/models/latest"] = Model(
    str(workdir / "model_artifact"),
    name=f"{experiment_name}-models-latest",
    version="artifact-v1",
)

experiment[f"{experiment_name}/models/object"] = Model(
    {"kind": "object", "layers": [4, 8, 2], "activation": "relu"},
    name=f"{experiment_name}-models-object",
    staging_dir=str(workdir / "staging" / "object"),
    version="object-v1",
)

checkpoint0 = workdir / "checkpoint-0.ckpt"
checkpoint0.write_text("checkpoint step 0\n", encoding="utf-8")
experiment[f"{experiment_name}/checkpoints"].append(
    Model(str(checkpoint0), name=f"{experiment_name}-checkpoints", version="v1"),
    step=0,
)

checkpoint1 = workdir / "checkpoint-1.ckpt"
checkpoint1.write_text("checkpoint step 1\n", encoding="utf-8")
experiment[f"{experiment_name}/checkpoints"].append(
    Model(str(checkpoint1), name=f"{experiment_name}-checkpoints", version="v2"),
    step=1,
)

print("metadata:", experiment.metadata)
print("artifacts:", sorted(experiment.artifacts))
print("metrics:", sorted(experiment.metrics))

experiment.finalize()

experiment = litlogger.init(name=experiment_name)
(workdir / "downloads").mkdir(parents=True, exist_ok=True)
(workdir / "downloads" / "model-cache").mkdir(parents=True, exist_ok=True)
(workdir / "downloads" / "checkpoints").mkdir(parents=True, exist_ok=True)

print(experiment["artifacts/readme"].save(str(workdir / "downloads" / "readme.txt")))
print(experiment["media/preview"].save(str(workdir / "downloads" / "preview.png")))
print(experiment[f"{experiment_name}/models/latest"].save(str(workdir / "downloads" / "artifact")))
print(experiment[f"{experiment_name}/models/object"].load(str(workdir / "downloads" / "model-cache" / "object")))

checkpoint_series = experiment[f"{experiment_name}/checkpoints"]
print("checkpoint series length:", len(checkpoint_series))
print(checkpoint_series[0].save(str(workdir / "downloads" / "checkpoints" / "checkpoint-0")))
print(checkpoint_series[1].save(str(workdir / "downloads" / "checkpoints" / "checkpoint-1")))
print(checkpoint_series[-1].save(str(workdir / "downloads" / "checkpoints" / "latest")))

experiment.finalize()
