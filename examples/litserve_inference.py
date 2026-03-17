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
import time

import litgpt
import litlogger
import litserve as ls


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.llm = litgpt.LLM.load("EleutherAI/pythia-14m")
        self.experiment = litlogger.init(
            name="litserve-inference",
            metadata={"model": "pythia-14m"},
        )

    def decode_request(self, request):
        return request["prompt"]

    def predict(self, prompt):
        start = time.perf_counter()
        output = self.llm.generate(prompt, max_new_tokens=200)
        self.experiment["generation_time"].append(time.perf_counter() - start)
        self.experiment["input_tokens"].append(self.llm.preprocessor.encode(prompt).numel())
        self.experiment["output_tokens"].append(self.llm.preprocessor.encode(output).numel())
        return {"output": output}

    def encode_response(self, output):
        return {"output": output}

    def teardown(self, devices):
        self.experiment.finalize()


if __name__ == "__main__":
    server = ls.LitServer(SimpleLitAPI())
    server.run(port=8000)
