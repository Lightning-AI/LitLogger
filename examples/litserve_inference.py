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
import litserve as ls
from litlogger import LightningLogger


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.llm = litgpt.LLM.load("EleutherAI/pythia-14m")
        self.logger = LightningLogger(
            name="litserve-inference",
            metadata={"model": "pythia-14m"},
        )

    def decode_request(self, request):
        return request["prompt"]

    def predict(self, prompt):
        start = time.perf_counter()
        output = self.llm.generate(prompt, max_new_tokens=200)
        self.logger.log_metrics(
            {
                "generation_time": time.perf_counter() - start,
                "input_tokens": self.llm.preprocessor.encode(prompt).numel(),
                "output_tokens": self.llm.preprocessor.encode(output).numel(),
            }
        )
        return {"output": output}

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    server = ls.LitServer(SimpleLitAPI())
    server.run(port=8000)
