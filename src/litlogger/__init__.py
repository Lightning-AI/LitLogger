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
"""Use litlogger to track machine learning experiments with Lightning.ai.

For guides and examples, see https://lightning.ai.

For reference documentation, see https://github.com/Lightning-AI/litlogger.
"""

__version__ = "0.1.6"

# Import core classes
# Import preinit utilities
from litlogger._preinit import pre_init_callable
from litlogger.experiment import Experiment

# Import SDK functions
from litlogger.init import finish, init

# Global variables
experiment: Experiment | None = None
log = pre_init_callable("litlogger.log", Experiment.log_metrics)
log_metrics = pre_init_callable("litlogger.log_metrics", Experiment.log_metrics)
log_file = pre_init_callable("litlogger.log_file", Experiment.log_file)
get_file = pre_init_callable("litlogger.get_file", Experiment.get_file)
log_model = pre_init_callable("litlogger.log_model", Experiment.log_model)
get_model = pre_init_callable("litlogger.get_model", Experiment.get_model)
log_model_artifact = pre_init_callable("litlogger.log_model_artifact", Experiment.log_model_artifact)
get_model_artifact = pre_init_callable("litlogger.get_model_artifact", Experiment.get_model_artifact)
finalize = pre_init_callable("litlogger.finalize", Experiment.finalize)

__all__ = [
    "Experiment",
    "init",
    "finish",
    "experiment",
    "log",
    "log_metrics",
    "log_file",
    "get_file",
    "log_model",
    "get_model",
    "log_model_artifact",
    "get_model_artifact",
    "finalize",
]

try:
    from litlogger.logger import LightningLogger

    __all__ += ["LightningLogger"]
except ImportError:
    pass
