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
"""Global state management for litlogger."""

from typing import Any, Callable, Dict

import litlogger
from litlogger._preinit import pre_init_callable


def set_global(
    experiment: Any | None = None,
    log: Callable | None = None,
    log_metrics: Callable | None = None,
    log_file: Callable | None = None,
    get_file: Callable | None = None,
    log_model: Callable | None = None,
    get_model: Callable | None = None,
    log_model_artifact: Callable | None = None,
    get_model_artifact: Callable | None = None,
    finalize: Callable | None = None,
) -> None:
    """Set global litlogger state after initialization."""
    if experiment:
        litlogger.experiment = experiment
    if log:
        litlogger.log = log
    if log_metrics:
        litlogger.log_metrics = log_metrics
    if log_file:
        litlogger.log_file = log_file
    if get_file:
        litlogger.get_file = get_file
    if log_model:
        litlogger.log_model = log_model
    if get_model:
        litlogger.get_model = get_model
    if log_model_artifact:
        litlogger.log_model_artifact = log_model_artifact
    if get_model_artifact:
        litlogger.get_model_artifact = get_model_artifact
    if finalize:
        litlogger.finalize = finalize


def get_global() -> Dict[str, Any]:
    """Get the global litlogger state."""
    return {
        "experiment": litlogger.experiment,
        "log": litlogger.log,
        "log_metrics": litlogger.log_metrics,
        "log_file": litlogger.log_file,
        "get_file": litlogger.get_file,
        "log_model": litlogger.log_model,
        "get_model": litlogger.get_model,
        "log_model_artifact": litlogger.log_model_artifact,
        "get_model_artifact": litlogger.get_model_artifact,
        "finalize": litlogger.finalize,
    }


def unset_globals() -> None:
    """Reset global litlogger state to pre-init state."""
    from litlogger.experiment import Experiment

    litlogger.experiment = None
    litlogger.log = pre_init_callable("litlogger.log", Experiment.log_metrics)
    litlogger.log_metrics = pre_init_callable("litlogger.log_metrics", Experiment.log_metrics)
    litlogger.log_file = pre_init_callable("litlogger.log_file", Experiment.log_file)
    litlogger.get_file = pre_init_callable("litlogger.get_file", Experiment.get_file)
    litlogger.log_model = pre_init_callable("litlogger.log_model", Experiment.log_model)
    litlogger.get_model = pre_init_callable("litlogger.get_model", Experiment.get_model)
    litlogger.log_model_artifact = pre_init_callable("litlogger.log_model_artifact", Experiment.log_model_artifact)
    litlogger.get_model_artifact = pre_init_callable("litlogger.get_model_artifact", Experiment.get_model_artifact)
    litlogger.finalize = pre_init_callable("litlogger.finalize", Experiment.finalize)
