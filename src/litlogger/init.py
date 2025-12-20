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
"""Initialize litlogger experiment for standalone usage (without PyTorch Lightning)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

from litlogger._module import set_global
from litlogger.experiment import Experiment
from litlogger.generator import _create_name

if TYPE_CHECKING:
    from lightning_sdk import Teamspace


def init(
    name: str | None = None,
    root_dir: str | None = None,
    teamspace: Union[str, "Teamspace"] | None = None,
    metadata: Dict[str, str] | None = None,
    store_step: bool | None = True,
    store_created_at: bool | None = False,
    save_logs: bool = False,
    print_url: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> Experiment:
    """Initialize a litlogger experiment for standalone usage.

    Args:
        name: Name of your experiment (defaults to a generated name).
        root_dir: Folder where logs and metadata are stored (default: ./lightning_logs).
        teamspace: Teamspace where charts and artifacts will appear.
        metadata: Extra metadata to associate with the experiment as tags.
        store_step: Whether to store the step field with each logged value.
        store_created_at: Whether to store a creation timestamp with each value.
        save_logs: If True, capture and upload terminal logs.
        print_url: Whether to print the experiment URL and initialization info.
        verbose: If True, print styled console output. Defaults to True.
        **kwargs: Additional keyword arguments. Will be forwarded to the Experiment constructor.

    Returns:
        Experiment: The initialized experiment instance.

    Example::

        import litlogger

        litlogger.init(name="my-experiment")

        for i in range(100):
            litlogger.log({"loss": 1.0 / (i + 1), "accuracy": i / 100.0}, step=i)

        litlogger.finalize()
    """
    root_dir = root_dir or "./lightning_logs"
    name = name or _create_name()

    # Auto-infer project name from git repo or current directory
    import os
    import subprocess

    try:
        # try to get the git repo name
        git_root = (
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL).decode().strip()
        )
        experiment_project_name = os.path.basename(git_root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # if no git repo, use current directory name
        experiment_project_name = Path.cwd().name

    # default to 'project' if no meaningful name was found
    experiment_project_name = experiment_project_name or "project"

    # Create version as proper RFC 3339 timestamp with Z suffix (required by protobuf)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    # Convert +00:00 to Z (both are valid RFC 3339, but Z might be preferred)
    version_str = timestamp.replace("+00:00", "Z")

    log_dir = os.path.join(root_dir, name)
    os.makedirs(log_dir, exist_ok=True)

    # Create experiment
    experiment = Experiment(
        name=name,
        version=version_str,
        teamspace=teamspace,
        metadata=metadata or {},
        store_step=store_step,
        store_created_at=store_created_at,
        log_dir=log_dir,
        save_logs=save_logs,
        verbose=verbose,
        **kwargs,
    )

    # Set global state
    set_global(
        experiment=experiment,
        log=experiment.log_metrics,
        log_metrics=experiment.log_metrics,
        log_file=experiment.log_file,
        get_file=experiment.get_file,
        log_model=experiment.log_model,
        get_model=experiment.get_model,
        log_model_artifact=experiment.log_model_artifact,
        get_model_artifact=experiment.get_model_artifact,
        finalize=experiment.finalize,
    )

    if print_url:
        experiment.print_url()

    return experiment


def finish(status: str | None = None) -> None:
    """Finalize the current experiment.

    This is an alias for litlogger.finalize().

    Args:
        status: Optional status string to mark the experiment with.
    """
    import litlogger

    if litlogger.experiment is None:
        raise RuntimeError("You must call litlogger.init() before litlogger.finish()")

    litlogger.experiment.finalize(status)
