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
"""Fabric/PyTorch Lightning logger that sends metrics and artifacts to Lightning.ai."""

import logging
import os
import warnings
from argparse import Namespace
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from lightning_utilities import module_available
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from litlogger.experiment import Experiment
from litlogger.generator import _create_name
from litlogger.types import MediaType

_base_classes: list[type] = []
_F = TypeVar("_F", bound=Callable[..., Any])

log = logging.getLogger(__name__)

# Backwards compatibility -- use LitLogger lightning class if available
# TODO: remove once pytorch lightning is released and we updated the code in frontend
try:
    from lightning.pytorch.loggers import LitLogger as _LightningLitLogger

    _base_classes.append(_LightningLitLogger)
except ImportError:
    pass

try:
    from pytorch_lightning.loggers import LitLogger as _PytorchLightningLitLogger

    _base_classes.append(_PytorchLightningLitLogger)
except ImportError:
    pass

if _base_classes:

    class LightningLogger(*_base_classes):  # type: ignore[misc]
        pass

else:
    if module_available("lightning"):
        from lightning.fabric.loggers.logger import Logger as _LightningLogger
        from lightning.fabric.loggers.logger import rank_zero_experiment
        from lightning.fabric.utilities.cloud_io import get_filesystem
        from lightning.fabric.utilities.logger import _add_prefix
        from lightning.fabric.utilities.rank_zero import rank_zero_only
        from lightning.pytorch.loggers.utilities import _scan_checkpoints

        _base_classes.append(_LightningLogger)

    if module_available("pytorch_lightning"):
        from lightning_fabric.loggers.logger import Logger as _PytorchLightningLogger
        from lightning_fabric.loggers.logger import rank_zero_experiment
        from lightning_fabric.utilities.cloud_io import get_filesystem
        from lightning_fabric.utilities.logger import _add_prefix
        from lightning_fabric.utilities.rank_zero import rank_zero_only
        from pytorch_lightning.loggers.utilities import _scan_checkpoints

        _base_classes.append(_PytorchLightningLogger)

    if not _base_classes:
        raise ModuleNotFoundError("Either `lightning` or `pytorch_lightning` must be installed")

    def _typed_rank_zero_only(fn: _F) -> _F:
        return cast(_F, rank_zero_only(fn))

    def _typed_rank_zero_experiment(fn: _F) -> _F:
        return cast(_F, rank_zero_experiment(fn))

    class LightningLogger(*_base_classes):  # type: ignore[misc, no-redef]
        """Logger that streams metrics and artifacts to the Lightning.ai platform."""

        LOGGER_JOIN_CHAR = "-"

        def __init__(
            self,
            name: str | None = None,
            root_dir: str | Path | None = None,
            teamspace: str | None = None,
            metadata: dict[str, str] | None = None,
            log_model: bool = False,
            save_logs: bool = False,
            checkpoint_name: str | None = None,
        ) -> None:
            """Initialize the LightningLogger.

            Args:
                root_dir: Folder where logs and metadata are stored (default: ./lightning_logs).
                name: Name of your experiment (defaults to a generated name).
                teamspace: Teamspace name where charts and artifacts will appear.
                metadata: Extra metadata to associate with the experiment as tags.
                store_step: Whether to store the step field with each logged value.
                store_created_at: Whether to store a creation timestamp with each value.
                log_model: If True, automatically log model checkpoints as artifacts.
                save_logs: If True, capture and upload terminal logs.
                checkpoint_name: Override the base name for logged checkpoints.

            Example::

                from lightning.pytorch import Trainer
                from lightning.pytorch.demos.boring_classes import BoringModel, BoringDataModule
                from litlogger import LightningLogger

                class LoggingModel(BoringModel):
                    def training_step(self, batch, batch_idx: int):
                        loss = self.step(batch)
                        # logging the computed loss
                        self.log("train_loss", loss)
                        return {"loss": loss}

                trainer = Trainer(
                    max_epochs=10,
                    enable_model_summary=False,
                    logger=LightningLogger("./lightning_logs", name="boring_model")
                )
                model = BoringModel()
                data_module = BoringDataModule()
                trainer.fit(model, data_module)
                trainer.test(model, data_module)

            """
            self._root_dir = os.fspath(root_dir or "./lightning_logs")
            self._name = name or _create_name()
            self._teamspace = teamspace
            self._experiment: Experiment | None = None
            self._sub_dir = None
            self._prefix = ""
            self._fs = get_filesystem(root_dir or ".")
            self._step = -1
            self._metadata = metadata or {}
            self._is_ready = False
            self._log_model = log_model
            self._save_logs = save_logs
            self._checkpoint_callback: Any | None = None
            self._logged_model_time: dict[str, float] = {}
            self._checkpoint_name = checkpoint_name

        @property
        @override
        def name(self) -> str:
            """Gets the name of the experiment."""
            return self._name

        @property
        @override
        def version(self) -> str | None:
            return None

        @property
        @override
        def root_dir(self) -> str:
            """Gets the save directory where the TensorBoard experiments are saved."""
            return self._root_dir

        @property
        @override
        def log_dir(self) -> str:
            """The directory for this run's tensorboard checkpoint."""
            log_dir = os.path.join(self.root_dir, self.name)
            if isinstance(self.sub_dir, str):
                log_dir = os.path.join(log_dir, self.sub_dir)
            log_dir = os.path.expandvars(log_dir)
            return os.path.expanduser(log_dir)

        @property
        def save_dir(self) -> str:
            return self.log_dir

        @property
        def sub_dir(self) -> str | None:
            """Gets the sub directory where the TensorBoard experiments are saved."""
            return self._sub_dir

        @property
        @_typed_rank_zero_experiment
        def experiment(self) -> Experiment | None:
            if self._experiment is not None:
                return self._experiment

            if not self._is_ready:
                # Set ready and continue to create experiment
                self._is_ready = True

            if self.root_dir:
                self._fs.makedirs(self.root_dir, exist_ok=True)

            self._experiment = Experiment(
                name=self._name,
                teamspace=self._teamspace,
                metadata={k: str(v) for k, v in self._metadata.items()},
                store_step=True,
                store_created_at=True,
                log_dir=self.log_dir,
                save_logs=self._save_logs,
            )
            self._experiment.print_url()
            return self._experiment

        def _require_experiment(self) -> Experiment:
            experiment = self.experiment
            if experiment is None:
                raise RuntimeError("Experiment is not initialized")
            return experiment

        @property
        @_typed_rank_zero_only
        def url(self) -> str:
            return self._require_experiment().url

        @override
        @_typed_rank_zero_only
        def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
            self._is_ready = True

            # FIXME: This should be handled by the tracker if this isn't defined by the user.
            self._step = self._step + 1 if step is None else step
            self._store_step = True

            metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
            metrics = {k: v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()}
            self._require_experiment().log_metrics(metrics, step=self._step)

        @override
        @_typed_rank_zero_only
        def log_hyperparams(
            self,
            params: dict[str, Any] | Namespace,
            metrics: dict[str, Any] | None = None,
        ) -> None:
            """Log hyperparams."""
            if isinstance(params, Namespace):
                params = params.__dict__
            params.update(self._metadata or {})
            self._metadata = params

        @_typed_rank_zero_only
        def log_metadata(
            self,
            params: dict[str, Any] | Namespace,
        ) -> None:
            """Log hyperparams."""
            if isinstance(params, Namespace):
                params = params.__dict__
            params.update(self._metadata or {})
            self._metadata = params

        @override
        @_typed_rank_zero_only
        def log_graph(self, model: Module, input_array: Tensor | None = None) -> None:
            warnings.warn("LightningLogger does not support `log_graph`", UserWarning, stacklevel=2)

        @_typed_rank_zero_only
        def log_model(
            self,
            model: Any,
            staging_dir: str | None = None,
            verbose: bool = False,
            version: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            """Save and upload a model object to cloud storage.

            Args:
                model: The model object to save and upload (e.g., torch.nn.Module).
                staging_dir: Optional local directory for staging the model before upload.
                verbose: Whether to show progress bar during upload.
                version: Optional version string for the model.
                metadata: Optional metadata dictionary to store with the model.
            """
            self._is_ready = True
            self._store_step = True
            self._require_experiment().log_model(model, staging_dir, verbose, version, metadata)

        @_typed_rank_zero_only
        def log_model_artifact(
            self,
            path: str,
            verbose: bool = False,
            version: str | None = None,
        ) -> None:
            """Upload a model file or directory to cloud storage using litmodels.

            Args:
                path: Path to the local model file or directory to upload.
                verbose: Whether to show progress bar during upload. Defaults to False.
                version: Optional version string for the model.
            """
            self._is_ready = True
            self._store_step = True
            self._require_experiment().log_model_artifact(path, verbose, version)

        @_typed_rank_zero_only
        def get_file(self, path: str, verbose: bool = True) -> str:
            """Download a file artifact from the cloud for this experiment.

            Args:
                path: Path where the file should be saved locally.
                verbose: Whether to print a confirmation message after download. Defaults to True.

            Returns:
                str: The local path where the file was saved.
            """
            self._is_ready = True
            self._store_step = True
            return self._require_experiment().get_file(path, verbose=verbose)

        @_typed_rank_zero_only
        def get_model(self, staging_dir: str | None = None, verbose: bool = False, version: str | None = None) -> Any:
            """Download and load a model object using litmodels.

            Args:
                staging_dir: Optional directory where the model will be downloaded.
                verbose: Whether to show progress bar.
                version: Optional version string for the model.

            Returns:
                The loaded model object.
            """
            self._is_ready = True
            self._store_step = True
            return self._require_experiment().get_model(staging_dir, verbose, version)

        @_typed_rank_zero_only
        def get_model_artifact(self, path: str, verbose: bool = False, version: str | None = None) -> str:
            """Download a model artifact file or directory from cloud storage using litmodels.

            Args:
                path: Path where the model should be saved locally.
                verbose: Whether to show progress bar during download.
                version: Optional version string for the model.

            Returns:
                str: The local path where the model was saved.
            """
            self._is_ready = True
            self._store_step = True
            return self._require_experiment().get_model_artifact(path, verbose, version)

        @override
        @_typed_rank_zero_only
        def save(self) -> None:
            pass

        @override
        @_typed_rank_zero_only
        def finalize(self, status: str | None = None) -> None:
            if self._experiment is not None:
                self._experiment.finalize(status)
            # log checkpoints as artifacts
            if self._checkpoint_callback and self._experiment is not None:
                self._scan_and_log_checkpoints(self._checkpoint_callback)

        def on_fit_end(self) -> None:
            """Called after fit completes. Ensures metrics are flushed."""
            self.finalize()

        def after_save_checkpoint(self, checkpoint_callback: Any) -> None:
            # log checkpoints as artifacts
            if self._log_model is False:
                return
            if checkpoint_callback.save_top_k == -1:
                self._scan_and_log_checkpoints(checkpoint_callback)
            else:
                self._checkpoint_callback = checkpoint_callback

        def _scan_and_log_checkpoints(self, checkpoint_callback: Any) -> None:
            """Find new checkpoints from the callback and log them as model artifacts."""
            # get checkpoints to be saved with associated score
            checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

            # log iteratively all new checkpoints
            for timestamp, path_ckpt, _score, tag in checkpoints:
                experiment = self._require_experiment()
                if not self._checkpoint_name:
                    self._checkpoint_name = experiment.name
                # Ensure the version tag is unique by appending a timestamp. TODO: make it work with tag as before https://github.com/Lightning-AI/litLogger/pulls
                unique_tag = f"{tag}-{int(datetime.utcnow().timestamp())}"
                self.log_model_artifact(path_ckpt, verbose=True, version=unique_tag)
                # remember logged models - timestamp needed in case filename didn't change (last ckpt or custom name)
                self._logged_model_time[path_ckpt] = timestamp

        def log_file(self, path: str) -> None:
            """Log a file as an artifact to the Lightning platform.

            The file will be logged in the Teamspace drive,
            under a folder identified by the experiment name.

            Args:
                path: Path to the file to log.

            Example::
                logger = LightningLogger(...)
                logger.log_file('config.yaml')
            """
            self._is_ready = True
            self._store_step = True
            self._require_experiment().log_file(path)

        def log_media(
            self,
            name: str,
            path: str,
            kind: MediaType | None = None,
            step: int | None = None,
            epoch: int | None = None,
            caption: str | None = None,
            verbose: bool = False,
        ) -> None:
            """Log a media file to the experiment.

            Args:
                name: Name of the media.
                path: Local path to the media file.
                kind: Kind of media (MediaType.IMAGE or MediaType.TEXT).
                      If None, attempts to guess from file extension or mime type.
                step: Optional training step.
                epoch: Optional training epoch.
                caption: Optional caption.
                verbose: Whether to print a confirmation message after upload.
            """
            self._is_ready = True
            self._store_step = True
            self._require_experiment().log_media(name, path, kind, step, epoch, caption, verbose)
