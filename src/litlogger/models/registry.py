"""High-level model registry helpers."""

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from litlogger.models.cloud import download_model_files, upload_model_files
from litlogger.models.serialization import _KERAS_AVAILABLE, _PYTORCH_AVAILABLE, dump_pickle, load_pickle

if _PYTORCH_AVAILABLE:
    import torch

if _KERAS_AVAILABLE:
    from tensorflow import keras  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from lightning_sdk.models import UploadedModelInfo


def upload_model(
    name: str,
    model: str | Path,
    progress_bar: bool = True,
    cloud_account: str | None = None,
    verbose: bool | int = 1,
    metadata: dict[str, str] | None = None,
    experiment: Any = None,
) -> "UploadedModelInfo":
    """Upload a local artifact (file or directory) to Lightning Cloud Models."""
    if not isinstance(model, str | Path):
        raise ValueError(
            "The `model` argument should be a path to a file or folder, not an python object."
            " For smooth integrations with PyTorch model, Lightning model and many more, use `save_model` instead."
        )

    return upload_model_files(
        path=model,
        name=name,
        progress_bar=progress_bar,
        cloud_account=cloud_account,
        verbose=verbose,
        metadata=metadata,
        experiment=experiment,
    )


def save_model(
    name: str,
    model: Any,
    progress_bar: bool = True,
    cloud_account: str | None = None,
    staging_dir: str | None = None,
    verbose: bool | int = 1,
    metadata: dict[str, str] | None = None,
    experiment: Any = None,
) -> "UploadedModelInfo":
    """Serialize an in-memory model and upload it to Lightning Cloud Models."""
    if isinstance(model, str | Path):
        raise ValueError(
            "The `model` argument should be a PyTorch model or a Lightning model, not a path to a file."
            " With file or folder path use `upload_model` instead."
        )

    staging_dir = staging_dir or tempfile.mkdtemp()
    if _PYTORCH_AVAILABLE and isinstance(model, torch.jit.ScriptModule):
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.ts")
        model.save(path)
    elif _PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.pth")
        torch.save(model.state_dict(), path)
    elif _KERAS_AVAILABLE and isinstance(model, keras.models.Model):
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.keras")
        model.save(path)
    else:
        path = os.path.join(staging_dir, f"{model.__class__.__name__}.pkl")
        dump_pickle(model=model, path=path)

    upload_metadata = dict(metadata or {})
    upload_metadata["litModels.integration"] = "save_model"

    return upload_model(
        model=path,
        name=name,
        progress_bar=progress_bar,
        cloud_account=cloud_account,
        verbose=verbose,
        metadata=upload_metadata,
        experiment=experiment,
    )


def download_model(
    name: str,
    download_dir: str | Path = ".",
    progress_bar: bool = True,
) -> str | list[str]:
    """Download a model version from Lightning Cloud Models."""
    return download_model_files(
        name=name,
        download_dir=download_dir,
        progress_bar=progress_bar,
    )


def load_model(name: str, download_dir: str | Path = ".") -> Any:
    """Download a model and load it into memory based on its file extension."""
    download_paths = download_model(name=name, download_dir=download_dir)
    if isinstance(download_paths, str):
        download_paths = [download_paths]
    download_paths = [path for path in download_paths if Path(path).suffix.lower() not in {".md", ".txt", ".rst"}]
    if len(download_paths) > 1:
        raise NotImplementedError("Downloaded model with multiple files is not supported yet.")

    model_path = Path(download_paths[0])
    if not model_path.is_absolute():
        model_path = Path(download_dir) / model_path

    if model_path.suffix.lower() == ".ts":
        return torch.jit.load(model_path)  # type: ignore[no-untyped-call]
    if model_path.suffix.lower() == ".keras":
        return keras.models.load_model(model_path)
    if model_path.suffix.lower() == ".pkl":
        return load_pickle(path=model_path)
    raise NotImplementedError(f"Loading model from {model_path.suffix} is not supported yet.")
