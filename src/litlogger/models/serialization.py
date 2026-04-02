"""Serialization helpers used by the model registry helpers."""

import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lightning_utilities import module_available
from lightning_utilities.core.imports import RequirementCache


@contextmanager
def _suppress_os_stderr() -> Iterator[None]:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)


_JOBLIB_AVAILABLE = module_available("joblib")
_PYTORCH_AVAILABLE = module_available("torch")
with _suppress_os_stderr():
    _TENSORFLOW_AVAILABLE = module_available("tensorflow")
    _KERAS_AVAILABLE = RequirementCache("tensorflow >=2.0.0")

if _JOBLIB_AVAILABLE:
    import joblib  # type: ignore[import-not-found]


def dump_pickle(model: Any, path: str | Path) -> None:
    """Serialize a Python object to disk."""
    if _JOBLIB_AVAILABLE:
        joblib.dump(model, filename=path, compress=7)
        return
    with open(path, "wb") as fp:
        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    """Load a Python object from disk."""
    if _JOBLIB_AVAILABLE:
        return joblib.load(path)
    with open(path, "rb") as fp:
        return pickle.load(fp)
