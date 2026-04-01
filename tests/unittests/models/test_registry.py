import os
from contextlib import nullcontext
from unittest import mock

import litlogger
import pytest
import torch
import torch.jit as torch_jit
from litlogger.models import download_model, load_model, save_model
from litlogger.models.cloud import upload_model_files
from litlogger.models.serialization import _KERAS_AVAILABLE, dump_pickle
from torch.nn import Module

LIT_USER = "i-am-ci"
LIT_TEAMSPACE = "OSS | litModels"


class PickleModel:
    pass


@pytest.mark.parametrize("name", ["/too/many/slashes", "org/model", "model-name"])
@pytest.mark.parametrize("in_studio", [True, False])
@mock.patch("litlogger.models.cloud.sdk_upload_model")
def test_upload_wrong_model_name(mock_sdk_upload, name, in_studio, monkeypatch):
    teamspace = mock.MagicMock()
    teamspace.name = LIT_TEAMSPACE
    teamspace.owner.name = LIT_USER
    monkeypatch.setattr(
        "lightning_sdk.models._resolve_teamspace", mock.MagicMock(return_value=teamspace if in_studio else None)
    )

    if in_studio:
        monkeypatch.setenv("LIGHTNING_USERNAME", LIT_USER)
        monkeypatch.setenv("LIGHTNING_TEAMSPACE", LIT_TEAMSPACE)
        monkeypatch.setattr("lightning_sdk.organization.Organization", mock.MagicMock)
        monkeypatch.setattr("lightning_sdk.teamspace.Teamspace", mock.MagicMock)
        monkeypatch.setattr("lightning_sdk.teamspace.TeamspaceApi", mock.MagicMock)
        monkeypatch.setattr("lightning_sdk.models._get_teamspace", mock.MagicMock)

    allow_short_name = in_studio and name == "model-name"
    with (
        pytest.raises(ValueError, match=r".*organization/teamspace/model.*") if not allow_short_name else nullcontext()
    ):
        upload_model_files(path="path/to/checkpoint", name=name)


@pytest.mark.parametrize(
    ("model", "model_path", "verbose"),
    [
        (torch_jit.script(Module()), f"%s{os.path.sep}RecursiveScriptModule.ts", True),
        (Module(), f"%s{os.path.sep}Module.pth", True),
        (PickleModel(), f"%s{os.path.sep}PickleModel.pkl", 1),
    ],
)
@mock.patch("litlogger.models.cloud.sdk_upload_model")
def test_save_model(mock_upload_model, tmp_path, model, model_path, verbose):
    mock_upload_model.return_value.name = "org-name/teamspace/model-name"

    save_model(
        model=model,
        name="org-name/teamspace/model-name",
        cloud_account="cluster_id",
        staging_dir=str(tmp_path),
        verbose=verbose,
    )
    expected_path = model_path % str(tmp_path) if "%" in model_path else model_path
    mock_upload_model.assert_called_once_with(
        path=expected_path,
        name="org-name/teamspace/model-name",
        cloud_account="cluster_id",
        progress_bar=True,
        metadata={"litModels": litlogger.__version__, "litModels.integration": "save_model"},
        experiment=None,
    )


@mock.patch("litlogger.models.cloud.sdk_download_model")
def test_download_model(mock_download_model):
    download_model(
        name="org-name/teamspace/model-name",
        download_dir="where/to/download",
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir="where/to/download", progress_bar=True
    )


@mock.patch("litlogger.models.cloud.sdk_download_model")
def test_load_model_pickle(mock_download_model, tmp_path):
    model_file = tmp_path / "dummy_model.pkl"
    test_data = PickleModel()
    dump_pickle(test_data, model_file)
    mock_download_model.return_value = [str(model_file.name)]

    model = load_model(
        name="org-name/teamspace/model-name",
        download_dir=str(tmp_path),
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir=str(tmp_path), progress_bar=True
    )
    assert isinstance(model, PickleModel)


@mock.patch("litlogger.models.cloud.sdk_download_model")
def test_load_model_torch_jit(mock_download_model, tmp_path):
    model_file = tmp_path / "dummy_model.ts"
    test_data = torch_jit.script(Module())
    test_data.save(model_file)
    mock_download_model.return_value = [str(model_file.name)]

    model = load_model(
        name="org-name/teamspace/model-name",
        download_dir=str(tmp_path),
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir=str(tmp_path), progress_bar=True
    )
    assert isinstance(model, torch.jit.ScriptModule)


@pytest.mark.skipif(not _KERAS_AVAILABLE, reason="TensorFlow/Keras is not available")
@mock.patch("litlogger.models.cloud.sdk_download_model")
def test_load_model_tf_keras(mock_download_model, tmp_path):
    from tensorflow import keras

    model_file = tmp_path / "dummy_model.keras"
    model = keras.Sequential(
        [
            keras.layers.Dense(10, input_shape=(784,), name="dense_1"),
            keras.layers.Dense(10, name="dense_2"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.save(model_file)
    mock_download_model.return_value = [str(model_file.name)]

    loaded_model = load_model(
        name="org-name/teamspace/model-name",
        download_dir=str(tmp_path),
    )
    mock_download_model.assert_called_once_with(
        name="org-name/teamspace/model-name", download_dir=str(tmp_path), progress_bar=True
    )
    assert isinstance(loaded_model, keras.models.Model)
