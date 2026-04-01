"""Integration tests for vendored model-registry uploads through litlogger."""

import tempfile
import uuid
from pathlib import Path
from time import sleep

import litlogger
import pytest
from lightning_sdk.lightning_cloud.openapi.models import LitLoggerServiceDeleteMetricsStreamBody
from litlogger.api.client import LitRestClient


def _delete_experiment(exp) -> None:
    client = LitRestClient()
    client.lit_logger_service_delete_metrics_stream(
        project_id=exp._teamspace.id,
        body=LitLoggerServiceDeleteMetricsStreamBody(ids=[exp._metrics_store.id]),
    )


def _delete_model(exp, model_name: str) -> None:
    client = LitRestClient()
    try:
        model_record = client.models_store_get_model_by_name(
            project_owner_name=exp._teamspace.owner.name,
            project_name=exp._teamspace.name,
            model_name=model_name,
        )
        client.models_store_delete_model(project_id=exp._teamspace.id, model_id=model_record.id)
    except Exception:
        pass


def _wait_for_model(exp, model_name: str):
    client = LitRestClient()
    last_exception = None
    for attempt in range(30):
        try:
            return client.models_store_get_model_by_name(
                project_owner_name=exp._teamspace.owner.name,
                project_name=exp._teamspace.name,
                model_name=model_name,
            )
        except Exception as ex:
            last_exception = ex
            if attempt < 29:
                sleep(1)
    raise AssertionError(f"Failed to find uploaded model '{model_name}'. Last error: {last_exception}")


@pytest.mark.cloud()
def test_top_level_log_model_registers_pickle_model():
    """Test top-level model logging registers a pickle-backed model."""
    experiment_name = f"registry_top_level_upload-{uuid.uuid4().hex}"
    model_version = "test-v1"
    payload = {"epoch": 5, "metrics": {"loss": 0.123, "accuracy": 0.987}}

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger", root_dir=tmpdir)

        try:
            litlogger.log_model(payload, staging_dir=tmpdir, verbose=False, version=model_version)
            model_record = _wait_for_model(exp, experiment_name)
            assert getattr(model_record, "id", None)
            assert model_record.latest_version.metrics_stream_id == exp.id
        finally:
            litlogger.finalize()
            _delete_experiment(exp)
            _delete_model(exp, experiment_name)


@pytest.mark.cloud()
def test_top_level_log_model_artifact_registers_model():
    """Test top-level model-artifact logging registers a model artifact."""
    experiment_name = f"registry_artifact_upload-{uuid.uuid4().hex}"
    model_version = "test-v1"

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = litlogger.init(name=experiment_name, teamspace="oss-litlogger", root_dir=tmpdir)
        artifact_dir = Path(tmpdir) / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "weights.txt").write_text("0.1,0.2,0.3\n", encoding="utf-8")
        (artifact_dir / "config.json").write_text('{"hidden_size": 128}', encoding="utf-8")

        try:
            litlogger.log_model_artifact(str(artifact_dir), verbose=False, version=model_version)
            model_record = _wait_for_model(exp, experiment_name)
            assert getattr(model_record, "id", None)
            assert model_record.latest_version.metrics_stream_id == exp.id
        finally:
            litlogger.finalize()
            _delete_experiment(exp)
            _delete_model(exp, experiment_name)
