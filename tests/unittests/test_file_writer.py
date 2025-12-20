import os
from unittest.mock import MagicMock

from litlogger.file_writer import BinaryFileWriter
from litlogger.types import Metrics, MetricValue


def test_file_writer(tmpdir):
    # Mock the ArtifactsApi to avoid actual uploads
    mock_artifacts_api = MagicMock()

    store = BinaryFileWriter(
        log_dir=str(tmpdir),
        version="fake_timestamp",
        store_step=False,
        store_created_at=False,
        teamspace_id="project_id",
        metrics_store_id="stream_id",
        cloud_account="cloud_account",
        client=None,
    )

    # Replace the artifacts API with our mock
    store._artifacts_api = mock_artifacts_api

    store.store(
        {
            "my_metrics": Metrics(
                name="my_metrics",
                values=[
                    MetricValue(value=1.1),
                    MetricValue(value=2.2),
                    MetricValue(value=3.3),
                    MetricValue(value=4.4),
                ],
            )
        }
    )

    assert os.listdir(tmpdir) == ["my_metrics.litbin"]
    filepath = os.path.join(tmpdir, "my_metrics.litbin")
    file_size = os.stat(filepath)
    assert file_size.st_size < 150

    store.upload()

    # Verify upload was called but file is still removed
    mock_artifacts_api.upload_metrics_binary.assert_called_once()
    assert not os.path.isfile(filepath)
