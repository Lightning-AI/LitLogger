# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Focused tests for internal experiment support helpers."""

from unittest.mock import MagicMock

from lightning_sdk.lightning_cloud.openapi import V1MediaType
from litlogger.experiment import Experiment
from litlogger.experiment_support import ExperimentIOSupport, ExperimentStateSupport
from litlogger.media import Model, Text
from litlogger.series import Series
from litlogger.types import MediaType


class TestExperimentIOSupport:
    """Targeted routing and metadata tests for ExperimentIOSupport."""

    def test_set_metadata_value_uses_concrete_code_tags(self):
        exp = MagicMock(spec=Experiment)
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-123"
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-123"
        exp._update_metrics_store = MagicMock()
        exp._code_tags.return_value = {"wrong": "value"}

        tag = MagicMock()
        tag.name = "lr"
        tag.value = "0.001"
        tag.from_code = True
        exp._metrics_store.tags = [tag]

        ExperimentIOSupport.set_metadata_value(exp, "batch_size", "32")

        metadata = exp._metrics_api.update_experiment_metrics.call_args.kwargs["metadata"]
        assert metadata == {"lr": "0.001", "batch_size": "32"}

    def test_set_static_file_routes_model_by_media_type(self):
        exp = MagicMock(spec=Experiment)
        exp._upload_model_value = MagicMock()
        exp._upload_media_value = MagicMock()
        exp._artifacts_api = MagicMock()
        exp._stats = MagicMock()
        exp._stats.artifacts_logged = 0

        model = Model("checkpoint.ckpt")
        assert model._media_type == MediaType.MODEL

        ExperimentIOSupport.set_static_file(exp, "model", model)

        exp._upload_model_value.assert_called_once_with("model", model)
        exp._upload_media_value.assert_not_called()

    def test_log_file_series_value_routes_text_with_exact_key_name(self):
        exp = MagicMock(spec=Experiment)
        exp._upload_media_value = MagicMock()
        exp._upload_model_value = MagicMock()
        exp._stats = MagicMock()
        exp._stats.media_logged = 0

        text = Text("hello")

        ExperimentIOSupport.log_file_series_value(exp, "logs", text, 2, step=7)

        exp._upload_media_value.assert_called_once_with("logs", text, name="logs", step=7)
        exp._upload_model_value.assert_not_called()


class TestExperimentStateSupport:
    """Focused reconstruction tests for ExperimentStateSupport."""

    def test_wrap_media_file_returns_text_wrapper(self):
        exp = MagicMock(spec=Experiment)

        wrapped = ExperimentStateSupport.wrap_media_file(exp, "logs/0", V1MediaType.TEXT)

        assert isinstance(wrapped, Text)
        assert wrapped.path == "logs/0"

    def test_rebuild_state_reconstructs_sorted_text_series(self):
        media1 = MagicMock()
        media1.name = "logs/1"
        media1.storage_path = "remote/logs/1"
        media1.id = "m1"
        media1.cluster_id = "acc-1"
        media1.media_type = V1MediaType.TEXT

        media0 = MagicMock()
        media0.name = "logs/0"
        media0.storage_path = "remote/logs/0"
        media0.id = "m0"
        media0.cluster_id = "acc-1"
        media0.media_type = V1MediaType.TEXT

        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_trackers_from_metrics_store.return_value = []
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._media_api = MagicMock()
        exp._media_api.client.lit_logger_service_list_lit_logger_media.return_value.media = [media1, media0]
        exp._wrap_media_file = lambda media_name, media_type: ExperimentStateSupport.wrap_media_file(
            exp, media_name, media_type
        )
        exp._create_media_download_fn = lambda storage_path, cloud_account=None: (
            ExperimentStateSupport.create_media_download_fn(exp, storage_path, cloud_account)
        )

        ExperimentStateSupport.rebuild_state(exp)

        assert exp._key_types["logs"] == "file_series"
        assert isinstance(exp._series["logs"], Series)
        assert [item.path for item in exp._series["logs"]] == ["logs/0", "logs/1"]

    def test_rebuild_state_reconstructs_same_name_media_series(self):
        media1 = MagicMock()
        media1.name = "logs"
        media1.step = 1
        media1.storage_path = "remote/logs-1"
        media1.id = "m1"
        media1.cluster_id = "acc-1"
        media1.media_type = V1MediaType.TEXT

        media0 = MagicMock()
        media0.name = "logs"
        media0.step = 0
        media0.storage_path = "remote/logs-0"
        media0.id = "m0"
        media0.cluster_id = "acc-1"
        media0.media_type = V1MediaType.TEXT

        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store-1"
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_trackers_from_metrics_store.return_value = []
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._media_api = MagicMock()
        exp._media_api.client.lit_logger_service_list_lit_logger_media.return_value.media = [media1, media0]
        exp._wrap_media_file = lambda media_name, media_type: ExperimentStateSupport.wrap_media_file(
            exp, media_name, media_type
        )
        exp._create_media_download_fn = lambda storage_path, cloud_account=None: (
            ExperimentStateSupport.create_media_download_fn(exp, storage_path, cloud_account)
        )

        ExperimentStateSupport.rebuild_state(exp)

        assert exp._key_types["logs"] == "file_series"
        assert isinstance(exp._series["logs"], Series)
        assert [item.path for item in exp._series["logs"]] == ["logs", "logs"]
