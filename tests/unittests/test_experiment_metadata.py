# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for adding and retrieving metadata via the experiment dict-like API."""

import sys
from unittest.mock import MagicMock

import pytest
from litlogger.experiment import Experiment
from litlogger.media import File

experiment_module = sys.modules["litlogger.experiment"]


def _make_exp(**overrides):
    """Create a MagicMock wired for the dict-like experiment API."""
    exp = MagicMock(spec=Experiment)
    exp._series = {}
    exp._key_types = {}
    exp._metadata_values = {}
    exp._static_files = {}
    exp._manager = MagicMock()
    exp._manager.exception = None
    exp.store_step = True
    exp.store_created_at = False
    exp._metrics_queue = MagicMock()
    exp._stats = MagicMock()

    type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)
    type(exp).__setitem__ = lambda self, key, value: Experiment.__setitem__(self, key, value)

    exp.update = lambda data: Experiment.update(exp, data)
    exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
    exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
    exp._log_metric_value = lambda key, value, step=None: Experiment._log_metric_value(exp, key, value, step=step)
    exp._set_metadata_value = MagicMock()
    exp._set_static_file = MagicMock()

    for k, v in overrides.items():
        setattr(exp, k, v)
    return exp


# ---------------------------------------------------------------------------
# Adding metadata
# ---------------------------------------------------------------------------


class TestAddMetadataSetitem:
    """Test experiment['key'] = 'value' for metadata."""

    def test_setitem_string(self):
        exp = _make_exp()
        exp["model"] = "resnet50"

        assert exp._key_types["model"] == "metadata"
        assert exp._metadata_values["model"] == "resnet50"
        exp._set_metadata_value.assert_called_once_with("model", "resnet50")

    def test_setitem_multiple_keys(self):
        exp = _make_exp()
        exp["model"] = "resnet50"
        exp["dataset"] = "imagenet"

        assert exp._metadata_values["model"] == "resnet50"
        assert exp._metadata_values["dataset"] == "imagenet"
        assert exp._set_metadata_value.call_count == 2

    def test_overwrite_same_type(self):
        """Overwriting a metadata key with another string is allowed."""
        exp = _make_exp()
        exp["lr"] = "0.001"
        exp["lr"] = "0.01"

        assert exp._metadata_values["lr"] == "0.01"
        assert exp._set_metadata_value.call_count == 2

    def test_empty_string_value(self):
        exp = _make_exp()
        exp["notes"] = ""

        assert exp._metadata_values["notes"] == ""


class TestAddMetadataUpdate:
    """Test experiment.update() for metadata."""

    def test_update_single(self):
        exp = _make_exp()
        exp.update({"model": "resnet50"})

        assert exp._metadata_values["model"] == "resnet50"

    def test_update_multiple(self):
        exp = _make_exp()
        exp.update({"model": "resnet50", "dataset": "imagenet"})

        assert exp._metadata_values["model"] == "resnet50"
        assert exp._metadata_values["dataset"] == "imagenet"

    def test_update_mixed_with_metrics(self):
        exp = _make_exp()
        exp.update({"model": "resnet50", "loss": 0.5})

        assert exp._metadata_values["model"] == "resnet50"
        assert exp["loss"][0] == 0.5


class TestAddMetadataApiCall:
    """Test that _set_metadata_value pushes to the API."""

    def test_calls_update_experiment_metrics(self):
        from litlogger.background import PhaseType

        exp = MagicMock(spec=Experiment)
        exp._metrics_store = MagicMock()
        exp._metrics_store.id = "store_123"
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts_123"
        exp._update_metrics_store = MagicMock()

        # Wire metadata property
        type(exp).metadata = Experiment.metadata

        Experiment._set_metadata_value(exp, "lr", "0.001")

        exp._metrics_api.update_experiment_metrics.assert_called_once()
        call_kwargs = exp._metrics_api.update_experiment_metrics.call_args.kwargs
        assert call_kwargs["teamspace_id"] == "ts_123"
        assert call_kwargs["metrics_store_id"] == "store_123"
        assert call_kwargs["phase"] == PhaseType.RUNNING
        assert call_kwargs["metadata"]["lr"] == "0.001"


# ---------------------------------------------------------------------------
# Retrieving metadata
# ---------------------------------------------------------------------------


class TestRetrieveMetadataByKey:
    """Test experiment['key'] retrieval for metadata."""

    def test_getitem_returns_string(self):
        exp = _make_exp()
        exp["model"] = "resnet50"

        result = exp["model"]
        assert result == "resnet50"
        assert isinstance(result, str)

    def test_getitem_returns_latest_value(self):
        exp = _make_exp()
        exp["lr"] = "0.001"
        exp["lr"] = "0.01"

        assert exp["lr"] == "0.01"


class TestRetrieveMetadataProperty:
    """Test experiment.metadata property."""

    def test_metadata_returns_code_tags(self):
        exp = MagicMock(spec=Experiment)
        exp._update_metrics_store = MagicMock()

        tag1 = MagicMock()
        tag1.name = "model"
        tag1.value = "resnet50"
        tag1.from_code = True

        tag2 = MagicMock()
        tag2.name = "system_tag"
        tag2.value = "auto"
        tag2.from_code = False

        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = [tag1, tag2]

        result = Experiment.metadata.fget(exp)
        assert result == {"model": "resnet50"}
        assert "system_tag" not in result

    def test_metadata_empty(self):
        exp = MagicMock(spec=Experiment)
        exp._update_metrics_store = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = []

        result = Experiment.metadata.fget(exp)
        assert result == {}

    def test_metadata_no_tags_attr(self):
        exp = MagicMock(spec=Experiment)
        exp._update_metrics_store = MagicMock()
        exp._metrics_store = MagicMock(spec=[])  # no .tags

        result = Experiment.metadata.fget(exp)
        assert result == {}


# ---------------------------------------------------------------------------
# Rebuild state (metadata from resumed experiment)
# ---------------------------------------------------------------------------


class TestRebuildStateMetadata:
    """Test that _rebuild_state populates metadata from remote tags."""

    def test_rebuilds_code_tags(self):
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_trackers_from_metrics_store.return_value = []

        tag = MagicMock()
        tag.name = "model"
        tag.value = "resnet50"
        tag.from_code = True

        non_code_tag = MagicMock()
        non_code_tag.name = "system"
        non_code_tag.value = "auto"
        non_code_tag.from_code = False

        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = [tag, non_code_tag]
        exp._metrics_store.artifacts = []
        exp._create_download_fn = MagicMock()

        Experiment._rebuild_state(exp)

        assert exp._key_types["model"] == "metadata"
        assert exp._metadata_values["model"] == "resnet50"
        # Non-code tags are not rebuilt
        assert "system" not in exp._key_types

    def test_rebuilds_metric_key_types(self):
        exp = MagicMock(spec=Experiment)
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_trackers_from_metrics_store.return_value = ["loss", "acc"]
        exp._create_download_fn = MagicMock()

        Experiment._rebuild_state(exp)

        assert exp._key_types["loss"] == "metric"
        assert exp._key_types["acc"] == "metric"


# ---------------------------------------------------------------------------
# Type conflicts
# ---------------------------------------------------------------------------


class TestMetadataTypeConflicts:
    """Test that metadata keys conflict properly with other types."""

    def test_cannot_assign_file_to_metadata_key(self):
        exp = _make_exp()
        exp["tag"] = "v1"

        with pytest.raises(KeyError, match="already used"):
            exp["tag"] = File("data.csv")

    def test_cannot_assign_string_to_file_key(self):
        exp = _make_exp()
        exp["config"] = File("config.yaml")

        with pytest.raises(KeyError, match="already used"):
            exp["config"] = "oops"

    def test_cannot_assign_string_to_metric_key(self):
        exp = _make_exp()
        exp["loss"].append(0.5)

        with pytest.raises(KeyError, match="time series"):
            exp["loss"] = "oops"

    def test_update_rejects_unsupported_type(self):
        exp = _make_exp()

        with pytest.raises(TypeError, match="Unsupported type"):
            exp.update({"key": object()})
