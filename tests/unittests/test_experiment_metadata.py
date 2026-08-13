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
from litlogger.primitives import Metadata
from litlogger.series import Series
from litlogger.session import ExperimentSession

experiment_module = sys.modules["litlogger.experiment"]


def _session_of(exp):
    """Build a session view over the experiment's current mock infrastructure."""

    def part(name):
        value = getattr(exp, name, None)
        return value if value is not None else MagicMock()

    metrics_api = part("_metrics_api")
    return ExperimentSession(
        client=metrics_api.client,
        metrics_api=metrics_api,
        media_api=part("_media_api"),
        artifacts_api=part("_artifacts_api"),
        teamspace=part("_teamspace"),
        experiment=exp,
        queue=part("_metrics_queue"),
        stats=part("_stats"),
        store_step=bool(getattr(exp, "store_step", True)),
        store_created_at=bool(getattr(exp, "store_created_at", False)),
        last_steps=getattr(exp, "_resumed_steps", None) or {},
        background=getattr(exp, "_manager", None),
    )


def _make_exp(**overrides):
    """Create a MagicMock wired for the dict-like experiment API."""
    exp = MagicMock(spec=Experiment)
    exp.name = "exp"
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
    exp._metrics_api = MagicMock()
    exp._media_api = MagicMock()
    exp._artifacts_api = MagicMock()
    exp._teamspace = MagicMock()
    exp._metrics_store = MagicMock()
    exp._metrics_store.id = "store-1"
    exp._metrics_store.name = "exp"
    exp._metrics_store.tags = []
    exp._metrics_store.cluster_id = "acc-1"
    # Metadata writes re-read the store from the API; keep the seeded one.
    exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store
    # Live session view so tests can reseed infrastructure after the factory.
    type(exp)._session = property(lambda self: _session_of(self))

    type(exp).__getitem__ = lambda self, key: Experiment.__getitem__(self, key)
    type(exp).__setitem__ = lambda self, key, value: Experiment.__setitem__(self, key, value)

    exp.update = lambda data: Experiment.update(exp, data)
    exp._ensure_series = lambda key: Experiment._ensure_series(exp, key)
    exp._register_key_type = lambda key, kt: Experiment._register_key_type(exp, key, kt)
    exp._log_metric_value = lambda key, value, step=None: Experiment._log_metric_value(exp, key, value, step=step)
    exp._coerce_static_value = lambda key, value: Experiment._coerce_static_value(exp, key, value)
    exp._coerce_series_value = lambda key, value, index, step: Experiment._coerce_series_value(
        exp, key, value, index, step
    )

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
        exp._metrics_api.update_experiment_metrics.assert_called_once()

    def test_setitem_multiple_keys(self):
        exp = _make_exp()
        exp["model"] = "resnet50"
        exp["dataset"] = "imagenet"

        assert exp._metadata_values["model"] == "resnet50"
        assert exp._metadata_values["dataset"] == "imagenet"
        assert exp._metrics_api.update_experiment_metrics.call_count == 2

    def test_overwrite_same_type(self):
        """Overwriting a metadata key with another string is allowed."""
        exp = _make_exp()
        exp["lr"] = "0.001"
        exp["lr"] = "0.01"

        assert exp._metadata_values["lr"] == "0.01"
        assert exp._metrics_api.update_experiment_metrics.call_count == 2

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
        exp._metrics_store.name = "test"
        exp._metrics_store.tags = []
        exp._metrics_api = MagicMock()
        # The metadata write re-reads the store from the API before merging
        exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts_123"

        # Wire metadata property
        type(exp).metadata = Experiment.metadata

        Metadata("lr", "0.001").log(_session_of(exp))

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
        type(exp)._session = property(lambda self: _session_of(self))
        exp._metrics_api = MagicMock()

        tag1 = MagicMock()
        tag1.name = "model"
        tag1.value = "resnet50"
        tag1.from_code = True

        tag2 = MagicMock()
        tag2.name = "system_tag"
        tag2.value = "auto"
        tag2.from_code = False

        exp._metrics_store = MagicMock()
        exp._metrics_store.name = "exp"
        exp._metrics_store.tags = [tag1, tag2]
        exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store

        result = Experiment.metadata.fget(exp)
        assert result == {"model": "resnet50"}
        assert "system_tag" not in result

    def test_metadata_empty(self):
        exp = MagicMock(spec=Experiment)
        type(exp)._session = property(lambda self: _session_of(self))
        exp._metrics_api = MagicMock()
        exp._metrics_store = MagicMock()
        exp._metrics_store.name = "exp"
        exp._metrics_store.tags = []
        exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store

        result = Experiment.metadata.fget(exp)
        assert result == {}

    def test_metadata_no_tags_attr(self):
        exp = MagicMock(spec=Experiment)
        type(exp)._session = property(lambda self: _session_of(self))
        exp._metrics_api = MagicMock()
        exp._metrics_store = MagicMock(spec=["name"])  # no .tags
        exp._metrics_store.name = "exp"
        exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store

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
        type(exp)._session = property(lambda self: _session_of(self))

        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._media_api = MagicMock()
        exp._media_api.list_media.return_value = []
        exp._artifacts_api = MagicMock()
        exp._artifacts_api.list_experiment_artifacts.return_value = None
        exp._merge_restored = lambda restored: Experiment._merge_restored(exp, restored)

        tag = MagicMock()
        tag.name = "model"
        tag.value = "resnet50"
        tag.from_code = True

        non_code_tag = MagicMock()
        non_code_tag.name = "system"
        non_code_tag.value = "auto"
        non_code_tag.from_code = False

        exp._metrics_store = MagicMock()
        exp._metrics_store.name = "exp"
        exp._metrics_store.tags = [tag, non_code_tag]
        exp._metrics_store.artifacts = []
        # The rebuild re-reads the store from the API first; keep the seeded one.
        exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store
        exp._create_download_fn = MagicMock()
        exp._resumed_steps = {}

        Experiment._rebuild_state(exp)

        assert exp._key_types["model"] == "metadata"
        assert exp._metadata_values["model"] == "resnet50"
        # Non-code tags are not rebuilt
        assert "system" not in exp._key_types

    def test_rebuilds_metric_key_types(self):
        exp = MagicMock(spec=Experiment)
        type(exp)._session = property(lambda self: _session_of(self))
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.name = "exp"
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store
        exp._metrics_api.get_metric_values.return_value = {}
        exp._resumed_steps = {"loss": 10, "acc": 5}
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._artifacts_api = MagicMock()
        exp._artifacts_api.list_experiment_artifacts.return_value = None
        exp._merge_restored = lambda restored: Experiment._merge_restored(exp, restored)
        exp._media_api = MagicMock()
        exp._media_api.list_media.return_value = []

        Experiment._rebuild_state(exp)

        assert exp._key_types["loss"] == "metric"
        assert exp._key_types["acc"] == "metric"

    def test_rebuild_hydrates_metric_values(self):
        exp = MagicMock(spec=Experiment)
        type(exp)._session = property(lambda self: _session_of(self))
        exp._key_types = {}
        exp._metadata_values = {}
        exp._static_files = {}
        exp._series = {}
        exp._metrics_store = MagicMock()
        exp._metrics_store.name = "exp"
        exp._metrics_store.tags = []
        exp._metrics_store.artifacts = []
        exp._metrics_api = MagicMock()
        exp._metrics_api.get_experiment_metrics_by_name.return_value = exp._metrics_store
        exp._metrics_api.get_metric_values.return_value = {"train/loss": [1.0, 0.5, 0.333]}
        exp._resumed_steps = {"train/loss": 2}
        exp._teamspace = MagicMock()
        exp._teamspace.id = "ts-1"
        exp._artifacts_api = MagicMock()
        exp._artifacts_api.list_experiment_artifacts.return_value = None
        exp._merge_restored = lambda restored: Experiment._merge_restored(exp, restored)
        exp._media_api = MagicMock()
        exp._media_api.list_media.return_value = []

        Experiment._rebuild_state(exp)

        assert exp._key_types["train/loss"] == "metric"
        series = exp._series["train/loss"]
        assert isinstance(series, Series)
        assert series._type == "metric"
        assert series._values == [1.0, 0.5, 0.333]

    def test_getitem_rebuilds_missing_series_for_metric_key(self):
        exp = _make_exp()
        exp._key_types["loss"] = "metric"

        series = exp["loss"]

        assert isinstance(series, Series)
        assert series._type == "metric"
        assert exp._series["loss"] is series


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

    def test_setitem_over_untyped_series_ok(self):
        """Assigning metadata to a key with an untyped (empty) series replaces the series."""
        exp = _make_exp()
        # Access creates an empty series
        _ = exp["maybe"]
        assert "maybe" in exp._series

        exp["maybe"] = "value"
        assert exp._key_types["maybe"] == "metadata"
        assert "maybe" not in exp._series

    def test_setitem_invalid_type_raises(self):
        """Setting an unsupported type (e.g. int) via __setitem__ raises TypeError."""
        exp = _make_exp()

        with pytest.raises(TypeError, match="Can only assign"):
            exp["bad"] = 42  # type: ignore[assignment]

    def test_update_rejects_unsupported_type(self):
        exp = _make_exp()

        with pytest.raises(TypeError, match="Unsupported type"):
            exp.update({"key": object()})
