# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Guards for the public API surface."""

import importlib
import sys

import pytest

import litlogger

#: The exports that existed before the primitives refactor. These must never
#: shrink or change; only additive changes are allowed.
LEGACY_EXPORTS = {
    "Experiment",
    "File",
    "Image",
    "Model",
    "Text",
    "Video",
    "experiment",
    "finalize",
    "finish",
    "get_metadata",
    "init",
    "log_metadata",
}

#: Additive exports introduced by the primitives refactor.
PRIMITIVE_EXPORTS = {
    "ExperimentSession",
    "Metadata",
    "Metric",
    "Primitive",
}

#: Present only when the optional PyTorch Lightning integration imports.
OPTIONAL_EXPORTS = {"LightningLogger"}


def test_all_is_exactly_legacy_plus_primitives():
    assert set(litlogger.__all__) - OPTIONAL_EXPORTS == LEGACY_EXPORTS | PRIMITIVE_EXPORTS


def test_unexported_module_attributes_survive():
    # Not in __all__, but real module attributes the ecosystem relies on.
    for name in (
        "log",
        "log_metrics",
        "log_file",
        "get_file",
        "log_model",
        "get_model",
        "log_model_artifact",
        "get_model_artifact",
    ):
        assert hasattr(litlogger, name), name


def test_primitive_exports_are_the_real_classes():
    from litlogger.primitives import Metadata, Metric, Primitive
    from litlogger.session import ExperimentSession

    assert litlogger.Metric is Metric
    assert litlogger.Metadata is Metadata
    assert litlogger.Primitive is Primitive
    assert litlogger.ExperimentSession is ExperimentSession


def test_media_module_is_a_deprecated_compatibility_facade():
    from litlogger.primitives import File, Image, Model, Text, Video

    sys.modules.pop("litlogger.media", None)
    with pytest.warns(DeprecationWarning, match="litlogger.media is deprecated"):
        media = importlib.import_module("litlogger.media")

    assert (media.File, media.Image, media.Model, media.Text, media.Video) == (File, Image, Model, Text, Video)


def test_file_hierarchy_satisfies_primitive_protocol():
    from litlogger import File, Image, Metadata, Metric, Model, Primitive, Text, Video

    for primitive in (
        File("a.txt"),
        Image("a.png"),
        Video("a.mp4"),
        Text("hello"),
        Model("a.ckpt"),
        Metric("loss", 1.0),
        Metadata("k", "v"),
    ):
        assert isinstance(primitive, Primitive), type(primitive).__name__
