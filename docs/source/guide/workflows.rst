#########
Workflows
#########

This guide maps the LitLogger feature set to concrete workflows so users can
quickly choose the right API surface.

Standalone New API
==================

Recommended for new code.

- initialize with :func:`litlogger.init.init`
- work with the returned :class:`~litlogger.experiment.Experiment`
- log metadata with ``experiment["key"] = "value"``
- log metrics with ``experiment["key"].append(...)`` and ``extend(...)``
- log files, media, and models with the file-like wrappers

Legacy Module-Level API
=======================

Supported for existing standalone scripts, but the dict-style API is the main
user-facing API going forward.

- ``litlogger.log_metrics(...)``
- ``litlogger.log_file(...)``
- ``litlogger.log_model(...)``
- ``litlogger.log_model_artifact(...)``
- ``litlogger.log_metadata(...)``

Lightning and Fabric
====================

Use :class:`~litlogger.logger.LightningLogger` with a Trainer or Fabric loop.

Files, Media, and Models
========================

Use the dedicated wrappers when you want data attached to a run:

- :class:`~litlogger.media.File`
- :class:`~litlogger.media.Image`
- :class:`~litlogger.media.Text`
- :class:`~litlogger.media.Model`

Resume and Retrieval
====================

Re-initialize with the same experiment name to continue logging or inspect
existing metadata, metrics, artifacts, and media.

End-to-End Log and Fetch
========================

See :doc:`../tutorials/complete_workflow` for the full train-and-fetch flow.

Inference Logging
=================

Use LitLogger inside a LitServe endpoint to track inference metrics and
request-level telemetry.
