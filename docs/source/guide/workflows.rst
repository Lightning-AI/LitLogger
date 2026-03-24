
#########
Workflows
#########

This guide maps the LitLogger feature set to concrete workflows so users can
quickly choose the right API surface.

Experiment API
==============

Recommended for new code.

- Initialize with :func:`litlogger.init.init`
- Work with the returned :class:`~litlogger.experiment.Experiment`
- Log metadata with ``experiment["key"] = "value"``
- Log metrics with :meth:`~litlogger.series.Series.append` and
  :meth:`~litlogger.series.Series.extend` on ``experiment["key"]``
- Log files, media, and models with the file-like wrappers

Legacy Module-Level API
=======================

Supported for existing standalone scripts, but the Experiment API is the main
user-facing API going forward.

- :func:`~litlogger.log_metrics`
- :func:`~litlogger.log_file`
- :func:`~litlogger.log_model`
- :func:`~litlogger.log_model_artifact`
- :func:`~litlogger.log_metadata`

PyTorch Lightning and Lightning Fabric
======================================

Use :class:`lightning:lightning.pytorch.loggers.LitLogger` with a Trainer or
Fabric loop. The local :class:`~litlogger.logger.LightningLogger` alias is
deprecated.

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
existing :attr:`~litlogger.experiment.Experiment.metadata`,
:attr:`~litlogger.experiment.Experiment.metrics`,
:attr:`~litlogger.experiment.Experiment.artifacts`, and media.

End-to-End Log and Fetch
========================

See :doc:`../tutorials/complete_workflow` for the full train-and-fetch flow.

Inference Logging
=================

Use LitLogger inside a LitServe endpoint to track inference metrics and
request-level telemetry.
