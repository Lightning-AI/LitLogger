#############
API Reference
#############

This reference documents the public LitLogger surface:

- initialization functions
- module-level compatibility helpers
- the :class:`~litlogger.experiment.Experiment` class
- file, media, and model wrappers
- series and public enums
- :class:`~litlogger.logger.LightningLogger` (deprecated compatibility alias)

Initialization
==============

.. currentmodule:: litlogger.init

.. autofunction:: init

.. autofunction:: finish

.. autofunction:: get_metadata

Module-Level Helpers
====================

These helpers are available on the top-level ``litlogger`` module after
initialization. They exist primarily for standalone compatibility workflows.

.. currentmodule:: litlogger

.. autofunction:: log

.. autofunction:: log_metrics

.. autofunction:: log_metadata

.. autofunction:: log_file

.. autofunction:: get_file

.. autofunction:: log_model

.. autofunction:: get_model

.. autofunction:: log_model_artifact

.. autofunction:: get_model_artifact

.. autofunction:: finalize

Experiment
==========

.. currentmodule:: litlogger.experiment

.. autoclass:: Experiment
   :members:
   :inherited-members:

Media Wrappers
==============

.. currentmodule:: litlogger.media

.. autoclass:: File
   :members:
   :show-inheritance:

.. autoclass:: Image
   :members:
   :show-inheritance:

.. autoclass:: Text
   :members:
   :show-inheritance:

.. autoclass:: Model
   :members:
   :show-inheritance:

Series
======

.. currentmodule:: litlogger.series

.. autoclass:: Series
   :members:
   :show-inheritance:

Types
=====

.. currentmodule:: litlogger.types

.. autoclass:: MediaType
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PhaseType
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MetricValue
   :members:
   :show-inheritance:

.. autoclass:: Metrics
   :members:
   :show-inheritance:

LightningLogger
===============

.. currentmodule:: litlogger.logger

Deprecated compatibility alias for Lightning/Fabric integration. Prefer
:class:`lightning:lightning.pytorch.loggers.LitLogger`, or use
:func:`litlogger.init.init` and the dict-style
:class:`~litlogger.experiment.Experiment` API for standalone usage.

.. autoclass:: LightningLogger
   :members:
   :show-inheritance:
