#############
API Reference
#############

Module-Level API
================

Top-level functions available directly on the ``litlogger`` module.

Initialization
--------------

.. currentmodule:: litlogger.init

.. autofunction:: init

.. autofunction:: finish

.. autofunction:: get_metadata

Logging Functions
-----------------

These functions are available as module-level callables after
``litlogger.init()`` is called. They delegate to the underlying
:class:`~litlogger.experiment.Experiment` instance.

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

----

Experiment
==========

.. currentmodule:: litlogger.experiment

.. autoclass:: Experiment
    :members:
    :show-inheritance:

----

Types
=====

.. currentmodule:: litlogger.types

.. autoclass:: MediaType
    :members:
    :undoc-members:

----

LightningLogger
================

.. currentmodule:: litlogger.logger

.. autoclass:: LightningLogger
    :members:
    :show-inheritance:
