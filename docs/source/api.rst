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

.. currentmodule:: litlogger.experiment

.. automethod:: Experiment.log_metrics
    :no-index:

.. automethod:: Experiment.log_metadata
    :no-index:

.. automethod:: Experiment.log_file
    :no-index:

.. automethod:: Experiment.get_file
    :no-index:

.. automethod:: Experiment.log_model
    :no-index:

.. automethod:: Experiment.get_model
    :no-index:

.. automethod:: Experiment.log_model_artifact
    :no-index:

.. automethod:: Experiment.get_model_artifact
    :no-index:

.. automethod:: Experiment.finalize
    :no-index:

----

Experiment
==========

.. currentmodule:: litlogger.experiment

.. autoclass:: Experiment
    :members:
    :show-inheritance:

----

LightningLogger
================

.. currentmodule:: litlogger.logger

.. autoclass:: LightningLogger
    :members:
    :show-inheritance:
