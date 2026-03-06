##########
Module API
##########

Top-level functions available directly on the ``litlogger`` module.

Initialization
--------------

.. currentmodule:: litlogger.init

.. autofunction:: init
    :no-index:

.. autofunction:: finish
    :no-index:

.. autofunction:: get_metadata
    :no-index:

Experiment Methods
------------------

These methods are available on the :class:`~litlogger.experiment.Experiment` instance
and also as module-level callables after ``litlogger.init()`` is called.

.. currentmodule:: litlogger.experiment

.. automethod:: Experiment.log_metrics
    :no-index:

.. automethod:: Experiment.log_file
    :no-index:

.. automethod:: Experiment.log_model
    :no-index:

.. automethod:: Experiment.log_model_artifact
    :no-index:

.. automethod:: Experiment.finalize
    :no-index:
