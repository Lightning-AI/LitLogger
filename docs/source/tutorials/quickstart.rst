##########
Quickstart
##########

This tutorial introduces the Experiment API through the smallest useful
workflow: create an experiment, log metadata and metrics, upload a file, and
finalize the run.

Create an Experiment
====================

.. code-block:: python

   import litlogger

   experiment = litlogger.init(name="quickstart")

:func:`litlogger.init() <litlogger.init.init>` returns an
:class:`~litlogger.experiment.Experiment` instance. That object is the main
entrypoint for the new API.

Log Metadata
============

Static string assignments become
:attr:`~litlogger.experiment.Experiment.metadata` entries:

.. code-block:: python

   experiment["model"] = "resnet50"
   experiment["dataset"] = "cifar10"
   experiment["optimizer"] = "adamw"

Log Metrics
===========

Time-series metrics use :class:`~litlogger.series.Series` through
:meth:`~litlogger.series.Series.append` or
:meth:`~litlogger.series.Series.extend` on ``experiment["key"]``:

.. code-block:: python

   for step in range(10):
       experiment["train/loss"].append(1.0 / (step + 1), step=step)
       experiment["train/accuracy"].append(step / 10.0, step=step)

You can also batch values:

.. code-block:: python

   experiment["val/loss"].extend([0.5, 0.4, 0.3], start_step=100)

Log a File
==========

Use :class:`~litlogger.media.File` for static artifacts:

.. code-block:: python

   from litlogger import File

   experiment["config"] = File("config.yaml")

Inspect Logged Data
===================

The experiment exposes runtime views through
:attr:`~litlogger.experiment.Experiment.metadata`,
:attr:`~litlogger.experiment.Experiment.metrics`,
:attr:`~litlogger.experiment.Experiment.artifacts`, and
:attr:`~litlogger.experiment.Experiment.url`:

.. code-block:: python

   print(experiment.metadata)
   print(experiment.metrics)
   print(experiment.artifacts)
   print(experiment.url)

Finalize
========

Use :meth:`~litlogger.experiment.Experiment.finalize` when the run is complete.

.. code-block:: python

   experiment.finalize()

Next Steps
==========

- :doc:`file_media_model` for files, media, and model logging
- :doc:`lightning` for Lightning/Fabric integration
- :doc:`complete_workflow` for a log-and-fetch workflow
- :doc:`../guide/standalone` for the full standalone guide
