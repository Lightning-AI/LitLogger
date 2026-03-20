################
Standalone Usage
################

Use LitLogger in any Python process without Lightning. For new code, the
recommended standalone workflow is the dict-style
:class:`~litlogger.experiment.Experiment` API returned by
:func:`litlogger.init() <litlogger.init.init>`.

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/litlogger/experiment_comparison_table.png
   :alt: Comparing experiment metrics in a table
   :width: 800px
   :align: center

Recommended New API
===================

.. code-block:: python

   import litlogger
   from litlogger import File

   experiment = litlogger.init(name="my-experiment")

   experiment["model"] = "resnet50"
   experiment["lr"] = "0.001"

   for step in range(100):
       experiment["train/loss"].append(1.0 / (step + 1), step=step)
       experiment["train/accuracy"].append(step / 100.0, step=step)

   experiment["config"] = File("config.yaml")
   experiment.finalize()

The new standalone API covers:

- metadata through ``experiment["key"] = "value"``
- metrics through :meth:`~litlogger.series.Series.append` and
  :meth:`~litlogger.series.Series.extend`
- artifacts, media, and models through file-like wrappers
- retrieval through :attr:`~litlogger.experiment.Experiment.metadata`,
  :attr:`~litlogger.experiment.Experiment.metrics`, and
  :attr:`~litlogger.experiment.Experiment.artifacts`

Metric Logging
==============

.. code-block:: python

   experiment["loss"].append(0.5, step=0)
   experiment["loss"].extend([0.4, 0.3, 0.2], start_step=1)

Metadata
========

Metadata can be provided at initialization time or added later:

.. code-block:: python

   experiment = litlogger.init(
       name="training-run",
       metadata={"dataset": "imagenet"},
   )

   experiment["optimizer"] = "adamw"

Files, Media, and Models
========================

.. code-block:: python

   from litlogger import File, Image, Model, Text

   experiment["config"] = File("config.yaml")
   experiment["preview"] = Image("sample.png")
   experiment["notes"] = Text("training summary")
   experiment["checkpoint"] = Model("checkpoint.ckpt")

Resume by Name
==============

Initializing with the same name reconnects to the same experiment.

.. code-block:: python

   experiment = litlogger.init(name="my-experiment")
   experiment["train/loss"].append(0.2, step=100)
   experiment.finalize()

Runtime Views
=============

Use :attr:`~litlogger.experiment.Experiment.metadata`,
:attr:`~litlogger.experiment.Experiment.metrics`,
:attr:`~litlogger.experiment.Experiment.artifacts`, and
:attr:`~litlogger.experiment.Experiment.url` to inspect the current run state.

.. code-block:: python

   print(experiment.metadata)
   print(experiment.metrics)
   print(experiment.artifacts)
   print(experiment.url)

Legacy Module-Level API
=======================

The older module-level API is still available for compatibility through
:func:`~litlogger.log_metrics`, :func:`~litlogger.log_file`,
:func:`~litlogger.log_metadata`, and :func:`~litlogger.finalize`:

.. code-block:: python

   import litlogger

   litlogger.init(name="legacy-script")
   litlogger.log_metrics({"loss": 0.5}, step=0)
   litlogger.log_file("config.yaml")
   litlogger.log_metadata({"model": "resnet50"})
   litlogger.finalize()

This API is useful for existing scripts, but the dict-style API is the primary
user-facing workflow.

Operational Options
===================

Common standalone options:

- ``teamspace=...`` to select a teamspace explicitly
- ``save_logs=True`` to capture terminal output as ``console_output.txt``
- ``light_color=...`` and ``dark_color=...`` to override chart colors
- ``store_step=False`` or ``store_created_at=True`` to control tracker data

Related Docs
============

- :doc:`../tutorials/quickstart`
- :doc:`../tutorials/file_media_model`
- :doc:`workflows`
