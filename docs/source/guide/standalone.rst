################
Standalone Usage
################

Use litlogger as a standalone experiment tracker in any Python script -- no
framework required.

Quick Start
===========

Initialize an experiment, log metrics in a loop, and finalize when done:

.. code-block:: python

   import litlogger

   litlogger.init(name="my-experiment")

   for step in range(100):
       loss = 1.0 / (step + 1)
       litlogger.log({"loss": loss, "accuracy": step / 100.0}, step=step)

   litlogger.finalize()

After calling ``init()``, a URL is printed where you can view live charts on
`lightning.ai <https://lightning.ai>`_.


Tracking Metrics
================

Use ``litlogger.log()`` (or equivalently ``litlogger.log_metrics()``) to send
metric values. Metrics are buffered and uploaded in the background so your
training loop is never blocked.

.. code-block:: python

   # Dictionary style
   litlogger.log({"train/loss": 0.25, "train/acc": 0.91}, step=10)

   # Keyword style
   litlogger.log({}, step=10, val_loss=0.30, val_acc=0.88)


Metadata
========

Log Metadata
------------

Metadata lets you tag experiments with key-value pairs like hyperparameters,
model names, or dataset versions. This makes it easy to filter, compare, and
identify experiments.

Pass a ``metadata`` dictionary to ``init()``:

.. code-block:: python

   import litlogger

   litlogger.init(
       name="training-run",
       metadata={
           "model": "GPT-2",
           "dataset": "WikiText",
           "learning_rate": "0.0001",
       },
   )

Retrieve Metadata
-----------------

Retrieve metadata from any experiment using ``litlogger.get_metadata()`` or
the ``experiment.metadata`` property:

.. code-block:: python

   import litlogger

   litlogger.init(name="training-run")

   # Get metadata using the global function
   metadata = litlogger.get_metadata()
   print(metadata)  # {"model": "GPT-2", "dataset": "WikiText", ...}

   # Or access via the experiment object
   print(litlogger.experiment.metadata)


Resume Experiments
==================

LitLogger automatically resumes experiments when you initialize with the same
name. This is useful for continuing interrupted training or adding more data.

.. code-block:: python

   import litlogger

   # First run - creates new experiment
   litlogger.init(name="my-experiment")
   for i in range(10):
       litlogger.log_metrics({"loss": 1.0 / (i + 1)}, step=i)
   litlogger.finalize()

   # Later run - resumes the same experiment
   litlogger.init(name="my-experiment")
   for i in range(10, 20):
       litlogger.log_metrics({"loss": 1.0 / (i + 1)}, step=i)
   litlogger.finalize()


Custom Chart Colors
===================

Override the default random colors with hex values:

.. code-block:: python

   exp = litlogger.init(
       name="my-experiment",
       light_color="#FF5733",
       dark_color="#3498DB",
   )


Terminal Log Capture
====================

Set ``save_logs=True`` to capture your script's terminal output and upload it
as a file artifact (``console_output.txt``):

.. code-block:: python

   litlogger.init(name="my-experiment", save_logs=True)


Selecting a Teamspace
======================

By default litlogger uses your default teamspace. Pass a name explicitly to log
into a different one:

.. code-block:: python

   litlogger.init(name="my-experiment", teamspace="research-team")


Finalizing
==========

``litlogger.finalize()`` flushes all remaining metrics to the cloud. It is
called automatically via an ``atexit`` handler, but calling it explicitly
guarantees that everything is uploaded before the process exits:

.. code-block:: python

   litlogger.finalize()
   # or equivalently:
   litlogger.finish()


Using the Experiment Object
===========================

``litlogger.init()`` returns an :class:`~litlogger.experiment.Experiment`
instance that you can use directly:

.. code-block:: python

   exp = litlogger.init(name="my-experiment")

   exp.log_metrics({"loss": 0.5}, step=0)
   exp.log_file("config.yaml")
   exp.log_model(model)

   print(exp.url)       # direct link to the experiment
   print(exp.metadata)  # metadata dict

   exp.finalize()
