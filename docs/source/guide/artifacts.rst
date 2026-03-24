#################
Logging Artifacts
#################

LitLogger can store and retrieve files and model artifacts associated with
your experiments. The modern path is the file-like API with
:class:`~litlogger.media.File` and :class:`~litlogger.media.Model`, while the
older helper methods remain available for compatibility.

Using the Experiment API
========================

Use the returned :class:`~litlogger.experiment.Experiment` and assign
file-like wrappers directly:

.. code-block:: python

   import litlogger
   from litlogger import File, Model

   experiment = litlogger.init(name="my-experiment")
   experiment["config"] = File("config.yaml")
   experiment["checkpoint"] = Model("checkpoint.ckpt")

   experiment["reports"].append(File("report-0.txt"))
   experiment["checkpoints"].append(Model("checkpoint-1.ckpt", version="step-1"))

   experiment.finalize()

Log and Retrieve Files
======================

The legacy helper API is still supported for explicit file upload/download
workflows through :func:`~litlogger.log_file` and :func:`~litlogger.get_file`:

.. code-block:: python

   import litlogger

   litlogger.init(name="my-experiment")
   litlogger.log_file("config.yaml")
   litlogger.log_file("checkpoint.pt", remote_path="checkpoints/best.pt")
   litlogger.finalize()

   litlogger.init(name="my-experiment")
   litlogger.get_file("/tmp/config.yaml", remote_path="config.yaml")
   litlogger.get_file("/tmp/best.pt", remote_path="checkpoints/best.pt")
   litlogger.finalize()

Logging Multiple Files
======================

:meth:`~litlogger.experiment.Experiment.log_files` uploads a list of files in
parallel for better throughput:

.. code-block:: python

   exp = litlogger.init(name="my-experiment")
   exp.log_files(
       paths=["img_0.png", "img_1.png", "img_2.png"],
       remote_paths=["images/0.png", "images/1.png", "images/2.png"],
   )

Log and Retrieve Models
=======================

LitLogger integrates with ``litmodels`` for storing and loading model objects.
The legacy helper API still supports direct model-object logging through
:func:`~litlogger.log_model` and :func:`~litlogger.get_model`:

.. code-block:: python

   import torch.nn as nn
   import litlogger

   litlogger.init(name="my-experiment")
   model = nn.Linear(10, 1)
   litlogger.log_model(model)
   litlogger.finalize()

   litlogger.init(name="my-experiment")
   loaded_model = litlogger.get_model()
   litlogger.finalize()

Logged models appear in the ``Weights`` tab of Lightning AI rather than the
experiment's ``Experiments`` tab, because model storage is handled through the
model registry rather than the artifact list. For background, see the
`Lightning model registry docs <https://lightning.ai/docs/overview/model-registry>`_.

Logging Model Files
===================

If you already have saved model files on disk, use
:meth:`~litlogger.experiment.Experiment.log_model_artifact`:

.. code-block:: python

   exp = litlogger.init(name="my-experiment")
   exp.log_model_artifact("checkpoints/epoch_10.ckpt", version="epoch-10")
   exp.log_model_artifact("model_export/", version="export-v2")

Download model artifacts back with
:meth:`~litlogger.experiment.Experiment.get_model_artifact`:

.. code-block:: python

   exp.get_model_artifact("local_checkpoint.ckpt", version="epoch-10")

Automatic Checkpoint Logging
============================

When using :class:`lightning:lightning.pytorch.loggers.LitLogger` with
``log_model=True``,
checkpoints saved by Lightning's ``ModelCheckpoint`` callback are
automatically uploaded as model artifacts.
