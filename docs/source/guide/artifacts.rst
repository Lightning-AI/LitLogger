#################
Logging Artifacts
#################

LitLogger can store and retrieve files and model artifacts associated with
your experiments.


Log and Retrieve Files
======================

.. code-block:: python

   import litlogger

   # Log files
   litlogger.init(name="my-experiment")
   litlogger.log_file("config.yaml")
   litlogger.log_file("checkpoint.pt", remote_path="checkpoints/best.pt")
   litlogger.finalize()

   # Retrieve files later
   litlogger.init(name="my-experiment")
   litlogger.get_file("/tmp/config.yaml", remote_path="config.yaml")
   litlogger.get_file("/tmp/best.pt", remote_path="checkpoints/best.pt")
   litlogger.finalize()


Logging Multiple Files
======================

``log_files`` uploads a list of files in parallel for better throughput:

.. code-block:: python

   exp = litlogger.init(name="my-experiment")
   exp.log_files(
       paths=["img_0.png", "img_1.png", "img_2.png"],
       remote_paths=["images/0.png", "images/1.png", "images/2.png"],
   )


Log and Retrieve Models
=======================

LitLogger integrates with ``litmodels`` for storing and loading model objects:

.. code-block:: python

   import torch.nn as nn
   import litlogger

   # Log a model
   litlogger.init(name="my-experiment")
   model = nn.Linear(10, 1)
   litlogger.log_model(model)
   litlogger.finalize()

   # Retrieve the model later
   litlogger.init(name="my-experiment")
   loaded_model = litlogger.get_model()
   litlogger.finalize()


Logging Model Files
===================

If you already have saved model files on disk (weights, checkpoints, or full
directories), use ``log_model_artifact``:

.. code-block:: python

   exp = litlogger.init(name="my-experiment")

   # Upload a single checkpoint file
   exp.log_model_artifact("checkpoints/epoch_10.ckpt", version="epoch-10")

   # Upload an entire directory
   exp.log_model_artifact("model_export/", version="export-v2")

Download model artifacts back:

.. code-block:: python

   exp.get_model_artifact("local_checkpoint.ckpt", version="epoch-10")


Automatic Checkpoint Logging
=============================

When using :class:`~litlogger.logger.LightningLogger` with
``log_model=True``, checkpoints saved by Lightning's ``ModelCheckpoint``
callback are automatically uploaded as model artifacts. See
:doc:`lightning` for details.
