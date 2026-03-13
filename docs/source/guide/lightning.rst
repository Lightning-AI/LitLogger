#######################
Lightning Integration
#######################

:class:`~litlogger.logger.LightningLogger` plugs directly into the PyTorch
Lightning ``Trainer`` and Lightning Fabric, streaming metrics, metadata,
artifacts, media, and models to `lightning.ai <https://lightning.ai>`_.

Using with Trainer
==================

Pass a :class:`~litlogger.logger.LightningLogger` as the ``logger`` argument to
the Trainer. Every ``self.log()`` call inside your LightningModule is
forwarded automatically.

.. code-block:: python

   from lightning import Trainer
   from litlogger import LightningLogger

   logger = LightningLogger(name="cifar10-resnet")
   trainer = Trainer(max_epochs=10, logger=logger)
   trainer.fit(model, datamodule)

After ``trainer.fit()`` starts, the logger prints a URL where you can view
live training curves.

Logging Hyperparameters
=======================

Lightning automatically calls ``log_hyperparams`` when your LightningModule
defines ``self.save_hyperparameters()``.

.. code-block:: python

   class MyModel(L.LightningModule):
       def __init__(self, lr: float = 1e-3, hidden_dim: int = 128):
           super().__init__()
           self.save_hyperparameters()

You can also pass metadata directly:

.. code-block:: python

   logger = LightningLogger(
       name="cifar10-resnet",
       metadata={"optimizer": "AdamW", "scheduler": "cosine"},
   )

Automatic Checkpoint Logging
============================

Set ``log_model=True`` to automatically upload checkpoints to the litmodels
registry whenever Lightning saves a checkpoint.

.. code-block:: python

   from lightning.pytorch.callbacks import ModelCheckpoint

   checkpoint_cb = ModelCheckpoint(save_top_k=2, monitor="val_loss")
   logger = LightningLogger(name="my-model", log_model=True)

   trainer = Trainer(
       max_epochs=20,
       logger=logger,
       callbacks=[checkpoint_cb],
   )
   trainer.fit(model, datamodule)

Using with Fabric
=================

:class:`~litlogger.logger.LightningLogger` also works as a Fabric logger:

.. code-block:: python

   import lightning as L
   from litlogger import LightningLogger

   logger = LightningLogger(name="fabric-run")
   fabric = L.Fabric(loggers=[logger])
   fabric.launch()

   for step in range(1000):
       loss = train_step()
       fabric.log("train_loss", loss, step=step)

Logging Files, Media, and Models
================================

The logger also exposes helper methods for artifact, media, and model logging:

.. code-block:: python

   trainer.logger.log_file("config.yaml")
   trainer.logger.log_media("sample", "output.png", step=epoch)
   trainer.logger.log_model_artifact("checkpoints/best.ckpt", version="best")

Accessing the Experiment URL
============================

.. code-block:: python

   logger = LightningLogger(name="my-run")
   trainer = Trainer(logger=logger)
   trainer.fit(model, datamodule)

   print(logger.url)
