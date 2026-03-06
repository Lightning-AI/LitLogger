#######################
Lightning Integration
#######################

:class:`~litlogger.logger.LightningLogger` plugs directly into the PyTorch
Lightning ``Trainer`` and Lightning Fabric, streaming metrics and artifacts to
`lightning.ai <https://lightning.ai>`_.


Using with Trainer
==================

Pass a ``LightningLogger`` as the ``logger`` argument to the Trainer.
Every ``self.log()`` call inside your LightningModule is automatically
forwarded to Lightning.ai:

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
defines ``self.save_hyperparameters()``. The hyperparameters appear as tags in
the experiment UI:

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
=============================

Set ``log_model=True`` to automatically upload checkpoints to the litmodels
registry whenever Lightning saves a checkpoint:

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

``LightningLogger`` also works as a Fabric logger:

.. code-block:: python

   import lightning as L
   from litlogger import LightningLogger

   logger = LightningLogger(name="fabric-run")
   fabric = L.Fabric(loggers=[logger])
   fabric.launch()

   for step in range(1000):
       loss = train_step()
       fabric.log("train_loss", loss, step=step)


Logging Files and Media
========================

The logger exposes ``log_file`` and ``log_media`` for uploading artifacts
during training:

.. code-block:: python

   # Inside a training callback or LightningModule
   trainer.logger.log_file("config.yaml")
   trainer.logger.log_media("sample", "output.png", step=epoch)


Accessing the Experiment URL
============================

.. code-block:: python

   logger = LightningLogger(name="my-run")
   trainer = Trainer(logger=logger)
   trainer.fit(model, datamodule)

   print(logger.url)  # direct link to the experiment


Third-Party Loggers
===================

Use any third-party logger you want with Lightning. However, LitLogger is free
and native to the Lightning platform.
