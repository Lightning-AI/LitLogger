##################
Lightning Tutorial
##################

Use :class:`lightning:lightning.pytorch.loggers.LitLogger` when you want
Lightning or Fabric to drive experiment logging for you.
:class:`~litlogger.logger.LightningLogger` is deprecated and remains only as a
compatibility alias.

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/litlogger/experiment_comparison_media.png
   :alt: Comparing experiment media across runs
   :width: 800px
   :align: center

Trainer Integration
===================

.. code-block:: python

   from lightning import Trainer
   from lightning.pytorch.loggers import LitLogger

   logger = LitLogger(
       name="mnist-autoencoder",
       metadata={"dataset": "mnist"},
   )

   trainer = Trainer(max_epochs=3, logger=logger)
   trainer.fit(model, datamodule)

Metrics
=======

Every ``self.log(...)`` call in your module flows through the logger into
LitLogger.

.. code-block:: python

   def training_step(self, batch, batch_idx):
       loss = ...
       self.log("train_loss", loss)
       return loss

Hyperparameters
===============

Lightning hyperparameters are captured through ``save_hyperparameters()`` or
via ``metadata=...`` on the logger itself.

Checkpoint Logging
==================

Enable automatic checkpoint uploads with ``log_model=True``.

.. code-block:: python

   logger = LitLogger(name="my-model", log_model=True)

   trainer = Trainer(
       logger=logger,
       callbacks=[checkpoint_callback],
   )

Files and Models
================

The logger also exposes
:meth:`~litlogger.logger.LightningLogger.log_file` and
:meth:`~litlogger.logger.LightningLogger.log_model_artifact` for file and model
logging:

.. code-block:: python

   trainer.logger.log_file("config.yaml")
   trainer.logger.log_model_artifact("checkpoints/best.ckpt", version="best")

For media uploads, use the standalone
:class:`~litlogger.experiment.Experiment` API directly. The upstream
:class:`lightning:lightning.pytorch.loggers.LitLogger` does not provide a
``log_media`` helper.

Runnable Example
================

See the complete runnable example in
``examples/lightning_autoencoder.py`` and the deeper guide at
:doc:`../guide/lightning`.
