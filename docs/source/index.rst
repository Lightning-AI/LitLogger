######################
Experiment Management
######################

Lightning AI comes with a free, bundled experiment manager called LitLogger.
Experiment managers let you track, compare and share ML model training runs.
It keeps them organized and reproducible, providing clear visibility into model
performance.

.. image:: https://storage.googleapis.com/lightning-avatars/litpages/01jrbsmwcrgj50pbybv827wc5d/9ee96e2e-58bd-4031-991f-1cff59009d19.png
   :alt: LitLogger experiment dashboard

----

Why LitLogger
=============

Without reliable experiment management, teams lose history, cannot compare
experiments, and struggle to reproduce or audit their work. LitLogger solves
this with a built-in, persistent experiment manager that automatically
organizes every run -- centralizing metrics, metadata, and artifacts so
experiments can be compared, restored, and shared easily. The result is
clarity of research results, faster iteration, and reproducible model
development without having to pay for a standalone platform.

Key features:

- Generous free tier
- PyTorch optimized
- Share with anyone
- Granular permissions (RBAC)
- Granular project management

----

Quick Start
===========

Install
-------

.. code-block:: bash

   pip install litlogger

Hello World
-----------

Enable logging by adding this to ANY Python code:

.. code-block:: python

   import litlogger

   litlogger.init()

   for i in range(10):
       litlogger.log({"my_metric": i})

   litlogger.finalize()

----

APIs
====

LitLogger provides two APIs: a **standalone API** for any Python code, and an
**Experiment** class for more control.

Standalone API
--------------

The standalone API uses ``litlogger.init()`` and module-level functions. This
is the recommended approach for most use cases.

.. code-block:: python

   import litlogger

   litlogger.init(
       name="my-experiment",
       metadata={"model": "ResNet50", "lr": "0.001"},
   )

   for epoch in range(10):
       litlogger.log_metrics({"loss": 1.0 / (epoch + 1)}, step=epoch)

   litlogger.finalize()

See :doc:`guide/standalone` for the full guide.

Experiment
----------

``litlogger.init()`` returns an :class:`~litlogger.experiment.Experiment`
instance. You can also create one directly for full control over the
experiment lifecycle.

.. code-block:: python

   from litlogger import Experiment

   exp = Experiment(
       name="my-experiment",
       metadata={"model": "ResNet50", "lr": "0.001"},
   )

   for epoch in range(10):
       exp.log_metrics({"loss": 1.0 / (epoch + 1)}, step=epoch)

   exp.log_file("config.yaml")
   exp.log_model(model)
   exp.finalize()

See the :doc:`api/experiment` reference for all available methods.

----

PyTorch Lightning Integration
=============================

The ``LightningLogger`` class integrates directly with the PyTorch Lightning
Trainer, so every ``self.log()`` call is automatically forwarded to
Lightning.ai.

.. code-block:: python

   from lightning import Trainer
   from litlogger import LightningLogger

   logger = LightningLogger(
       name="my-experiment",
       metadata={"model": "ResNet50"},
   )

   trainer = Trainer(max_epochs=10, logger=logger)
   trainer.fit(model, datamodule)

See :doc:`guide/lightning` for the full guide.

----

View and Share Runs
===================

All experiments are collected in the "Experiments" tab in your Teamspace.

.. image:: https://storage.googleapis.com/lightning-avatars/litpages/01j07an7zgc2ewmnnxj8gngg72/889a6cdd-c27f-499e-8d4b-6856e575e97a.png
   :alt: Experiments tab in your Teamspace

Open an experiment detail to share with everyone or specific users.

.. image:: https://storage.googleapis.com/lightning-avatars/litpages/01j07an7zgc2ewmnnxj8gngg72/19d278b1-32a4-43ac-9a03-200aa0687e69.png
   :alt: Sharing option for experiments

----

Third-Party Loggers
===================

Use any third-party logger you want with Lightning. However, LitLogger is free
and native to the Lightning platform.

----

.. raw:: html

    <div style="display:none">

.. toctree::
    :maxdepth: 1
    :name: start
    :caption: Home

    self
    Install <install>

.. toctree::
    :maxdepth: 1
    :caption: Guides

    guide/standalone
    guide/lightning
    guide/artifacts
    guide/media
    guide/examples

.. toctree::
    :maxdepth: 1
    :caption: API Reference

    api/experiment
    api/logger
    api/module_api

.. raw:: html

    </div>
