####################
Files, Media, Models
####################

This tutorial covers the new file-like API surface:

- :class:`~litlogger.media.File`
- :class:`~litlogger.media.Image`
- :class:`~litlogger.media.Text`
- :class:`~litlogger.media.Model`

Static Files
============

.. code-block:: python

   import litlogger
   from litlogger import File

   experiment = litlogger.init(name="artifacts-demo")
   experiment["config"] = File("config.yaml")

File Series
===========

Append file-like values to create a time series:

.. code-block:: python

   for step in range(3):
       experiment["reports"].append(File(f"report-{step}.txt"))

Text
====

Use :class:`~litlogger.media.Text` for text content you want rendered to a
temporary file and uploaded automatically.

.. code-block:: python

   from litlogger import Text

   experiment["notes"] = Text("experiment summary")
   experiment["captions"].append(Text("checkpoint at step 0"), step=0)

Images
======

:class:`~litlogger.media.Image` accepts a path or common in-memory image data.

.. code-block:: python

   from litlogger import Image

   experiment["preview"] = Image("preview.png")
   experiment["samples"].append(Image("sample-0.png"), step=0)

Models
======

Use :class:`~litlogger.media.Model` for model objects or saved model files.

.. code-block:: python

   from litlogger import Model

   experiment["checkpoint"] = Model("checkpoint.ckpt", version="latest")
   experiment["checkpoints"].append(Model("checkpoint-1.ckpt", version="step-1"))

Unlike files, images, and text entries, logged models are registered in the
``Weights`` tab of Lightning AI rather than the experiment's ``Experiments``
tab. For background, see the `Lightning model registry docs <https://lightning.ai/docs/overview/model-registry>`_.

Current Recovery Note
=====================

Within the current process, uploaded ``Model`` values are rebound with download
or load behavior. Resumed experiment recovery for models still needs dedicated
backend support.

Finalize
========

.. code-block:: python

   experiment.finalize()

Runnable Example
================

See :doc:`../guide/examples` for the full
``examples/file_media_model_usage.py`` walkthrough.
