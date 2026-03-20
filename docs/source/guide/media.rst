#############
Logging Media
#############

LitLogger supports uploading images and text files that are displayed alongside
your metrics in the experiment view. For new code, prefer the dict-style
wrappers :class:`~litlogger.media.Image` and :class:`~litlogger.media.Text`.

New Dict-Style API
==================

.. code-block:: python

   import litlogger
   from litlogger import Image, Text

   experiment = litlogger.init(name="media-demo")

   experiment["preview"] = Image("generated.png")
   experiment["notes"] = Text("epoch 0 summary")

   experiment["samples"].append(Image("sample-0.png"), step=0)
   experiment["captions"].append(Text("reconstruction"), step=0)

Legacy Helper API
=================

The legacy helper method :meth:`~litlogger.experiment.Experiment.log_media`
remains available for existing code, but new standalone code should prefer the
dict-style wrappers.

Logging Images
==============

.. code-block:: python

   import litlogger
   from litlogger.types import MediaType

   exp = litlogger.init(name="my-experiment")

   exp.log_media("sample_output", "generated.png", step=0)

   exp.log_media(
       name="reconstruction",
       path="recon.jpg",
       kind=MediaType.IMAGE,
       step=10,
       caption="Epoch 10 reconstruction",
   )

Logging Text
============

.. code-block:: python

   exp.log_media("predictions", "predictions.txt", kind=MediaType.TEXT, step=5)

Supported Types
===============

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - MediaType
     - File extensions
     - Description
   * - ``IMAGE``
     - ``.png``, ``.jpg``, ``.jpeg``, ``.gif``, ``.bmp``, ``.webp``
     - Displayed as images in the experiment view
   * - ``TEXT``
     - ``.txt``, ``.csv``, ``.json``, ``.log``
     - Displayed as text in the experiment view

When ``kind`` is not provided, LitLogger guesses the type from the file's MIME
type. If the type cannot be determined, a ``ValueError`` is raised.

Deprecated Compatibility Logger
===============================

:class:`~litlogger.logger.LightningLogger` is deprecated. For Lightning or
Fabric integrations, prefer
:class:`lightning:lightning.pytorch.loggers.LitLogger` for metrics, files, and
model artifacts. For media uploads, use the standalone
:class:`~litlogger.experiment.Experiment` API directly; the upstream
:class:`lightning:lightning.pytorch.loggers.LitLogger` does not expose a
``log_media`` helper.
