#############
Logging Media
#############

litlogger supports uploading images and text files that are displayed alongside
your metrics in the `lightning.ai <https://lightning.ai>`_ experiment view.


Logging Images
==============

.. code-block:: python

   import litlogger
   from litlogger.types import MediaType

   exp = litlogger.init(name="my-experiment")

   # Auto-detected from file extension
   exp.log_media("sample_output", "generated.png", step=0)

   # Explicit type
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

When ``kind`` is not provided, litlogger guesses the type from the file's MIME
type. If the type cannot be determined, a ``ValueError`` is raised.


Parameters
==========

:meth:`litlogger.log_media() <litlogger.experiment.Experiment.log_media>` accepts
the following parameters:

- **name** -- Name of the media entry.
- **path** -- Local path to the file.
- **kind** -- ``MediaType.IMAGE`` or ``MediaType.TEXT`` (optional, auto-detected).
- **step** -- Training step number (optional).
- **epoch** -- Training epoch number (optional).
- **caption** -- Descriptive caption (optional).


Using with LightningLogger
===========================

The :class:`~litlogger.logger.LightningLogger` exposes the same
:meth:`litlogger.log_media() <litlogger.experiment.Experiment.log_media>` method, which you can call from callbacks or your LightningModule:

.. code-block:: python

   from litlogger import LightningLogger

   logger = LightningLogger(name="vision-run")

   # Inside a callback
   logger.log_media(
       "val_sample",
       "val_output.png",
       step=trainer.global_step,
       caption="Validation sample",
   )
