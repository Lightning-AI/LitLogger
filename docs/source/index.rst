######################
Experiment Management
######################

LitLogger is the experiment tracker built into Lightning AI. It supports
standalone scripts, the new dict-style
:class:`~litlogger.experiment.Experiment` API, Lightning/Fabric integration,
files, media, models, and retrieval workflows.

.. image:: https://storage.googleapis.com/lightning-avatars/litpages/01jrbsmwcrgj50pbybv827wc5d/9ee96e2e-58bd-4031-991f-1cff59009d19.png
   :alt: LitLogger experiment dashboard

These docs are organized as a narrative:

- tutorials to get from zero to a working flow
- guides for specific workflows and API patterns
- runnable examples from the repository
- an API reference for the public surface

Start Here
==========

Use this path if you are new to LitLogger:

1. :doc:`install`
2. :doc:`tutorials/quickstart`
3. :doc:`tutorials/file_media_model`
4. :doc:`tutorials/lightning`
5. :doc:`tutorials/complete_workflow`

Supported Workflows
===================

LitLogger currently supports these main workflows:

- standalone logging with the dict-style ``Experiment`` API
- legacy module-level logging helpers for existing scripts
- Lightning and Fabric integration through upstream ``LitLogger`` classes
- file, image, text, and model logging through dedicated wrappers
- experiment resume and later retrieval by name
- inference logging from a LitServe endpoint

Quick Example
=============

.. code-block:: python

   import litlogger
   from litlogger import File, Text

   experiment = litlogger.init(name="quickstart")

   experiment["model"] = "resnet50"
   experiment["notes"] = Text("first run")

   for step in range(10):
       experiment["train/loss"].append(1.0 / (step + 1), step=step)

   experiment["config"] = File("config.yaml")
   experiment.finalize()

Documentation Map
=================

:ref:`Tutorials <tutorials-section>` provide end-to-end, task-oriented walkthroughs.

:ref:`Guides <guides-section>` explain workflows, migration paths, and logging patterns in more depth.

:ref:`Examples <examples-section>` include every runnable example currently shipped in the repository.

The :ref:`API reference <api-reference-section>` documents the public classes, functions, and enums.

.. _start-section:

Start
=====

.. toctree::
    :maxdepth: 1

    install

.. _tutorials-section:

Tutorials
=========

.. toctree::
    :maxdepth: 1

    tutorials/quickstart
    tutorials/file_media_model
    tutorials/lightning
    tutorials/complete_workflow
    tutorials/litserve

.. _guides-section:

Guides
======

.. toctree::
    :maxdepth: 1

    guide/standalone
    guide/lightning
    guide/artifacts
    guide/media
    guide/workflows
    guide/examples

.. _examples-section:

Examples
========

See :doc:`guide/examples` for the full list of runnable repository examples.

.. _api-reference-section:

API Reference
=============

.. toctree::
    :maxdepth: 1

    api
