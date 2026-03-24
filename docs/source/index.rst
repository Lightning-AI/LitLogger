##################################
Experiment Tracking with LitLogger
##################################

LitLogger is the experiment tracker built into Lightning AI.
Instrument your code with metrics, upload media and artifacts,
and then visualize these over time inside the Lightning AI platform.
LitLogger can be integrated into standalone scripts,
or works seamlessly with PyTorch Lightning and Lightning Fabric.

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/litlogger/experiment_comparison_charts.png
   :alt: Comparing experiment metrics with charts
   :width: 800px
   :align: center

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

LitLogger supports several key workflows:

- Start or resume an :class:`~litlogger.experiment.Experiment` with :func:`litlogger.init.init`
- Log numeric metrics, :class:`~litlogger.media.Model` or :class:`~litlogger.media.File` artifacts, and media such as :class:`~litlogger.media.Image` and
  :class:`~litlogger.media.Text`
- Integrate seamlessly with PyTorch Lightning or Lightning Fabric using the
  :class:`lightning:lightning.pytorch.loggers.LitLogger` API
- Log inference metrics from :doc:`LitServe <tutorials/litserve>`

Quick Example
=============

.. code-block:: python

    import litlogger
    from litlogger import Text, File, Model, Image

    # Create or resume an experiment with an optional name
    experiment = litlogger.init(name="quickstart")

    # Set experiment-level metadata, media, or artifacts
    experiment["model_name"] = "resnet50"
    # experiment["model"] = Model(<filepath or object>)
    # experiment["config"] = File(<filepath>)

    # Use append to log metrics or media at each step or epoch
    global_step = 0
    for epoch in range(5):
        for step in range(10):
            experiment["epoch"].append(epoch, step=global_step)
            experiment["train/loss"].append(1.0 / (global_step + 1), step=global_step)
            global_step += 1

        # experiment["validation_image"].append(Image(<filepath or object>), step=global_step)

    experiment["summary"] = Text(f"Completed {global_step} steps across {epoch} epochs")

    # Manually mark the experiment as complete
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
