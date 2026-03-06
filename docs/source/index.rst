##########################
Welcome to litlogger
##########################

.. raw:: html

   <p style="font-size: 18px">
      Track machine learning experiments with <a href="https://lightning.ai">Lightning.ai</a>.
      Log metrics, hyperparameters, checkpoints, and artifacts from any training script.
   </p>

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">


Install litlogger
-----------------

.. code-block:: bash

    pip install litlogger


.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">


Get Started
-----------

- :doc:`guide/standalone` -- Track experiments from any Python script
- :doc:`guide/lightning` -- Use with PyTorch Lightning Trainer or Fabric
- :doc:`guide/artifacts` -- Upload models, checkpoints, and files


.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">


API Reference
-------------

- :doc:`api/experiment` -- Core experiment class for logging
- :doc:`api/logger` -- Logger for PyTorch Lightning and Fabric
- :doc:`api/module_api` -- Top-level functions (init, log, finalize)


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

.. toctree::
    :maxdepth: 1
    :caption: API Reference

    api/experiment
    api/logger
    api/module_api

.. raw:: html

    </div>
