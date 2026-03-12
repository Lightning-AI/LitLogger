########
Examples
########


Standalone
==========

A standalone training script using the module-level API to log metrics,
metadata, and files.

.. literalinclude:: ../../../examples/standalone_usage.py
   :language: python
   :start-after: limitations under the License.


Files, Media, and Models
========================

A dict-style example showing static uploads, file and text series, image logging,
and model artifacts with ``File``, ``Text``, ``Image``, and ``Model``.

.. literalinclude:: ../../../examples/file_media_model_usage.py
   :language: python
   :start-after: limitations under the License.


PyTorch Lightning
=================

An MNIST autoencoder trained with PyTorch Lightning and ``LightningLogger``.

.. literalinclude:: ../../../examples/lightning_autoencoder.py
   :language: python
   :start-after: limitations under the License.


LitServe
========

Track inference metrics from a LitServe endpoint. Requires ``pip install litgpt``.

.. literalinclude:: ../../../examples/litserve_inference.py
   :language: python
   :start-after: limitations under the License.

Run with ``python litserve_inference.py``, then test with:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What do llamas eat"}'


Complete Workflow
=================

A complete workflow demonstrating logging and retrieval of metrics, metadata,
and files.

**train.py** -- Log experiment:

.. literalinclude:: ../../../examples/complete_workflow/train.py
   :language: python
   :start-after: limitations under the License.

**fetch.py** -- Get experiment data:

.. literalinclude:: ../../../examples/complete_workflow/fetch.py
   :language: python
   :start-after: limitations under the License.
