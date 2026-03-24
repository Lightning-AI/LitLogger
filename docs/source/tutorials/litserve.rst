#################
LitServe Tutorial
#################

LitLogger can also be used in inference-serving workflows through LitServe.

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/litlogger/experiment_comparison_side_by_side.png
   :alt: Comparing inference experiments side by side
   :width: 800px
   :align: center

Endpoint Logging
================

The LitServe example initializes a logger in ``setup()`` and records metrics
for each request in ``predict()``.

.. literalinclude:: ../../../examples/litserve_inference.py
   :language: python
   :start-after: limitations under the License.

What This Workflow Covers
=========================

- Request-time metric logging
- Inference latency tracking
- Token-count or request-shape metrics
- Deployment-oriented usage outside training loops

Related Docs
============

- :doc:`../guide/workflows`
- :doc:`../guide/examples`
