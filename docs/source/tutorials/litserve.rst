#################
LitServe Tutorial
#################

LitLogger can also be used in inference-serving workflows through LitServe.

Endpoint Logging
================

The LitServe example initializes a logger in ``setup()`` and records metrics
for each request in ``predict()``.

.. literalinclude:: ../../../examples/litserve_inference.py
   :language: python
   :start-after: limitations under the License.

What This Workflow Covers
=========================

- request-time metric logging
- inference latency tracking
- token-count or request-shape metrics
- deployment-oriented usage outside training loops

Related Docs
============

- :doc:`../guide/workflows`
- :doc:`../guide/examples`
