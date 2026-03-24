#################
Complete Workflow
#################

This tutorial covers a full experiment lifecycle:

1. Initialize a run
2. Log metadata, metrics, and artifacts
3. Finalize
4. Reconnect later
5. Retrieve metadata and files

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/litlogger/experiment_comparison_charts.png
   :alt: Comparing experiment metrics with charts
   :width: 800px
   :align: center

Train Script
============

.. literalinclude:: ../../../examples/complete_workflow/train.py
   :language: python
   :start-after: limitations under the License.

Fetch Script
============

.. literalinclude:: ../../../examples/complete_workflow/fetch.py
   :language: python
   :start-after: limitations under the License.

What This Covers
================

- Create experiment by name
- Log metadata
- Log metrics
- Upload artifacts
- Reconnect to an existing experiment
- Download files and access metadata

Related Guides
==============

- :doc:`../guide/standalone`
- :doc:`../guide/artifacts`
- :doc:`../guide/workflows`
