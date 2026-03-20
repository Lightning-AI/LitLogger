#################
Complete Workflow
#################

This tutorial covers a full experiment lifecycle:

1. initialize a run
2. log metadata, metrics, and artifacts
3. finalize
4. reconnect later
5. retrieve metadata and files

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

- experiment creation by name
- metadata logging
- metric logging
- artifact upload
- reconnection to an existing experiment
- file download and metadata access

Related Guides
==============

- :doc:`../guide/standalone`
- :doc:`../guide/artifacts`
- :doc:`../guide/workflows`
