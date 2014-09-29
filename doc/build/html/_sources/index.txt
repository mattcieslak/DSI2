.. dsi2 documentation master file, created by
   sphinx-quickstart on Mon Jul 29 00:58:49 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dsi2 documentation
============================

This software is designed to enable the analysis of
diffusion-weighted MRI-based tractography datasets.
**DSI2** stands for **D** iscrete **S** patial **I** ndexing
of **D** iffusion **S** pectrum **I** mages. Image
reconstruction, tractography and parcellation must be done
outside of this software, but indexing, searching and
analysis are all implemented here. Specifically, this package
provides the tools necessary to perform Local Termination Pattern
Analysis (LTPA, [1]_ )

The idea is that
 1. A set of coordinates in the brain can be used to query a database.
 2. The query returns a set of streamlines that that pass through the set
    of coordinates.
 3. The streamlines are divided into groups based on

    a) Their termination regions
    b) Clustering based on their shape
 4. The groups of streamlines are compared across individuals

Take, for example, the
corticospinal tract. If we select a sphere of
coordinates in the internal capsule, we should see tracks
following the pathway of the corticospinal tract.

.. figure:: _static/sphere_fibs.png
   :scale: 80 %
   :alt: some *tracks*
   :align: center

We can then group these streamlines based on the regions in which
they terminate. Using the Lausanne2008 scale 33 atlas, we get the
following distribution of streamline counts:

.. figure:: _static/patterns.png
   :align: center



Getting Started
===============

:doc:`installation`
--------------------

:doc:`preprocessing/preproc`
----------------------------
  Use DSI Studio and ``connectome_mapper`` to prepare data

:doc:`analysis/overview`
------------------------
  An overview of possible analyses and pitfalls to beware of.

.. toctree::

   installation
   preprocessing/preproc
   analysis/overview
   analysis/ltpa
   analysis/aggregation
   workflows/browsing

References
----------

.. [1] Cieslak, M., & Grafton, S.T. Local termination pattern analysis:
    a tool for comparing white matter morphology. Brain Imaging Behav, DOI 10.1007/s11682-013-9254-z (2013).


Indices and tables
==================

* :ref:`modindex`
* :ref:`search`

