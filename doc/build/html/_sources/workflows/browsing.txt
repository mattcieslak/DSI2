Browsing the Streamline Database
======================================

The goal of this workflow is to load some data into the DSI2 Browser 
and be able to interactively query and cluster streamlines.

Assuming that you have imported your data through ``dsi2_import``, you can launch the data browser from your shell:

.. code-block:: bash

  $ dsi2_browse 


Red Text: Data Source and Aggregator
"""""""""""""""""""""""""""""""""""""

Launching the Browser, you can select which individual datasets will be queried when you launch the 
Sphere Browser. 

Select your Data Source in the top left of the screen
------------------------------------------------------
Will you be loading data from your local hard drive or querying a remote database? You can choose either of these options from the *Data Source* listbox.

Change **Aggregation Algorithm** to *Region Labels*
----------------------------------------------------
Select which streamline aggregation method you will use to label the streamlines returned by your searches in the Voxel Browser. In this example we will choose a Region Label Aggregator, which groups streamlines based on which regions they connect. Other aggregation options include k-means clustering or DiPy's QuickBundles algorithm.
	
.. figure:: ../_static/dsi2browse.png
   :scale: 70%
   :alt: browser builder
   :align: center
   

Yellow Box: Specifying data source properties
""""""""""""""""""""""""""""""""""""""""""""""
Choosing a data set
	- Select your ``-.json`` file
	- Enter either a Scan id, Subject id, Study, or Scan group to search for an individual dataset
		- These fields can be used to search for specific properties of datasets.
	- Once your specifications are filled out, click **Search for Datasets** highlighted in green above
		- This will populate the list of results

Blue Box: Matching datasets
""""""""""""""""""""""""""""
Datasets matching your search criteria are listed in the blue box above. 

The first few columns present information about the individual who was scanned. The rest 
offer options on how the streamlines from each dataset will be displayed in the
Sphere Browser. 

If the *dynamic color clusters* column is checked, then the colormap from the *color map* 
column will be applied to that dataset's streamlines. This is useful for interactive clustering 
in the Sphere Browser since you can see which group each streamline was assigned to based on its
color. 

On the other hand, if you are interested in plotting streamlines that are colored only 
according to which individual they came from, then *dynamic color clusters* should be unchecked 
and a *static color* can be assigned. 

If there are datasets in the list that you would not like to be included
in your Sphere Browser, click anywhere in the row and click the *delete row*
button. 

Once you are happy with the list of datasets, click the **Launch sphere browser* button in the purple box.

The Sphere Browser: Querying and visualizing streamlines 
========================================================

The Sphere Browser lets a user choose a set of coordinates, queries the data source
at those coordinates, then aggregates the streamlines before displaying them in the
3D viewer. 


.. figure:: ../_static/vb_annot.png
   :scale: 20%
   :alt: browser builder
   :align: center


Selecting Coordinates: Using a sphere
""""""""""""""""""""""""""""""""""""""

A sphere is a handy way to select a set of coordinates. 

Sliders in the green box let you move the sphere in x, y and z coordinates 
and increase its radius. Moving these will directly affect the visualization 
of your search coordinates, which appear as red cubes (circled in the 3D viewer
in orange).

Selecting Coordinates: From an ROI
""""""""""""""""""""""""""""""""""""""

If there is a set of voxels defined in a NIfTI file, you can load them into the 
browser by selecting :menuselection:`Data --> Search from NIfTI`. You will see
the following dialog box:

.. figure:: ../_static/roi_search.png
   :scale: 80%
   :alt: browser builder
   :align: center

Select a .nii file in the :guilabel:`Filepath` box. If there are multiple regions
in the file you'd like to include (for example a region labeled 1 and a separate
region labeled 2) you can right click the arrow to include a second region. If you'd
like to expand the regions in space, you can dilate them by n voxels. Clicking :guilabel:`OK`
will clear any previous search coordinates and render your new coordinates in the 3D viewer.

Visualizing Streamlines
"""""""""""""""""""""""""
Widgets in the magenta box provide control over how streamlines are plotted. If 
:guilabel:`Auto Aggregate` is selected, the cluster assignments will be updated 
any time one of the aggregation-related widgets is interacted with. In the case 
of this image the aggregator is a k-means aggregator, which has two parameters.
``k``, or the number of clusters to define, and ``min_tracks`` as the minimum 
number of streamlines that must be assigned to a cluster before the cluster can
be considered legitimate. As these two sliders are dragged around, the new cluster
assignments are visible in the blue box. The color of each row in this list corresponds
to the color of the streamlines in the 3D viewer. 

The :guilabel:`Render tracks` checkbox can be used to prevent the rendering of stremalines
in the 3D viewer. If :guilabel:`Auto aggregate` is enabled while :guilabel:`Render tracks`
is disabled, cluster assignments will still be visible in the cluster list and will be updated
as you interact with the clustering widgets. 

The :guilabel:`Render clusters` checkbox used to render glyphs that summarize each cluster. This 
functionality no longer works.

Changing the appearance of streamlines
"""""""""""""""""""""""""""""""""""""""



