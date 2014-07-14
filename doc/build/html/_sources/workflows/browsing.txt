Browsing the Streamline Database
======================================

The goal of this workflow is to load some data into the VoxelBrowser GUI 
and be able to interactively query and cluster streamlines. This process
requires two steps: 

  1. The Browser Builder GUI is uset to select a data source and streamline aggregation method
  2. The Voxel Browser is launched and used to query spatial coordinates

The Browser Builder: Creating a Data Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launching the BrowserBuilder, you can select which individual datasets will be
queried when you open a VoxelBrowser. 

.. figure:: ../_static/annotated_browser_builder.png
   :scale: 20%
   :alt: browser builder
   :align: center

Red box: Data source and aggregator
"""""""""""""""""""""""""""""""""""""
Will you be loading data from your local hard drive or querying a remote
database? You can choose either of these options from the *Data Source* 
listbox. Next, you select which streamline aggregation method you will
use to label the streamlines returned by your searches in the Voxel Browser.
In this example we will choose a Region Label Aggregator, which groups
streamlines based on which regions they connect. Other aggregation options
include k-means clustering or DiPy's QuickBundles algorithm.

Blue Box: Specifying data source properties
"""""""""""""""""""""""""""""""""""""""""""
These fields can be used to search for specific properties of datasets. 
If a field without an arrow in front of it is left blank, it means that 
anything is OK for that field. If there are arrows in front of boxes, 
this means that a value matching *any* of the arrowed boxes will be a 
match.

Currently most of these search fields do nothing. For most uses, ``scan_id`` 
``study`` and ``scan_group`` are sufficient to find the data you need.
Once your specifications are filled out here, click **Search For Datasets**
in the magenta box. This will populate the list of results.

Green Box: Matching datasets
""""""""""""""""""""""""""""
Datasets matching your search criteria are listed here. The first few columns 
present information about the individual who was scanned. The rest offer
options on how the streamlines from each dataset will be displayed in the
Voxel Browser. If the *dynamic color clusters* column is checked, then 
the colormap from the *color map* column will be applied to that dataset's 
streamlines. This is useful for interactive clustering in the Voxel Browser 
since you can see which group each streamline was assigned to based on its
color. On the other hand, if you are interested in plotting streamlines 
that are colored only according to which individual they came from, then
*dynamic color clusters* should be unchecked and a *static color* can be 
assigned. 

If there are datasets in the list that you would not like to be included
in your Voxel Browser, click anywhere in the row and click the *delete row*
button to the right of the superimposed green arrow. Once you are happy with
the list of datasets, click the *Launch sphere browser* button in the majenta
box.


The Voxel Browser: Querying and visualizing streamlines 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Voxel Browser lets a user choose a set of coordinates, queries the data source
at those coordinates, then aggregates the streamlines before displaying them in the
3D viewer. 


.. figure:: ../_static/vb_annot.png
   :scale: 20%
   :alt: browser builder
   :align: center


Selecting Coordinates: Using a sphere
""""""""""""""""""""""""""""""""""""""

A sphere is a handy way to select a set of coordinates. Sliders in the green
box let you move the sphere in x, y and z coordinates and increase its
radius. Moving these will directly affect the visualization of your search 
coordinates, which appear as red cubes (circled in the 3D viewer in orange).

Selecting Coordinates: From a ROI
""""""""""""""""""""""""""""""""""""""

If there is a set of voxels defined in a NIfTI file, you can load them into the 
browser by selecting :menuselection:`Data --> Search from NIfTI`. You will see
the following dialog

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



