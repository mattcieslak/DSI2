Analysis of local termination patterns
======================================

.. toctree::
   :hidden:

.. warning::
  The examples presented below don't currently work. It might be nice to
  have a full json file that works with a bare-bones data tarball.

Here we describe how to write a script that runs LTPA on a group
of tractography datasets.

Setting up your environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``dsi2`` requires some environment variables to be set so it can find
resources in your file system. On a \*nix system these are set using the
``export`` command.

 * ``DSI2_DATA`` is the path to the ``example_data`` directory that 
    is provided on this website (**NOTE: upload the example data**).
 * ``LOCAL_TRACKDB`` is the path to your directory containing your 
    locally stored .pkl files. If this variable is not set, dsi2 will
    check your home directory for a ``local_trackdb`` directory.


Organizing your local datasource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data for LTPA has two parts: the .pkl file containing the streamline mapping for
a single subject and the corresponding metadata. The streamline mapping is a 
pickled :py:class:`~dsi2.streamlines.track_dataset.TrackDataset` object. It 
contains the mapping from voxel :math:`(i,j,k)` index to a set of id's referencing
the streamlines that pass through the voxel.

Metadata provides information
about the person whose white matter is stored in the pkl - their age, gender, 
brain injuries, etc. ``dsi2`` stores paths and metadata for local data in a .json file. 

.. note::
  When beginning a project, the first step should be to create this json file.
  Run DSI Studio and connectome mapper to process the raw data and make sure
  the files are placed in the locations specified in the json file. Streamline
  labels can be created by ``dsi2`` based on the paths in the json file.

Here is an example ``local_data.json``

.. code-block:: javascript

   [
    {
        "scan_id": "subj1",
        "length_min": 10,
        "pkl_trk_path": "subj1/subj1.Louvain.nsm.MNI.trk",
        "cutoff_angle": 55,
        "scan_group": "control",
        "qa_threshold": 1.0,
        "smoothing": 0.0,
        "length_max": 400,
        "gfa_threshold": 0,
        "reconstruction": "qsdr",
        "trk_space": "qsdr",
        "study": "example study",
        "attributes": {
            "handedness": "R",
            "weight": 190
        },
        "pkl_path": "subj1/subj1.Louvain.nsm.MNI.pkl",
        "trk_file": "subj1/QSDR.100000.nsm.trk",
        "software": "DSI Studio"
        "track_scalars": [],
        "track_labels": [
            {
                "name": "Lausanne",
                "parameters": {
                    "scale": 33,
                    "dilation": 2
                },
                "notes": "",
                "graphml_path": "lausanne2008/resolution83/resolution83.graphml",
                "numpy_path": "subj1/subj1.scale33.thick2.npy",
                "volume_path": "atlases/subj1/scale33.thick2.nii.gz"
            },
            {
                "name": "Lausanne",
                "parameters": {
                    "scale": 60,
                    "dilation": 2
                },
                "notes": "",
                "graphml_path": "lausanne2008/resolution150/resolution150.graphml",
                "numpy_path": "subj1/subj1.thick2.npy",
                "volume_path": "atlases/subj1/scale60.thick2.nii.gz"
            }
           ]
     }, 
     ...
  ]

Basic information
"""""""""""""""""
The following fields are all optional *except for scan_id and scan_group*

"scan_id"
  A unique identifier for this dataset
 
"length_min"
  Minimum streamline length in millimeters

"length_max"
  Maximum streamline length in millimeters

"pkl_path"
  The pickled streamline mapping object

"pkl_trk_path"
  Path to the trackvis .trk file used to create the hashed pkl file 

"cutoff_angle"
  Maximum turning angle used in tracking

"scan_group"
  To which experimental group does this data belong (eg "control" or "tbi")

"qa_threshold"
  QA threshold used during tracking

"smoothing"
  DSI Studio smoothing parameter 

"reconstruction"
  How was the diffusion data reconstructed?

"trk_space"
  Which coordinate system is the trk file in? 

"study"
  An identifier for the study this person was a part of

"attributes"
  A dictionary of attributes for this subject. Key/value pairs could include 
  handedness, weight, gender, scores on questionnaires, etc

"trk_file": 
  Path to the original trackvis formatted file

"software": 
  Software used to reconstruct DWI and/or perform tractography

Streamline Labels
"""""""""""""""""
Since the :py:class:`~dsi2.streamlines.track_dataset.TrackDataset` object maps from
voxel to streamline ID, it is easy to store a label value for each streamline ID. 
Labels can be defined based on which regions the streamline connects. 

It is useful to have multiple labels per streamline. For instance, you might have 
multiple resolutions of an atlas and want to have access to the streamlines' labels
under each resolution. That is why this field is a list of objects describing a 
single labeling.

"name"
   A name identifying which family of atlases these labels come from

"notes"
   A place to store notes about this set of labels

"graphml_path"
   If the regions have labels, they should be stored in a graphml file
   like those included in the connectome mapping toolkit.

"numpy_path"
   The labels are stored in a numpy file on disk.

"volume_path"
   path to the nifti file that contains the regions for this labeling
   scheme. It should be in the same space (ie qsdr/MNI/native) as the
   trk file "trk_space".

"parameters"
    key/value pairs that describe this version of the atlas. For example
    the Lausanne family of atlases has a range of "scale" parameters.


Creating streamline labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assuming DSI Studio and Connectome Mapper ran successfully, you can pass your
json file to a :py:class:`~dsi2.ui.local_data_importer.LocalDataImporter`::
  
  from dsi2.ui.local_data_importer import LocalDataImporter

  ldi = LocalDataImporter(json_file="path/to/file.json")
  ldi.validate_localdb()

This will "fill-in" the missing files from your json. If only trk files and 
nii files exist, it will create the pkl files and produce numpy files for
each atlas in the "track_labels" list. Alternatively, you could call the 
``edit_traits()`` method on ``ldi`` and check your configuration visually.

Searching your local datasource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the necessary streamline labels and pickle files are in their specified 
locations, we can begin to search the database. ::

  from dsi2.database.local_data import get_local_data
  from dsi2.database.track_datasource import TrackDataSource

  # Load the json file to a list of Scan objects
  scans = get_local_data("path/to/file.json")

``scans`` will be a list of objects that can be used to select only the data 
you want for this analysis. It does **not** load the pkl into memory until 
you call its :py:meth:`~dsi2.database.traited_query.Scan.get_track_dataset()` function.
Let's select only scans from the "example" study and load them into memory.::

  example_scans = [ scan for scan in scans if scan.study_id == "example study ]
  # Use them to build a queryable data source
  trk_src = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])

``trk_src`` provides an interface to a searchable MNI space. Raw streamlines 
aren't particularly useful, so we will create an object that uses this interface
to search through and aggregate streamlines.

Aggregating streamlines based on termination regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To apply perform the analysis from [1]_ we need to create an *aggregator*. An 
aggregator 
  
  * Subclasses :py:class:`~dsi2.aggregation.cluster_ui.ClusterEditor`
  * Overrides the ``aggregate`` method 
  * operates on a :py:class:`~dsi2.database.track_datasource.TrackDataSource` 

For a region-based LTPA, we can create a :py:class:`~dsi2.aggregation.region_labeled_clusters.RegionLabelAggregator`
that will provide some methods for easily analyzing these streamlines resulting from the
search. Suppose we'd like to search a set of coordinates around :math:`(33,54,45)`.::
  
  from dsi2.aggregation.region_labeled_clusters import RegionLabelAggregator
  from dsi2.streamlines.track_math import sphere_around_ijk

  region_agg = RegionLabelAggregator()
  # give the aggregator access to the TrackDataSource
  region_agg.set_track_source(trk_src)
  
  # create a set of search coordinates
  sphere_radius = 2                  # voxels
  center_coordinate = (33,54,45)     # in MNI152 i,j,k coordinates
  search_coords = sphere_around_ijk(sphere_radius, center_coordinate)

  # Put the coordinates to use:
  region_agg.query_track_source_with_coords(search_coords)
  region_agg.update_clusters()

  # which regions pairs were found and how many streamlines to each?
  conn_ids, connection_vectors = region_agg.connection_vector_matrix()

  # if running an interactive ipython session, plot it
  region_agg.plot_connection_vector_lines(connection_vectors,conn_ids)

``conn_ids`` is a list of connections (regionA, regionB) found that pass through
the search coordinates. If there are :math:`n` individuals in the :py:class:`~dsi2.database.track_datasource.TrackDataSource`
and :math:`m` elements in ``conn_ids``, then ``connection_vectors`` will be an
:math:`n \times m` matrix where row :math:`i` column :math:`j` contains the streamline
count connection region pair math:`j` in subject :math:`i`\'s data.


Running a whole-brain LTPA
~~~~~~~~~~~~~~~~~~~~~~~~~~~





References
~~~~~~~~~~~

.. [1] Cieslak, M., & Grafton, S.T. Local termination pattern analysis:
    a tool for comparing white matter morphology. Brain Imaging Behav, DOI 10.1007/s11682-013-9254-z (2013).
