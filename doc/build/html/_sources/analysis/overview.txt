Organizing your local datasource (Hard way)
============================================

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
        "software": "DSI Studio",
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

