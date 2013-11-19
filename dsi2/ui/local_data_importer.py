#!/usr/bin/env python
import sys, json, os
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File, Int
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn
from ..database.traited_query import Scan
from ..streamlines.track_dataset import TrackDataset
from ..streamlines.track_math import connection_ids_from_tracks
from ..volumes.mask_dataset import MaskDataset
import networkx as nx
import numpy as np
import nibabel as nib
import gzip
from scipy.io.matlab import loadmat

dsi2_data = os.getenv("DSI2_DATA")
local_tdb_var = os.getenv("LOCAL_TRACKDB")
home_pkl  = os.path.join(os.getenv("HOME"),"local_trackdb")
if local_tdb_var:
    pkl_dir = local_tdb_var
    print "Using $LOCAL_TRACKDB environment variable",
elif os.path.exists(home_pkl):
    pkl_dir = home_pkl
    print "Using local_trackdb in home directory for data"
if dsi2_data:
    print "Using $DSI2_DATA environment variable",
else:
    raise OSError("DSI2_DATA needs to be set")

def __get_region_ints_from_graphml(graphml):
    """
    Returns an array of region ints from a graphml file.
    """
    graph = nx.read_graphml(graphml)
    return sorted(map(int, graph.nodes()))

def b0_to_qsdr_map(fib_file, b0_atlas, output_v):
    """
    Creates a qsdr atlas from a DSI Studio fib file and a b0 atlas.
    """
    # Load the mapping from the fib file
    fibf = gzip.open(fib_file,"rb")
    m = loadmat(fibf)
    fibf.close()
    volume_dimension = m['dimension'].squeeze().astype(int)
    mx = m['mx'].squeeze().astype(int)
    my = m['my'].squeeze().astype(int)
    mz = m['mz'].squeeze().astype(int)
    
    # Load the QSDR template volume from DSI studio
    QSDR_vol = os.path.join("/storage2/cieslak/bin/dsi_studio64/dsi_studio_64/NTU90_QA.nii.gz")
    QSDR_nim = nib.load(QSDR_vol)
    QSDR_data = QSDR_nim.get_data()
    
    # Labels in b0 space
    old_atlas = nib.load(b0_atlas).get_data()
    
    # Fill up the output atlas with labels from b0,collected through the fib mappings
    new_atlas = old_atlas[mx,my,mz].reshape(volume_dimension,order="F")
    aff = QSDR_nim.get_affine()
    aff[(0,1,2),(0,1,2)]*=2
    onim = nib.Nifti1Image(new_atlas,aff)
    onim.to_filename(output_v)

def create_missing_files(scan):
    """
    Creates files on disk that are needed to visualize data

    Discrete space indexing
    -----------------------
    If the file stored in ``pkl_file`` does not exist,
    The ``trk_file`` attribute is loaded and indexed in MNI152
    space.
    Looks into all the track_labels and track_scalars and ensures
    that they exist at loading time

    """
    # Ensure that the path where pkls are to be stored exists
    if not os.path.isabs(scan.pkl_path):
        abs_pkl_file = os.path.join(scan.pkl_dir,scan.pkl_path)
    else:
        abs_pkl_file = scan.pkl_path
    pkl_directory = os.path.split(abs_pkl_file)[0]
    if not os.path.exists(pkl_directory):
        print "\t+ making directory for pkl_files"
        os.makedirs(pkl_directory)
    print "\t\t++ pkl_directory is", pkl_directory

    # Check that the pkl file exists
    if not os.path.exists(abs_pkl_file):
        if not os.path.exists(scan.trk_file):
            raise ValueError(scan.trk_file + " does not exist")
    # Load the tracks
    print "\t+ loading", scan.trk_file
    tds = TrackDataset(fname=scan.trk_file)
    ## NOTE: If these were MNI 152 @ 1mm, we could do something like
    # tds.tracks_at_ijk = streamline_mapping(tds.tracks,(1,1,1))
    print "\t+ hashing tracks in qsdr space"
    tds.hash_voxels_to_tracks()
    print "\t\t++ Done."


    # =========================================================
    # Loop over the track labels, creating .npy files as needed
    n_labels = len(scan.track_label_items)
    print "\t+ Intersecting", n_labels, "label datasets"
    for lnum, label_source in enumerate(scan.track_label_items):
        # Load the mask
        # File containing the corresponding label vector
        npy_path = label_source.numpy_path if \
            os.path.isabs(label_source.numpy_path) else \
            os.path.join(scan.pkl_dir,label_source.numpy_path)
        print "\t\t++ Ensuring %s exists" % npy_path
        if os.path.exists(npy_path):
            print "\t\t++", npy_path, "already exists"
            continue
        print "\t\t++ Loading volume %d/%d:\n\t\t\t %s" % (
                lnum, n_labels, label_source.volume_path )
        mds = MaskDataset(label_source.volume_path)

        # Get the region labels from the parcellation
        if label_source.graphml_path == "":
            print "\t\t++ No graphml exists: using unique region labels"
            regions = mds.roi_ids
        else:
            print "\t\t++ Using graphml regions",label_source.graphml
            regions = __get_region_ints_from_graphml(label_source.graphml_path)

        # Save it.
        conn_ids = connection_ids_from_tracks(mds, tds,
              save_npy=npy_path,
              scale_coords=tds.header['voxel_size'],
              region_ints=regions)
        print "\t\t++ Saved %s" % npy_path
        print "\t\t\t*** %.2f streamlines not accounted for by regions"%(
                np.sum(conn_ids==0)/float(len(conn_ids)))

    # =========================================================
    # Loop over the track scalars, creating .npy files as needed
    print "\t Dumping trakl GFA/QA values"
    for label_source in scan.track_scalar_items:
        # File containing the corresponding label vector
        npy_path = label_source.numpy_path if \
            os.path.isabs(label_source.numpy_path) else \
            os.path.join(scan.pkl_dir,label_source.numpy_path)
        if os.path.exists(npy_path):
            print npy_path, "already exists"
            continue
        print "\t\t++ saving values to", npy_path
        scalars = np.loadtxt(label_source.text_path)
        np.save(npy_path,scalars)
        print "\t\t++ Done."

    if not os.path.isabs(scan.pkl_trk_path):
        abs_pkl_trk_file = os.path.join(scan.pkl_dir,scan.pkl_trk_path)
    else:
        abs_pkl_trk_file = scan.pkl_trk_path
    print "\t+ Dumping MNI152 hash table"
    tds.dump_qsdr2MNI_track_lookup(abs_pkl_file,abs_pkl_trk_file)

scan_table = TableEditor(
    columns =
    [   ObjectColumn(name="scan_id",editable=True),
        ObjectColumn(name="study",editable=True),
        ObjectColumn(name="scan_group",editable=True),
        ObjectColumn(name="software",editable=True),
        ObjectColumn(name="reconstruction",editable=True),
    ],
    deletable  = True,
    auto_size  = True,
    show_toolbar = True,
    edit_view="import_view",
    row_factory=Scan
    #edit_view_height=500,
    #edit_view_width=500,
    )

class LocalDataImporter(HasTraits):
    """
    Holds a list of Scan objects. These can be loaded from
    and saved to a json file.
    """
    json_file = File()
    datasets = List(Instance(Scan))
    save = Button()

    def _json_file_changed(self):
        if not os.path.exists(self.json_file):
            print "no such file", self.json_file
            return
        fop = open(self.json_file, "r")
        jdata = json.load(fop)
        fop.close()
        self.datasets = [
          Scan(pkl_dir=pkl_dir, data_dir=dsi2_data, **d) for d in jdata]

    def validate_localdb(self):
        """ Looks at every entry in the loaded db and
        checks that the loadable files exist.
        """
        for scan in self.datasets:
            create_missing_files(scan)



    # UI definition for the local db
    traits_view = View(
        Group(
            Item("json_file"),
            Group(
                Item("datasets",editor = scan_table),
                orientation="horizontal",
                show_labels=False
                ),
            Group(
                Item("save"),
                orientation="horizontal"
                ),
            orientation="vertical"
        ),
        resizable=True,
        width=900,
        height=500

    )



