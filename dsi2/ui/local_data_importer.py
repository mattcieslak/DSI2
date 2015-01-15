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
from ..volumes import get_NTU90, graphml_from_label_source
import networkx as nx
import numpy as np
import nibabel as nib
import gzip
from scipy.io.matlab import loadmat

import dsi2.config
#pkl_dir = dsi2.config.local_trackdb_path


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
    QSDR_nim = get_NTU90()
    QSDR_data = QSDR_nim.get_data()

    # Labels in b0 space
    _old_atlas = nib.load(b0_atlas)
    old_atlas = _old_atlas.get_data()
    old_aff = _old_atlas.get_affine()
    # QSDR maps from RAS+ space.  Force the input volume to conform
    if old_aff[0,0] < 0:
        print "\t\t+++ Flipping X axis"
        old_atlas = old_atlas[::-1,:,:]
    if old_aff[1,1] < 0:
        print "\t\t+++ Flipping Y axis"
        old_atlas = old_atlas[:, ::-1, :]
    if old_aff[2,2] < 0:
        print "\t\t+++ Flipping Z axis"
        old_atlas = old_atlas[:, :, ::-1]
    
    # Fill up the output atlas with labels from b0, collected through the fib mappings
    new_atlas = old_atlas[mx,my,mz].reshape(volume_dimension,order="C")
    aff = QSDR_nim.get_affine()
    aff[(0,1,2),(0,1,2)]*=2
    onim = nib.Nifti1Image(new_atlas,aff)
    onim.to_filename(output_v)


def create_missing_files(scan,input_dir, output_dir):
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

    ## Ensure that the path where pkls are to be stored exists
    abs_pkl_file = os.path.join(output_dir,scan.pkl_path)
    pkl_directory = os.path.split(abs_pkl_file)[0]
    if not os.path.exists(pkl_directory):
        print "\t+ making directory for pkl_files"
        os.makedirs(pkl_directory)
    print "\t\t++ pkl_directory is", pkl_directory


    if os.path.isabs(scan.trk_file):
        abs_trk_file = scan.trk_file
    else:
        abs_trk_file = os.path.join(input_dir,scan.trk_file)
    # Check that the pkl file exists, or the trk file
    if not os.path.exists(abs_pkl_file):
        if not os.path.exists(abs_trk_file):
            raise ValueError(abs_trk_file + " does not exist")
    # Load the tracks
    print "\t+ loading", abs_trk_file
    tds = TrackDataset(abs_trk_file)
    # NOTE: If these were MNI 152 @ 1mm, we could do something like
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
            os.path.join(output_dir,label_source.numpy_path)
        print "\t\t++ Ensuring %s exists" % npy_path
        if os.path.exists(npy_path):
            print "\t\t++", npy_path, "already exists"
            continue

        # Check to see if the qsdr volume exists. If not, create it from
        # the B0 volume
        abs_qsdr_path = label_source.qsdr_volume_path if os.path.isabs(
            label_source.qsdr_volume_path) else os.path.join(
                output_dir,label_source.qsdr_volume_path)
        abs_b0_path = label_source.b0_volume_path if os.path.isabs(
            label_source.b0_volume_path) else os.path.join(
                input_dir,label_source.b0_volume_path)
        abs_fib_file = scan.fib_file if os.path.isabs(
            scan.fib_file ) else os.path.join(
                input_dir,scan.fib_file )

        if not os.path.exists(abs_qsdr_path):
            # If neither volume exists, the data is incomplete
            if not os.path.exists(abs_b0_path):
                print "\t\t++ ERROR: must have a b0 volume and .map.fib.gz OR a qsdr_volume"
                continue
            print "\t\t++ mapping b0 labels to qsdr space"
            b0_to_qsdr_map(abs_fib_file, abs_b0_path,
                           abs_qsdr_path)

        print "\t\t++ Loading volume %d/%d:\n\t\t\t %s" % (
                lnum, n_labels, abs_qsdr_path )
        mds = MaskDataset(abs_qsdr_path)
        
        # Get the region labels from the parcellation
        graphml = graphml_from_label_source(label_source)
        if graphml is None:
            print "\t\t++ No graphml exists: using unique region labels"
            regions = mds.roi_ids
        else:
            print "\t\t++ Recognized atlas name, using Lausanne2008 atlas", graphml
            regions = __get_region_ints_from_graphml(graphml)

        # Save it.
        conn_ids = connection_ids_from_tracks(mds, tds,
              save_npy=npy_path,
              scale_coords=tds.header['voxel_size'],
              region_ints=regions)
        print "\t\t++ Saved %s" % npy_path
        print "\t\t\t*** %.2f\% streamlines not accounted for by regions"%(
                np.sum(conn_ids==0)/len(conn_ids)*100.)

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
        abs_pkl_trk_file = os.path.join(output_dir,scan.pkl_trk_path)
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
    upload_to_mongodb = Button()
    process_inputs = Button()
    input_directory = File()
    output_directory = File()

    def _json_file_changed(self):
        if not os.path.exists(self.json_file):
            print "no such file", self.json_file
            return
        fop = open(self.json_file, "r")
        jdata = json.load(fop)
        fop.close()
        self.datasets = [
          Scan(pkl_dir=self.output_directory,
               data_dir="", **d) for d in jdata]
        
    def _save_fired(self):
        json_data = [scan.to_json() for scan in self.datasets]
        with open(self.json_file,"w") as outfile:
            json.dump(json_data,outfile,indent=4)
        print "Saved", self.json_file
        pass
    
    def _process_inputs_fired(self):
        print "Processing input data"
        for scan in self.datasets:
            create_missing_files(scan,"","")
        print "Finished!"
    

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
                Item("process_inputs"),
                Item("save"),
                Item("upload_to_mongodb"),
                orientation="horizontal",
                show_labels=False
                ),
            orientation="vertical"
        ),
        resizable=True,
        width=900,
        height=500,
        title="Import Tractography Data"

    )



