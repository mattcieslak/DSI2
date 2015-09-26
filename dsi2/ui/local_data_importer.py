#!/usr/bin/env python
import sys, json, os
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File, Int, Bool, cached_property
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn
from ..database.traited_query import Scan
from ..streamlines.track_dataset import TrackDataset
from ..streamlines.track_math import connection_ids_from_tracks
from ..volumes.mask_dataset import MaskDataset
from ..volumes import graphml_from_label_source 
import numpy as np
import multiprocessing
from dsi2.volumes import get_region_ints_from_graphml
from dsi2.database.mongodb import MongoCreator


class lazy_tds(HasTraits):
    file_name = File
    tds = Property(Instance(TrackDataset))
    
    @cached_property
    def _get_tds(self):
        # Load the tracks
        tds = TrackDataset(self.file_name)
        print "\t+ [%s] hashing tracks in qsdr space"
        tds.hash_voxels_to_tracks()
        print "\t\t++ [%s] Done." 
        return tds

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

    ## Ensure that the path where pkls are to be stored exists
    sid = scan.scan_id
    abs_pkl_file = scan.pkl_path
    pkl_directory = os.path.split(abs_pkl_file)[0]
    if not os.path.exists(pkl_directory):
        print "\t+ [%s] making directory for pkl_files" % sid
        os.makedirs(pkl_directory)
    print "\t\t++ [%s] pkl_directory is" % sid, pkl_directory


    abs_trk_file = scan.trk_file
    # Check that the pkl file exists, or the trk file
    if not os.path.exists(abs_pkl_file):
        if not os.path.exists(abs_trk_file):
            raise ValueError(abs_trk_file + " does not exist")
    
    # prevent loading and hashing of tracks unless necessary
    # to create missing files
    tds = lazy_tds(file_name=abs_trk_file, subject_name=sid)

    # =========================================================
    # Loop over the track labels, creating .npy files as needed
    n_labels = len(scan.track_label_items)
    print "\t+ [%s] Intersecting"%sid, n_labels, "label datasets"
    for lnum, label_source in enumerate(scan.track_label_items):
        # Load the mask
        # File containing the corresponding label vector
        npy_path = label_source.numpy_path 
        print "\t\t++ [%s] Ensuring %s exists" % (sid, npy_path)
        if os.path.exists(npy_path):
            print "\t\t++ [%s]"%sid, npy_path, "already exists"
            continue

        # Check to see if the qsdr volume exists. If not, create it from
        # the B0 volume
        abs_qsdr_path = label_source.qsdr_volume_path
        abs_b0_path = label_source.b0_volume_path 
        abs_fib_file = scan.fib_file

        if not os.path.exists(abs_qsdr_path):
            # If neither volume exists, the data is incomplete
            if not os.path.exists(abs_b0_path):
                print "\t\t++ [%s] ERROR: must have a b0 volume and .map.fib.gz OR a qsdr_volume"%sid
                continue
            print "\t\t++ [%s] mapping b0 labels to qsdr space"%sid
            b0_to_qsdr_map(abs_fib_file, abs_b0_path,
                           abs_qsdr_path)

        print "\t\t++ [%s] Loading volume %d/%d:\n\t\t\t %s" % (
                sid, lnum + 1, n_labels, abs_qsdr_path )
        mds = MaskDataset(abs_qsdr_path)
        
        # Get the region labels from the parcellation
        graphml = graphml_from_label_source(label_source)
        if graphml is None:
            print "\t\t++ [%s] No graphml exists: using unique region labels"%sid
            regions = mds.roi_ids
        else:
            print "\t\t++ [%s] Recognized atlas name, using Lausanne2008 atlas"%sid, graphml
            regions = get_region_ints_from_graphml(graphml)

        # Save it.
        conn_ids = connection_ids_from_tracks(mds, tds.tds,
              save_npy=npy_path,
              scale_coords=tds.tds.header['voxel_size'],
              region_ints=regions)
        print "\t\t++ [%s] Saved %s" % (sid, npy_path)
        print "\t\t\t*** [%s] %.2f percent streamlines not accounted for by regions"%( sid, 100. * np.sum(conn_ids==0)/len(conn_ids) )

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
        fop = open(label_source.txt_path,"r")
        scalars = np.array(
            [np.fromstring(line,sep=" ").mean() for line in fop] )
        fop.close()
        np.save(npy_path,scalars)
        print "\t\t++ Done."

    if not os.path.isabs(scan.pkl_trk_path):
        abs_pkl_trk_file = os.path.join(output_dir,scan.pkl_trk_path)
    else:
        abs_pkl_trk_file = scan.pkl_trk_path
    print "\t+ Dumping MNI152 hash table"
    tds.tds.dump_qsdr2MNI_track_lookup(abs_pkl_file,abs_pkl_trk_file)
    return True


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
    mongo_creator = Instance(MongoCreator)
    upload_to_mongodb = Button()
    connect_to_mongod = Button()
    process_inputs = Button()
    input_directory = File()
    output_directory = File()
    n_processors = Int(1)
    def _connect_to_mongod_fired(self):
        self.mongo_creator.edit_traits()
        
    def _mongo_creator_default(self):
        return MongoCreator()
    
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
        if self.n_processors > 1:
            print "Using %d processors" % self.n_processors
            pool = multiprocessing.Pool(processes=self.n_processors)
            result = pool.map(create_missing_files, self.datasets)
            pool.close()
            pool.join()
        else:    
            for scan in self.datasets:
                create_missing_files(scan)
        print "Finished!"
    
    def _upload_to_mongodb_fired(self):
        print "Connecting to mongodb"
        try:
            connection = self.mongo_creator.get_connection()
        except Exception:
            print "unable to establish connection with mongod"
            return
        db = connection.dsi2
        print "initializing dsi2 indexes"
        init_db(db)
        
        print "Uploading to MongoDB"
        for scan in self.datasets:
            print "\t+ Uploading", scan.scan_id
            if not check_scan_for_files(scan):
                raise ValueError("Missing files found for " + scan.scan_id)
            upload_succeeded, because = upload_local_scan(db, scan)
            if not upload_succeeded:
                raise ValueError(because)
            

    # UI definition for the local db
    traits_view = View(
            Item("json_file"),
            Group(
                Item("datasets",editor = scan_table),
                orientation="horizontal",
                show_labels=False
                ),
            Group(
                Item("save"),
                Item("process_inputs"),
                Item("connect_to_mongod"),
                Item("upload_to_mongodb"),
                orientation="horizontal",
                show_labels=False
                ),
            Item("n_processors"),
        resizable=True,
        width=900,
        height=500,
        title="Import Tractography Data"

    )
