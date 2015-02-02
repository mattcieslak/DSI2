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
from ..volumes import get_NTU90, graphml_from_label_source, get_builtin_atlas_parameters
import networkx as nx
import numpy as np
import nibabel as nib
import gzip
from scipy.io.matlab import loadmat
import subprocess
import cPickle as pickle
import pymongo
from bson.binary import Binary
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
import re
import multiprocessing

def init_db(db):
    """
    adds collections and makes sure that indexes work for them
    """
    db.streamlines.ensure_index([("scan_id",pymongo.ASCENDING),
        ("sl_id",pymongo.ASCENDING)])
    db.coordinates.ensure_index([("scan_id",pymongo.ASCENDING),
        ("ijk",pymongo.ASCENDING)])
    db.connections.ensure_index([("con_id",pymongo.ASCENDING),
        ("scan_id",pymongo.ASCENDING),("atlas_id",pymongo.ASCENDING)])
    db.streamline_labels.ensure_index([("scan_id",pymongo.ASCENDING),
        ("atlas_id",pymongo.ASCENDING)])
    db.atlases.ensure_index([("name",pymongo.ASCENDING)])
    db.scans.ensure_index([("scan_id",pymongo.ASCENDING),("subject_id",pymongo.ASCENDING)])


def upload_atlases(db, trackds, sc):
    """
    Reads the atlas info from a Scan and loads the npy files from disk. Then

    1) uploads the atlas info into db.atlases
    2) uploads the label array for each atlas/scan into db.streamline_labels 
    3) uploads the streamline ids for each connection in each atlas/scan into connections

    """
    try:
        atlases = []
        logging.info("processing %d atlases for %s", len(trackds.properties.track_label_items), sc.scan_id)
        for label in trackds.properties.track_label_items:
            #pdb.set_trace()
            atlas_labels = label.load_array()

            # Does this atlas already exist? If not, add it to the collection.
            atlas = None
            result = db.atlases.find( { "name": label.name, "parameters": label.parameters } )
            if result.count() != 0:
                atlas = result[0]["_id"]
            else:
                atlas = db.atlases.insert( { "name": label.name, "parameters": label.parameters } )

            atlases.append(atlas)
            db.streamline_labels.insert([
                    {
                        "scan_id": sc.scan_id,
                        "atlas_id": atlas,
                        "con_ids": list(map(int,atlas_labels))
                    }
            ])

            # -------------------------------------------
            # Insert data into the connections collection
            # -------------------------------------------
            inserts = []
            con_ids = set(atlas_labels)
            print "Building connections collection for %s %d..." % (label.name, label.parameters["scale"])
            for con_id in con_ids:
                sl_ids = list(map(int,np.where(atlas_labels == con_id)[0]))
                inserts.append(
                        {
                            "con_id":"%d" % con_id,
                            "scan_id":sc.scan_id,
                            "atlas_id":atlas,
                            "sl_ids":sl_ids
                        }
                )
            db.connections.insert(inserts)
            print "done."
    except Exception, e:
        print "Failed to upload atlases", e
        return False
    return True


def upload_streamlines(db, trackds, sc):
    """
    Inserts the binary streamline data into db.streamlines
    """
    try:
        logging.info("Building streamline collection")
        inserts = []
        for ntrk, trk in enumerate(trackds.tracks):
            # continue appending to inserts until it gets too big
            if len(inserts) >= 1000:
                # then insert it and clear inserts
                db.streamlines.insert(inserts)
                inserts = []
            inserts.append(
                {
                  "scan_id":sc.scan_id,
                  "sl_id": ntrk,
                  "data":Binary(pickle.dumps(trk,protocol=2))
                 }
            )
        # Finally, insert the leftovers
        db.streamlines.insert(inserts)
    except Exception, e:
        print "Failed to upload streamline data", e
        return False
    return True


def upload_coordinate_info(db, trackds, sc):
    try:
        inserts = []
        print "Building coordinate collection..."
        for coord,indices in trackds.tracks_at_ijk.iteritems():
            inserts.append(
               {
                "ijk":"(%d, %d, %d)" % tuple(map(int,coord)),
                "scan_id":sc.scan_id,
                "sl_id":list(map(int,indices))
               }
              )
        db.coordinates.insert(inserts)
        print "done."
    except Exception, e:
        print "Failed to upload coordinate info", e
        return False

    return True

def upload_scan_info(db, trackds, sc):
    try:
        #db.scans.insert([sc.original_json])
        atlases = []
        for label in sc.track_label_items:
            # Does this atlas already exist? If not, add it to the collection.
            atlas = None
            result = db.atlases.find( { "name": label.name, "parameters": label.parameters } )
            if result.count() != 0:
                atlas = result[0]["_id"]
            else:
                atlas = db.atlases.insert( { "name": label.name, "parameters": label.parameters } )
    
            atlases.append(atlas)
        db.scans.insert([
                {
                    "scan_id":sc.scan_id,
                    "subject_id":sc.subject_id,
                    "gender":sc.scan_gender,
                    "age":sc.scan_age,
                    "study":sc.study,
                    "group":sc.scan_group,
                    "smoothing":sc.smoothing,
                    "cutoff_angle":sc.cutoff_angle,
                    "qa_threshold":sc.qa_threshold,
                    "gfa_threshold":sc.gfa_threshold,
                    "length_min":sc.length_min,
                    "length_max":sc.length_max,
                    "institution":sc.institution,
                    "reconstruction":sc.reconstruction,
                    "scanner":sc.scanner,
                    "n_directions":sc.n_directions,
                    "max_b_value":sc.max_b_value,
                    "bvals":sc.bvals,
                    "bvecs":sc.bvecs,
                    "label":sc.label,
                    "trk_space":sc.trk_space,
                    "atlases":list(set(atlases)),
                    "sls": len(trackds.tracks),
                    "header":Binary(pickle.dumps(trackds.header,protocol=2)),
                    "original_json":sc.original_json
                }
        ])
    except Exception, e:
        print "Failed to upload scan info", e
        return False
    return True


def upload_local_scan(db, sc):
    logging.info("uploading %s", sc.scan_id)
    try:
        trackds = sc.get_track_dataset()
    except:
        print "failed to read pkl file"
        return False, "pkl file corrupt"

    if not upload_atlases(db, trackds, sc):
        print "failed to upload atlases"
        return False, "upload_atlases"

    if not upload_streamlines(db, trackds, sc):
        print "failed to upload streamlines"
        return False, "upload_streamlines"

    if not upload_coordinate_info(db, trackds, sc):
        print "failed to upload spatial mapping"
        return False, "upload_coordinate_info"

    if not upload_scan_info(db, trackds, sc):
        print "failed to upload spatial mapping"
        return False, "upload scan info"

    return True, "hooray!"

def check_scan_for_files(sc):
    """Checks to make sure that all the necessary files for this scan are on disk.
    If they are, it returns True, otherwise False
    """
    pkl_file = sc.pkl_path
    if not os.path.exists(pkl_file):
        print "Unable to locate pickle file %s" % pkl_file
        logging.error("Unable to locate pickle file %s", pkl_file)
        return False

    # Check that all the npy files exist
    for label in sc.track_label_items:
        npy_path = label.numpy_path
        if not os.path.exists(npy_path):
            print "unable to load %s" % npy_path
            logging.error("unable to load %s" % npy_path)
            return False
    return True


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
    if old_aff[0,0] > 0:
        print "\t\t+++ Flipping X"
        old_atlas = old_atlas[::-1,:,:]
    if old_aff[1,1] > 0:
        print "\t\t+++ Flipping Y"
        old_atlas = old_atlas[:,::-1,:]
    if old_aff[2,2] < 0:
        print "\t\t+++ Flipping Z"
        old_atlas = old_atlas[:,:,::-1]
        
    # XXX: there is an error when importing some of the HCP datasets where the
    # map-from index is out of bounds from the b0 image. This will check for
    # any indices that would cause an index error and sets them to 0.
    bx, by, bz = old_atlas.shape
    idx_err_x = np.flatnonzero( mx >= bx)
    if len(idx_err_x):
        print "\t\t+++ WARNING: %d voxels are out of original data x range" % len(idx_err_x)
        mx[idx_err_x] = 0
    idx_err_y = np.flatnonzero( my >= by)
    if len(idx_err_y):
        print "\t\t+++ WARNING: %d voxels are out of original data y range" % len(idx_err_y)
        my[idx_err_y] = 0
    idx_err_z = np.flatnonzero( mz >= bz)
    if len(idx_err_z):
        print "\t\t+++ WARNING: %d voxels are out of original data z range" % len(idx_err_z)
        mz[idx_err_z] = 0
        
    
    # Fill up the output atlas with labels from b0, collected through the fib mappings
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

    ## Ensure that the path where pkls are to be stored exists
    sid = scan.scan_id
    abs_pkl_file = scan.pkl_path
    pkl_directory = os.path.split(abs_pkl_file)[0]
    if not os.path.exists(pkl_directory):
        print "\t+ [%s] making directory for pkl_files" % sid
        os.makedirs(pkl_directory)
    print "\t\t++ [%s] pkl_directory is" % sid, pkl_directory


    if os.path.isabs(scan.trk_file):
        abs_trk_file = scan.trk_file
    else:
        abs_trk_file = os.path.join(input_dir,scan.trk_file)
    # Check that the pkl file exists, or the trk file
    if not os.path.exists(abs_pkl_file):
        if not os.path.exists(abs_trk_file):
            raise ValueError(abs_trk_file + " does not exist")
    # Load the tracks
    print "\t+ [%s] loading" %sid , abs_trk_file
    tds = TrackDataset(abs_trk_file)
    # NOTE: If these were MNI 152 @ 1mm, we could do something like
    # tds.tracks_at_ijk = streamline_mapping(tds.tracks,(1,1,1))
    print "\t+ [%s] hashing tracks in qsdr space"%sid
    tds.hash_voxels_to_tracks()
    print "\t\t++ [%s] Done." % sid


    # =========================================================
    # Loop over the track labels, creating .npy files as needed
    n_labels = len(scan.track_label_items)
    print "\t+ [%s] Intersecting"%sid, n_labels, "label datasets"
    for lnum, label_source in enumerate(scan.track_label_items):
        # Load the mask
        # File containing the corresponding label vector
        npy_path = label_source.numpy_path if \
            os.path.isabs(label_source.numpy_path) else \
            os.path.join(output_dir,label_source.numpy_path)
        print "\t\t++ [%s] Ensuring %s exists" % (sid, npy_path)
        if os.path.exists(npy_path):
            print "\t\t++ [%s]"%sid, npy_path, "already exists"
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
            regions = __get_region_ints_from_graphml(graphml)
            if not len(label_source.parameters):
                label_source.parameters = get_builtin_atlas_parameters(graphml)
            

        # Save it.
        conn_ids = connection_ids_from_tracks(mds, tds,
              save_npy=npy_path,
              scale_coords=tds.header['voxel_size'],
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


class MongoCreator(HasTraits):
    database_dir = File()
    log_path = File()
    b_start = Button("Start mongod")
    restrict_ips = Bool(True)
    numactl_interleave = Bool(False)
    
    def get_command(self):
        cmd = []
        if self.numactl_interleave:
            cmd += ["numactl", "--interleave=all" ]
        
        cmd += ["mongod", "--fork", "--dbpath", self.database_dir,
                "--logpath", self.log_path ]
        
        if self.restrict_ips:
            cmd += ["--bind_ip", "127.0.0.1"]
        return cmd
        
    def _b_start_fired(self):
        print "Starting mongod"
        cmd = self.get_command()
        print cmd
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir)
        proc = subprocess.Popen(cmd,
                                stdout = subprocess.PIPE)
        result = proc.communicate()
        print result
        
    traits_view = View(
        VGroup(
            Item("database_dir"),
            Item("log_path"),
            Item("restrict_ips"),
            Item("numactl_interleave"),
            Item("b_start")
            )
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
            connection = pymongo.MongoClient()
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
        Group(
            Item("json_file"),
            Group(
                Item("datasets",editor = scan_table),
                orientation="horizontal",
                show_labels=False
                ),
            Group(
                Item("process_inputs"),
                Item("connect_to_mongod"),
                Item("upload_to_mongodb"),
                orientation="horizontal",
                show_labels=False
                ),
            Item("n_processors"),
            orientation="vertical"
        ),
        resizable=True,
        width=900,
        height=500,
        title="Import Tractography Data"

    )
