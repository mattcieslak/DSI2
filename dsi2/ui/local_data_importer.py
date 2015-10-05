#!/usr/bin/env python
import sys, json, os
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File, Int, Bool, cached_property
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn, SetEditor
from ..database.traited_query import Scan
from ..streamlines.track_dataset import TrackDataset
from ..streamlines.track_math import connection_ids_from_tracks
from ..volumes.mask_dataset import MaskDataset
from ..volumes import graphml_from_label_source 
import numpy as np
import multiprocessing
from dsi2.volumes import get_region_ints_from_graphml, b0_to_qsdr_map
from dsi2.database.mongodb import MongoCreator
from dsi2.database.mongodb import upload_local_scan, init_db

import time                                                

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r: %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed

from dipy.tracking.metrics import length as _length
@timeit
def length(tds):
    return np.array(
        [_length(stream) for stream in tds])

from dipy.tracking.metrics import winding as _winding
@timeit
def winding(tds):
    return np.array(
        [_winding(stream) for stream in tds])

from dipy.tracking.metrics import mean_curvature
@timeit
def curvature(tds):
    return np.array(
        [mean_curvature(stream) for stream in tds])

from dipy.tracking.metrics import center_of_mass as _center_of_mass
@timeit
def center_of_mass(tds):
    return np.array(
        [_center_of_mass(stream) for stream in tds])

    
function_registry = {
        "Length (voxels)":length, 
        "Winding":winding, 
        "Curvature":curvature,
        "Center of Mass":center_of_mass,
        #"Voxel Density":voxel_density
}

def matlab_variable(label_item):
    return  "_".join(
        ["%s_%s" % (k,v) for k,v in label_item.parameters.iteritems()]) + "_"

def measure_variable(s):
    return s.replace(" ","_").replace("(","").replace(")","").lower()

import multiprocessing, random, sys, os, time
# found http://stackoverflow.com/questions/23937189/how-do-i-use-subprocesses-to-force-python-to-release-memory/24126616#24126616
def _get_voxelized_and_properties(scan, state, measures):
    t0 = time.time()
    streamline_measures = {}
    streamlines = scan.get_voxel_coordinate_streamlines()
    # Calculate the measurements requested
    for func_name in measures:
        save_name = measure_variable(func_name)
        filename = scan.pkl_path+"."+save_name+".npy"
        if os.path.exists(filename):
            print "Loading", filename, "from disk" 
            streamline_measures[save_name] = np.load(filename)
        else:
            print "running",  func_name
            computation = function_registry[func_name]
            result = computation(streamlines)
            print "saving", filename
            np.save(filename,result)
            streamline_measures[save_name] = result
    state['streamline_measures'] = streamline_measures
    state['voxelized_streamlines'] = scan.get_voxelized_streamlines()
    t1 = time.time()
    state['time'] = t1 - t0

def create_connectivity_matrix(scan):
    print scan.connectivity_matrix_path
    if os.path.exists(scan.connectivity_matrix_path):
        print scan.connectivity_matrix_path, "exists - skipping"
        return
    from dsi2.aggregation.connectivity_matrix import ConnectivityMatrixCalculator
    manager = multiprocessing.Manager()
    state = manager.dict(streamline_properties={},voxelized_streamlines=[],time=0)  # shared state
    p = multiprocessing.Process(target=_get_voxelized_and_properties, args=(scan,state,function_registry.keys()))
    p.start()
    p.join()
    print 'time to sort: %.3f' % state['time']
    cmc= ConnectivityMatrixCalculator(
                        voxelized_steramlines=state['voxelized_streamlines'],
                        streamline_properties=state['streamline_properties'],
                        scan = scan)
    cmc.process_connectivity()
    cmc.save()

def create_missing_files(scan,overwrite=False):
    """
    """

    # Ensure that the path where pkls are to be stored exists
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
    
    # Perform all operations necessary to get labels from each label item
    scan.label_streamlines()
    # Read in aux data for streamlines
    scan.load_streamline_scalars()
    # write out a trk file that's usable by dsi studio
    scan.save_streamline_lookup_in_template_space()
    scan.clearmem()
    return True


scan_table = TableEditor(
    columns =
    [   ObjectColumn(name="scan_id",editable=True),
        ObjectColumn(name="study",editable=True),
        ObjectColumn(name="scan_group",editable=True),
        ObjectColumn(name="streamline_space",editable=True),
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
    b_process_inputs = Button(label="Process Local Data")
    b_create_connectivity_matrices = Button(label="Create Connectivity Matrices")
    input_directory = File()
    output_directory = File()
    n_processors = Int(1)
    overwrite = Bool(False)
    downsample_before_upload = Enum("Approximate Polygon","Approdimate MDL","None")
    
    def _connect_to_mongod_fired(self):
        self.mongo_creator.edit_traits()
        
    def _mongo_creator_default(self):
        return MongoCreator()
    
    def create_connectivity_matrices(self):
        print "Creating connectivity matrices"
        if self.n_processors > 1:
            print "Using %d processors" % self.n_processors
            pool = multiprocessing.Pool(processes=self.n_processors)
            result = pool.map(create_connectivity_matrix, self.datasets)
            pool.close()
            pool.join()
        else:    
            for scan in self.datasets:
                create_connectivity_matrix(scan)
        print "Finished!"
    
    def _b_create_connectivity_matrices_fired(self):
        self.create_connectivity_matrices()
    
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
        
    def save_json(self):
        json_data = [scan.to_json() for scan in self.datasets]
        with open(self.json_file,"w") as outfile:
            json.dump(json_data,outfile,indent=4)
        print "Saved", self.json_file
        
    def _save_fired(self):
        self.save_json()
        
    def process_inputs(self):
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
    
    def _b_process_inputs_fired(self):
        self.process_inputs()
    
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
            p = multiprocessing.Process(target=upload_local_scan, 
                                        args=(db,scan,self.downsample_before_upload))
            p.start()
            p.join()
            

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
                Item("b_process_inputs"),
                Item("connect_to_mongod"),
                Item("upload_to_mongodb"),
                Item("b_create_connectivity_matrices"),
                orientation="horizontal",
                show_labels=False
                ),
            Group(
                Item("overwrite"),
                Item("n_processors"),
                Item("downsample_before_upload"),
                orientation="horizontal",
                show_labels=True
                ),
        resizable=True,
        width=900,
        height=500,
        title="Import Tractography Data"

    )
