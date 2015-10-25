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
from dsi2.streamlines.trajectory_filtering import TrajectoryFilter
from copy import deepcopy
from glob import glob
import re
import cPickle as pickle
import time                                                


class TrajectoryTrainer(HasTraits):
    """
    Holds a list of Scan objects. These can be loaded from
    and saved to a json file.
    """
    save = Button()
    input_directory = File()
    trajectory_filter = Instance(TrajectoryFilter)
    
    @on_trait_change("input_directory")
    def setup_trajectory_filter(self):
        # get the pkl file
        pkl_file = glob(self.input_directory + "/*.pkl")
        assert len(pkl_file) == 1
        pkl_file = pkl_file[0]
        
        # Check the npy files
        npy_files = glob(self.input_directory + "/*.npy")
        streamline_properties = {}
        for npy_file in npy_files:
            measure_name = re.match(".*\.trk\.(.*)\.npy",npy_file).groups()[0].strip(".txt")
            if measure_name.startswith("pkl."): measure_name=measure_name[4:]
            print "found measure:", measure_name
            data = np.load(npy_file)
            data[np.isnan(data)] = -1
            streamline_properties[measure_name] = data
        
        # What is the reference volume?
        ref_vol = os.path.join(self.input_directory,"native_T1.nii.gz")
        
        output_prefix = os.path.join(self.input_directory,"streamline_classification")
            
        tf = TrajectoryFilter(streamline_properties=streamline_properties,
                              reference_volume=ref_vol, output_prefix=output_prefix)
        
        print "Loading pkl file"
        fop = open(pkl_file,"rb")
        tf.set_track_dataset(pickle.load(fop))
        fop.close()
        self.trajectory_filter = tf
        
    def _save_fired(self):
        self.trajectory_filter.save()
        

    # UI definition for the local db
    traits_view = View(
            Item("input_directory"),Item("save"),
            Group(
            Item("trajectory_filter",style="custom"),
            show_labels=False),
            resizable=True
    )
