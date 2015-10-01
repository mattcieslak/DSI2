#!/usr/bin/env python
import sys, json, os
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File, Int, Bool, cached_property
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn, SetEditor
import numpy as np
from dsi2.database.traited_query import Scan
from dsi2.volumes.mask_dataset import MaskDataset
"""
Connectivity self.data can calculate many things.  Functions work by returning
an array with one value per streamline
"""

from dipy.tracking.metrics import length as _length
def length(tds):
    return np.array(
        [_length(stream) for stream in tds.tracks])

from dipy.tracking.metrics import winding as _winding
def winding(tds):
    return np.array(
        [_length(stream) for stream in tds.tracks])

from dipy.tracking.metrics import mean_curvature
def curvature(tds):
    return np.array(
        [mean_curvature(stream) for stream in tds.tracks])

from dipy.tracking.metrics import center_of_mass
def center_of_mass(tds):
    return np.array(
        [center_of_mass(stream) for stream in tds.tracks])

def voxel_density(tds):
    pass
    
function_registry = {
        "Length (voxels)":length, 
        "Winding":winding, 
        "Curvature":curvature,
        "Center of Mass":center_of_mass,
        "Voxel Density":voxel_density
}

def matlab_variable(label_item):
    return  "_".join(
            ["%s_%s" % (k,v) for k,v in label_item.parameters.iteritems()]) + "_"

def measure_variable(s):
    return s.replace(" ","_").replace("(","").replace(")","").lower()

class ConnectivityMatrixCalculator(HasTraits):
    measures = List( editor = SetEditor(
        values = sorted(function_registry.keys()),
        left_column_title  = 'Available Measurements',
        right_column_title = 'Measurements to Calculate'))    
    save_file = File("")
    calcluate_region_summaries = Bool(True)
    scan = Instance(Scan)
    
    def __init__(self,**traits):
        super(ConnectivityMatrixCalculator,self).__init__(**traits)
        # everything gets stored in data until it gets saved
        self.data = {}
        
    def get_streamline_summaries(self):
        """
        Calls a some functions that return a single value for each
        streamline. Then it loops over all the streamline labelings 
        and summarizes these values for each region pair.
        """
        streamline_measures = {}
        # Calculate the measurements requested
        for func_name in self.measures:
            computation = function_registry[func_name]
            track_dset = scan.get_streamlines()
            if not scan.from_pkl: raise AttributeError("Must 'Process Inputs' first!")
            streamline_measures[measure_variable(func_name)] = computation(track_dset)
            
    
    def calculate_measurements(self):
        if self.calcluate_region_summaries:
            self.process_regions()
        self.calculate_streamline_summaries()
            
    def process_connectivity(self):
        """
        calculates surface area in mm2, region center of mass (in voxel coordinates)
        and region volume in mm3.
        """
        # First process every streamline in the dataset
        streamline_measures = self.get_streamline_summaries()
        
        self.data = {}
        from dsi2.volumes import load_lausanne_graphml, find_graphml_from_filename
        for label in self.scan.track_label_items:
            img_path = label.get_tracking_image_filename()
            node_data = load_lausanne_graphml(find_graphml_from_filename(img_path))
            prefix= matlab_variable(label)
            region_ints = np.array(node_data['regions'])
            region_names = np.array(node_data["region_labels"])
            self.data[prefix + "region_ids"] = region_ints
            self.data[prefix + "region_names"] = region_names
            
            # Process the rois used in this atlas            
            mask = MaskDataset(img_path, region_int_labels=region_ints,
                               region_names=region_names)
            for k,v in mask.get_stats():
                self.data[prefix + k] = v
                
            # process the simple streamline measurements
            connection_ids = label.load_array()
            # Make empty self.data for user-requested measurements
            self.data[prefix + "streamline_count"] =  np.zeros((len(region_ints),len(region_ints))) 
            for measure in streamline_measures.keys():
                self.data[prefix + measure + "_mean" ] = np.zeros((len(region_ints),len(region_ints)))
                self.data[prefix + measure + "_std" ] = np.zeros((len(region_ints),len(region_ints)))
                self.data[prefix + measure + "_median" ] = np.zeros((len(region_ints),len(region_ints)))
                self.data[prefix + measure + "_mad" ] = np.zeros((len(region_ints),len(region_ints)))
                
            # Make empty self.data for streamline scalars
            for scalar in streamline_scalars.keys():
                self.data[prefix + scalar + "_mean" ] = np.zeros((len(region_ints),len(region_ints)))
                self.data[prefix + scalar + "_std" ] = np.zeros((len(region_ints),len(region_ints)))
                self.data[prefix + scalar + "_median" ] = np.zeros((len(region_ints),len(region_ints)))
                self.data[prefix + scalar + "_mad" ] = np.zeros((len(region_ints),len(region_ints)))
                
             # Loop over all region pairs for this parcellation 
            for conn_id, (_i,_j) in node_data["index_to_region_pairs"].iteritems():
                i,j = _i-1, _j-1
                indexes = connection_ids == conn_id
                n_streamlines = indexes.sum()
                if n_streamlines == 0: continue
                self.data[prefix + "streamline_count"][i,j] = n_streamlines

                # extract user-requested measurements                
                for measure,values in streamline_measures.iteritems():
                    conn_values = values[indexes]
                    val_median = np.median(conn_values)
                    self.data[prefix + measure + "_mean" ] = conn_values.mean()
                    self.data[prefix + measure + "_std" ] = conn_values.std()
                    self.data[prefix + measure + "_median" ] = val_median
                    self.data[prefix + measure + "_mad" ] = np.median(np.abs(conn_values-val_median))
                    
                # extract streamline scalar means
                for scalar, values in streamline_scalars.iteritems():
                    conn_values = values[indexes]
                    val_median = np.median(conn_values)
                    self.data[prefix + scalar + "_mean" ] = conn_values.mean()
                    self.data[prefix + scalar + "_std" ] = conn_values.std()
                    self.data[prefix + scalar + "_median" ] = val_median
                    self.data[prefix + scalar + "_mad" ] = np.median(np.abs(conn_values-val_median))
                
                    
                            
                    
                                
                
                
                