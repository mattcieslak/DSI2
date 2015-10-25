#!/usr/bin/env python
import sys, json, os
from time import time
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File, Int, Bool, cached_property
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn, SetEditor
import numpy as np
from scipy.io.matlab import savemat
from dsi2.database.traited_query import Scan
from dsi2.volumes.mask_dataset import MaskDataset
from collections import defaultdict
from scipy.sparse import lil_matrix
"""
Connectivity self.data can calculate many things.  Functions work by returning
an array with one value per streamline
"""
def matlab_variable(label_item):
    return  "_".join(
            ["%s_%s" % (k,v) for k,v in label_item.parameters.iteritems()]) + "_"

def group_center_of_mass(coordinates):
    # Make empty matrices for centers of mass
    means = coordinates.mean(axis=0)
    stds = coordinates.std(axis=0)
    medians = np.median(coordinates,axis=0)
    mads = np.median(np.abs(coordinates-medians),axis=0)
    return {
              "cx_mean" :means[0],
              "cx_std" :stds[0],
              "cx_median":medians[0], 
              "cx_mad" :mads[0],
              "cy_mean" :means[1],
              "cy_std" :stds[1],
              "cy_median":medians[1], 
              "cy_mad" :mads[1],
              "cz_mean" :means[2],
              "cz_std" :stds[2],
              "cz_median":medians[2], 
              "cz_mad" :mads[2]
              }


def voxel_density(list_of_voxel_lists):
    voxels = defaultdict(int)
    for voxel_stream in list_of_voxel_lists:
        for voxel in voxel_stream:
            voxels[voxel] += 1
    densities = np.array([v for k,v in voxels.iteritems()])
    density_median = np.median(densities)    
    return {
        "density_mean":densities.mean(),
        "density_std":densities.std(),
        "density_median":density_median,
        "density_mad":np.median(np.abs(densities - density_median)),
        "unique_voxels":len(densities)
        }

def measure_variable(s):
    return s.replace(" ","_").replace("(","").replace(")","").lower()

class ConnectivityMatrixCalculator(HasTraits):
    measures = List(["Length (voxels)", "Winding", "Curvature","Center of Mass"],
                    editor = SetEditor(
        values = ["Length (voxels)", "Winding", "Curvature","Center of Mass"],
        left_column_title  = 'Available Measurements',
        right_column_title = 'Measurements to Calculate'))    
    save_file = File("")
    calcluate_region_summaries = Bool(True)
    scan = Instance(Scan)
    
    def __init__(self,voxelized_streamlines, streamline_properties,**traits):
        super(ConnectivityMatrixCalculator,self).__init__(**traits)
        # everything gets stored in data until it gets saved
        self.data = {}
        self.voxelized_streamlines = voxelized_streamlines
        self.streamline_properties = streamline_properties
        
    def get_streamline_summaries(self):
        """
        Calls a some functions that return a single value for each
        streamline. Then it loops over all the streamline labelings 
        and summarizes these values for each region pair.
        """
        return self.streamline_properties
    
    def calculate_measurements(self):
        if self.calcluate_region_summaries:
            self.process_regions()
        self.calculate_streamline_summaries()
        
    def process_connectivity(self):
        """
        calculates surface area in mm2, region center of mass (in voxel coordinates)
        and region volume in mm3.
        """
        if not self.scan.connectivity_matrix_path:
            raise AttributeError("No output path specified")
        
        if os.path.exists(self.scan.connectivity_matrix_path):
            print self.scan.connectivity_matrix_path, "exists - skipping"
            return
        
        # First process every streamline in the dataset
        streamline_measures = self.streamline_properties
        
        # Then calculate the set of voxels for each streamline
        print "getting voelized streamlines"
        voxel_sets = self.voxelized_streamlines
        print "done"
        
        # Load in the streamline scalars
        streamline_scalars = {}        
        for scalar in self.scan.track_scalar_items:
            streamline_scalars[scalar.name] = scalar.load_array()
        
        #self.scan.clearmem()
        from dsi2.volumes import load_lausanne_graphml, find_graphml_from_filename
        for label in self.scan.track_label_items:
            img_path = label.get_tracking_image_filename()
            print img_path
            node_data = load_lausanne_graphml(find_graphml_from_filename(img_path))
            prefix= matlab_variable(label)
            region_ints = np.array(node_data['regions'])
            region_names = np.array([node_data['region_labels'][str(_id)]['dn_name'] for _id in region_ints])
            self.data[prefix + "region_ids"] = region_ints
            self.data[prefix + "region_names"] = region_names
            
            # Process the rois used in this atlas            
            mask = MaskDataset(img_path, region_int_labels=region_ints,
                               region_names=region_names)
            print "calculating volume measurements"
            for k,v in mask.get_stats().iteritems():
                self.data[prefix + k] = v
                
            # process the simple streamline measurements
            connection_ids = label.load_array()
            
            # Make empty matrices for user-requested measurements
            self.data[prefix + "streamline_count"] =  lil_matrix((len(region_ints),len(region_ints))) 
            for measure in streamline_measures.keys():
                if measure == "center_of_mass": continue
                self.data[prefix + measure + "_mean" ] = lil_matrix((len(region_ints),len(region_ints)))
                self.data[prefix + measure + "_std" ] = lil_matrix((len(region_ints),len(region_ints)))
                self.data[prefix + measure + "_median" ] = lil_matrix((len(region_ints),len(region_ints)))
                self.data[prefix + measure + "_mad" ] = lil_matrix((len(region_ints),len(region_ints)))
                
            # Make empty matrices for streamline scalars
            for scalar in streamline_scalars.keys():
                self.data[prefix + scalar + "_mean" ] = lil_matrix((len(region_ints),len(region_ints)))
                self.data[prefix + scalar + "_std" ] = lil_matrix((len(region_ints),len(region_ints)))
                self.data[prefix + scalar + "_median" ] = lil_matrix((len(region_ints),len(region_ints)))
                self.data[prefix + scalar + "_mad" ] = lil_matrix((len(region_ints),len(region_ints)))
                
            # Make empty matrices for density
            self.data[prefix +  "density_mean" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "density_std" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "density_median" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "density_mad" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "unique_voxels" ] = lil_matrix((len(region_ints),len(region_ints)))
            
            # Make empty matrices for centers of mass
            self.data[prefix +  "cx_mean" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cx_std" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cx_median" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cx_mad" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cy_mean" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cy_std" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cy_median" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cy_mad" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cz_mean" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cz_std" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cz_median" ] = lil_matrix((len(region_ints),len(region_ints)))
            self.data[prefix +  "cz_mad" ] = lil_matrix((len(region_ints),len(region_ints)))
                
             # Loop over all region pairs for this parcellation 
            for (_i,_j), conn_id in node_data["region_pairs_to_index"].iteritems():
                i,j = _i-1, _j-1
                indexes = connection_ids == conn_id
                n_streamlines = indexes.sum()
                if n_streamlines == 0: continue
                self.data[prefix + "streamline_count"][i,j] = n_streamlines

                # extract user-requested measurements                
                for measure,values in streamline_measures.iteritems():
                    if measure == "center_of_mass": continue
                    conn_values = values[indexes]
                    val_median = np.median(conn_values)
                    self.data[prefix + measure + "_mean" ][i,j] = conn_values.mean()
                    self.data[prefix + measure + "_std" ][i,j] = conn_values.std()
                    self.data[prefix + measure + "_median" ][i,j] = val_median
                    self.data[prefix + measure + "_mad" ][i,j] = np.median(np.abs(conn_values-val_median))
                    
                # extract streamline scalar means
                for scalar, values in streamline_scalars.iteritems():
                    conn_values = values[indexes]
                    val_median = np.median(conn_values)
                    self.data[prefix + scalar + "_mean" ][i,j] = conn_values.mean()
                    self.data[prefix + scalar + "_std" ][i,j] = conn_values.std()
                    self.data[prefix + scalar + "_median" ][i,j] = val_median
                    self.data[prefix + scalar + "_mad" ][i,j] = np.median(np.abs(conn_values-val_median))
                    
                # get the center of mass
                centers = group_center_of_mass(streamline_measures['center_of_mass'][indexes])
                for measure, value in centers.iteritems():
                    self.data[prefix+measure][i,j] = value
                
                # add in the density results for this connection
                density_results = voxel_density([map(tuple,stream) for stream \
                                                 in voxel_sets[indexes]])
                for density_measure, density_value in density_results.iteritems():
                    self.data[prefix + density_measure][i,j] = density_value                    
                    
    def save(self):
        savemat(self.scan.connectivity_matrix_path, self.data)
