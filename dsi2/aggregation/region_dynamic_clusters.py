#!/usr/bin/env python
import os
import numpy as np
from ..streamlines.track_dataset import RegionCluster, TrackDataset
from .cluster_ui import ClusterEditor, ClusterAdapter
from ..streamlines.track_math import tracks_to_endpoints
from ..database.track_datasource import TrackDataSource

import matplotlib.pyplot as plt

from dipy.tracking import metrics as tm
from dipy.tracking import distances as td

from traits.api import HasTraits, Instance, Array, Enum, \
    Str, File, on_trait_change, Bool, Dict, Range, Color, List, Int, \
    Property, Button, DelegatesTo, on_trait_change, Str, Tuple
from traitsui.api import View, Group, Item, RangeEditor, EnumEditor, OKButton, CancelButton
from ..streamlines.track_math import region_pair_dict_from_roi_list
import networkx as nx

#graphml_lookup = {
#                  "scale33":"resolution83",     "scale60":"resolution150",
#                  "scale125":"resolution258",   "scale250":"resolution500",
#                  "scale500":"resolution1015"
#                  }


from traitsui.editors.tabular_editor import TabularEditor
from traitsui.tabular_adapter import TabularAdapter


class TerminationPatternPlotter(HasTraits):
    pass

class RegionAggregator(ClusterEditor):
    # Maps roi integer to a name
    region_labels = Dict
    # Maps pairs of roi integers to an index
    region_pairs_to_index  = Dict
    index_to_region_pairs  = Dict
    regions = Array
    # will a connection be allowed if it appears in ANY or ALL subjects?
    across_subject_comparison_operation = Enum("Union","Intersection")
    parameters = ["min_tracks"]
    min_tracks = Range(low=0,high=100,value=1, auto_set=False,name="min_tracks",
                          desc="A cluster label must be assigned to at least this many tracks",
                          label="Minimum tracks",
                          parameter=True
                          )
    atlas_name       = Str("None",parameter=True)

    # Previously from "Atlas"
    possible_atlases = List
    atlas_parameters = Dict
    

    def _b_plot_connection_vector_fired(self):
        """Listen for button clicks"""
        self.plot_connection_vectors()

    def update_atlas(self):
        """
        Loads a numpy array from disk based on the string `atlas_name`.
        Parameters:
        -----------
        atlas_name:Str
          Must exist as a key in the TrackDataset.properties.atlases of each subject
        """
        print ""
        print "\t+ Updating Atlas"
        print "\t+ =============="
        if not len(self.track_sets): return
        applicable_properties = self.atlas_parameters[self.atlas_name].keys()
        query = {"name":self.atlas_name}
        for ap in applicable_properties:
            query[ap] = getattr(self, "%s_%s" % (self.atlas_name, ap))
        print "\t\t++ querying track_source for labels labels with", query
        new_labels = self.track_source.change_atlas(query)
        
        # Set the .connections attribute on each TrackDataset
        print "\t\t++ updating graphical objects"
        for tds,c in zip(self.track_sets,new_labels):
            tds.set_connections(c)
        print "\t+ Done"

    def clusterize(self, ttracks):
        """
        Operates **On a single TrackDataset**.
        1) track_dataset.connections are tabulated
        2) a RegionCluster is created for each connection_id
        Nothing is returned.
        """
        # Holds the cluster assignments for each track
        clusters = []
        label_id_map = {}

        if self.atlas_name == "None":
            print "Requires an atlas to be specified!!"
            return clusters, label_id_map

        labels = ttracks.connections
        # Populate the clusters list
        clustnum = 0

        for k in np.unique(labels):
            indices = np.flatnonzero(labels == k)
            ntracks = len(indices)
            if ntracks < self.min_tracks: continue
            clustnum += 1
            clusters.append(
                RegionCluster(
                     start_coordinate = "a",
                     end_coordinate =   "a",
                     ntracks = ntracks,
                     id_number = clustnum,
                     indices = indices,
                     connection_id = k,
                     scan_id = ttracks.scan_id
                )
            )
        return clusters


    def set_track_source(self,tsource):
        """
        Overwriting the 
        """
        self.track_source = tsource
        self.track_source.set_render_tracks(self.render_tracks)
        # The track source contains label data, NOTE the track_source will 
        # cache the label vectors for each subject
        self.atlas_parameters = self.track_source.load_label_data()
        
    @on_trait_change('+parameter')
    def clustering_param_changed(self,obj, name, old, new):
        print "\t+ %s parameter on clusterer changed" % name
        if not name in self.parameters:
            print "\t\t++ not in self.parameters"
            return
        if name.startswith(self.atlas_name):
            print "\t\t++ parameter is applicable to current atlas"
            self.update_atlas()
        else:
            print "\t\t++ param not applicable to current atlas"
        # Update clusters either way
        if self.auto_clusterize:
            self.update_clusters()
                
    def default_traits_view(self):
        """
        The editable traits in the algorithm_widgets will depend on the atlas
        and therefore we must build the ui on the fly
        """
        if self.track_source is None:
            raise ValueError("Must have a track_source set to determine traits")
        
        
        self.possible_atlases = self.atlas_parameters.keys()
        groups = []        
        # Loop over the available atlases
        for atlas_name in self.possible_atlases:
            # define a trait for each editable parameter
            group_items = []
            for editable_param, possible_values in \
                    self.atlas_parameters[atlas_name].iteritems():
                # use a modified version of the param name as a new trait
                new_trait_name = atlas_name + "_" + editable_param
                self.parameters.append(new_trait_name)
                self.add_trait(new_trait_name,Enum(possible_values,parameter=True))
                setattr(self,new_trait_name,possible_values[0])
                #self.on_trait_change(new_trait_name,self.atlas_parameter_trait_changed)
                group_items.append(
                    Item(new_trait_name, label=editable_param)
                )
            groups.append(
                Group(
                      *tuple(group_items),
                      visible_when="atlas_name=='%s'" % atlas_name,
                      show_border=True,
                      label=atlas_name+" parameters")
            )
        
        # widgets for editing algorithm parameters
        traits_view = View(
          # All cluster editors have these          
          Group( 
            Group(
              Item("auto_clusterize"),
              Item("render_clusters"),
              Item("render_tracks"),
            ),
            Group(
              Item(name="min_tracks",
                    editor=RangeEditor(mode="slider",
                    high = 100,low = 0,format = "%i")),
              # Specific to the dynamic atlas class
              Item("atlas_name", editor= EnumEditor(name="possible_atlases")),
              Group(*tuple(groups))
            ),
            Group(
              Item("clusters",
                    name='clusters',
                    editor=TabularEditor(
                             adapter=ClusterAdapter(),
                             editable=False),
              height=400, width=200, show_label=False),
              label="Clustering Options",
              show_border=True)
            )
          )
        return traits_view
