#!/usr/bin/env python
import os
import numpy as np
from ..streamlines.track_dataset import RegionCluster, TrackDataset
from .cluster_ui import ClusterEditor
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

graphml_lookup = {
                  "scale33":"resolution83",     "scale60":"resolution150",
                  "scale125":"resolution258",   "scale250":"resolution500",
                  "scale500":"resolution1015"
                  }




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
    # Buttons for the algorithm_widgets
    b_plot_connection_vector = Button(label="Connection Vectors")
    b_query_region_pairs = Button(label="Query a Region Pair")
    b_change_postproc = Button(label="Change PostProcessor")

    # Previously from "Atlas"
    possible_atlases = List
    atlas_graphml    = File
    connection_vector_plot_type = Enum("lines","imshow")

    def _b_plot_connection_vector_fired(self):
        """Listen for button clicks"""
        self.plot_connection_vectors()

    @on_trait_change("atlas_name")
    def set_atlas(self,atlas_name):
        """
        Loads a numpy array from disk based on the string `atlas_name`.
        Parameters:
        -----------
        atlas_name:Str
          Must exist as a key in the TrackDataset.properties.atlases of each subject

        Does:
        -----
        * Reads the graphml file from CMP corresponding to `atlas_name`
        * Builds lookup tables for connection_id -> (region1, region2)
        * Builds lookup tables for connection_id -> ("region1", region2)
        """
        if not len(self.track_sets): return
        self.atlas_name = atlas_name
        # only use the first part of the atlas name to get lausanne labels
        if not all([atlas_name in tds.properties.atlases.keys() for tds in self.track_sets]):
            print "WARNING: Not all TrackDatasets have ", atlas_name
            return
        
        # Set the .connections attribute on each TrackDataset
        for tds in self.track_sets:
            tds.load_connections(self.atlas_name)
        
        # =====================================================================================
        # IF there is a graphml, load it to get the region names
        self.atlas_graphml = self.track_sets[0].properties.atlases[atlas_name]["graphml_path"]
        if self.atlas_graphml:
            print "loading regions from", self.atlas_graphml
            graph = nx.read_graphml(self.atlas_graphml)
            for roi_id,roi_data in graph.nodes(data=True):
                self.region_labels[roi_id] = roi_data
            self.regions = np.array(sorted(map( int,graph.nodes() )))

            self.region_pairs_to_index = region_pair_dict_from_roi_list(self.regions)
            # Which regionpairs map to which unique id in this dataset?
            self.index_to_region_pairs = dict(
                [
                (idxnum,(self.region_labels[str(id1)]['dn_name'],
                           self.region_labels[str(id2)]['dn_name']) ) \
                 for (id1,id2), idxnum in self.region_pairs_to_index.iteritems()
                ]
            )
            self.region_pair_strings_to_index = dict([
                (value,key) for key,value in self.index_to_region_pairs.iteritems()
            ])
        #self.update_clusters()

    def get_region_pair_code(self,region1,region2):
        if (region1,region2) in self.region_pair_strings_to_index:
            return self.region_pair_strings_to_index[(region1,region2)]
        if (region2,region1) in self.region_pair_strings_to_index:
            return self.region_pair_strings_to_index[(region2,region1)]
        else:
            return None

    def update_clusters(self):
        """
        OVERRIDES THE BASE update_clusters so we can apply the
        postproc filter
        1) A first-pass "clusterize" is run over all the track_sets
        2) The conection-vector matrix is built using all subjects
        3) IF a post-processing is selected
           3a) Statistics for each connection_id are used to determine
               which connection_ids should be visualized
           3b) clusterize is run again to collect stats for each dataset
               AFTER they've been subset to contain only selected
               connection_ids
        """
        ## First-pass: collect all the termination patterns
        _clusters = []
        self.label_lookup = []
        # If we're not supposed to query tracks, clear the clusters and do nothing
        if not self.render_tracks:
            self.clusters = _clusters
            return

        # First-pass clustering
        for tds in self.track_sets:
            clusts, connection_id_map = self.clusterize(tds)
            self.label_lookup.append(connection_id_map)
            tds.set_clusters(clusts, 
                             update_glyphs=(self.filter_operation=="None"))
            _clusters += tds.clusters # collect the colorized version
        self.clusters = _clusters
        self.pre_filter_connections, self.pre_filter_matrix = self.connection_vector_matrix()
        if self.filter_operation == "None":
            self.post_filter_matrix, self.post_filter_connections,= \
                self.pre_filter_matrix, self.pre_filter_connections
            print "%"*5, "No Postprocessing Filter"
            return

        # A post-processing filter is to be applied
        print "%"*5, "Applying", self.filter_operation, "filter"
        OK_regions = self.post_processor.filter_clusters(
                  ( self.pre_filter_connections,self.pre_filter_matrix)  )
        # Second pass: subset the track_sets so that only OK regions
        #              are plotted
        filt_tracks = []
        _clusters = []
        self.label_lookup = []
        for tds in self.track_sets:
            ftds = tds.subset(tds.get_tracks_by_connection_id(OK_regions))
            clusts, connection_id_map = self.clusterize(ftds)
            self.label_lookup.append(connection_id_map)
            ftds.set_clusters(clusts)
            _clusters += ftds.clusters # collect the colorized version
            filt_tracks.append(ftds)
        self.clusters = _clusters
        self.set_track_sets(filt_tracks)
        self.post_filter_connections, self.post_filter_matrix = \
                                   self.connection_vector_matrix()

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
            try:
                start_region, end_region = self.index_to_region_pairs[k]
            except KeyError:
                print "Region clusterizer encountered an unexpected connection id: ", k
                start_region, end_region = "undefined","undefined"
            clustnum += 1
            clusters.append(
                RegionCluster(
                     start_coordinate = start_region,
                     end_coordinate =   end_region,
                     ntracks = ntracks,
                     id_number = clustnum,
                     indices = indices,
                     connection_id = k,
                     scan_id = ttracks.scan_id
                )
            )
            label_id_map[k] = clustnum-1

        return clusters, label_id_map

    def found_connections(self):
        """ Returns a list of connections found by the tracks considered """
        found_conns = [set([cl.connection_id for cl in ts.clusters]) for ts in self.track_sets]

        connection_ids = []
        if len(found_conns) == 0:
            print "Found no connections"
            return

        connection_ids = set([])
        for conns in found_conns:
            if self.across_subject_comparison_operation == "Union":
                connection_ids.update(conns)
            elif self.across_subject_comparison_operation == "Intersection":
                connection_ids.intersection_update(conns)
        return sorted(list(connection_ids))

    def connection_vector_matrix(self,n_top=0):
        """
        Parameters
        ----------
        n_top:int
          only return the `n_top` highest ranked connections

        Returns
        -------
        observed_connections:list
          the results of the self.found_connections()
        connection_vectors:np.ndarray
          one row per dataset, one column per region pair
        """
        observed_connections = self.found_connections()
        connection_vectors = np.zeros((len(self.track_sets),
                                      len(observed_connections)))
        row_labels = []
        for tds_num,(tds,lut) in enumerate(zip(self.track_sets,self.label_lookup)):
            row_labels.append(tds.properties.scan_id)
            for nconn, connection in enumerate(observed_connections):
                idx = lut.get(connection,-999)
                if idx < 0: continue
                connection_vectors[tds_num,nconn] = tds.clusters[idx].ntracks
        if n_top > 0:
            top_indices = np.flatnonzero(( connection_vectors.shape[1] -1 \
                            - connection_vectors.argsort().argsort().sum(0).argsort().argsort()) \
                           < n_top )

            return [observed_connections[n] for n in top_indices], connection_vectors[:,top_indices]
        return observed_connections, connection_vectors


    def plot_connection_vectors(self):
        # Which data should we use?
        if self.post_filter_connections.size == 0:
            matrix = self.pre_filter_matrix
            labels = self.pre_filter_connections
        else:
            matrix = self.post_filter_matrix
            labels = self.post_filter_connections

        # Pad dimensions
        if matrix.ndim == 1:
            matrix.shape = (1,matrix.shape[0])

        if self.connection_vector_plot_type == "imshow":
            self.plot_connection_vector_tiles(matrix, labels)

        elif self.connection_vector_plot_type == "lines":
            self.plot_connection_vector_lines(matrix, labels)

    def plot_connection_vector_lines(self,matrix,labels):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        static_colors = all(
            [not s.properties.dynamic_color_clusters for s in self.track_sets])
        # Is each TrackDataset assigned a specific color?
        if static_colors:
            print "plotting using static colors"
            for tds, vec in zip(self.track_sets,self.post_filter_matrix):
                ax.plot( vec, label=tds.properties.scan_id,
                         color=[x/255. for x in tds.properties.static_color[:3]],
                         linewidth=4)
        else:
            for tds, vec in zip(self.track_sets, self.post_filter_matrix):
                ax.plot(vec,label=tds.properties.scan_id,linewidth=4)
        ax.legend()
        ax.set_xticks(np.arange(self.post_filter_matrix.shape[1]))
        ax.set_title("Observed Connections")
        ax.set_ylabel("Streamline Count")

        ax.set_xticklabels(
            ["(%s, %s)" % self.index_to_region_pairs.get(conn,("no-label","no-label")) \
             for conn in self.post_filter_connections ] )
        plt.xticks(rotation=90,size=12)
        plt.subplots_adjust(bottom=0.55)
        fig.show()
        return fig

    def plot_connection_vector_tiles(self,matrix,labels):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(cv,interpolation="nearest", aspect="auto")
        ax.set_yticks(np.arange(cv.shape[0]))
        ax.set_yticklabels([tds.properties.scan_id for tds in self.track_sets])

        ax.set_xticks(np.arange(cv.shape[1]))
        ax.set_title("Observed Connections")
        ax.set_ylabel("Streamline Count")

        ax.set_xticklabels(
            ["(%s, %s)" % self.index_to_region_pairs[conn] \
             for conn in connections ] )
        plt.xticks(rotation=90,size=12)
        plt.subplots_adjust(bottom=0.45)
        fig.show()
        return fig
    
    def get_R_dat(self,subject_ids,roi_name):
        """
        Saves a tab-delimited text file of the Termination Patterns found.
        Parameters
        ----------
        subject_ids:list
          subject names corresponding to the rows returned by `self.connection_vector_matrix()`
        roi_name:str
          name of the ROI to be saved in the ROI column of the dat file

        Returns
        -------
        results:list
          a string row for each subject
        """
        conn_ids, cvec = self.connection_vector_matrix()
        region_pairs = ["%s.%s" % self.index_to_region_pairs[conn] \
             for conn in conn_ids ]
        header = "\t".join(
            ["subject","roi", "region.pair", "count"])
        results = [ header ]
        for subject_id,subject_data in zip(subject_ids,cvec):
            for pair_id, pair_count in zip(region_pairs,subject_data):
                results.append( "\t".join(
                    ['"s%s"'%subject_id, roi_name, pair_id, "%.2f"%pair_count ]
                    ))
        return results
    
    # widgets for editing algorithm parameters
    algorithm_widgets = Group(
                             Item(name="min_tracks",
                                    editor=RangeEditor(mode="slider",
                                    high = 100,low = 0,format = "%i")),
                             Item("atlas_name", editor= EnumEditor(name="possible_atlases")),

                         Group(
                             Item("post_processor", style="simple"),
                             Item(name="b_plot_connection_vector"),
                             show_border=True,
                             show_labels=False
                         )
                        )



    def query_region_pair(self):
        """ 
        Opens a little GUI where you can select two regions.
        Each TrackDataset is subset so that only streamlines connecting
        that region-pair are visible.
        """
        if len(self.track_source) == 0: return
        # launch the ui and stop everything else
        ui = self.region_pair_query.edit_traits()
        if not ui.result: 
            print "canceled, exiting"
            return
        self.save_name = "%s__to__%s" % (
            self.region_pair_query.region1,
            self.region_pair_query.region2)
        region_id = self.clusterer.get_region_pair_code(
            self.region_pair_query.region1,
            self.region_pair_query.region2)
        if region_id is None:
            print "region pair not found"
        else:
            self.render_region_pairs(region_id)
        
    def render_region_pairs(self,region_id):
        self.clusterer.clear_clusters()
        self.clusterer.set_track_sets(
            self.track_source.query_connection_id(region_id,
                                                  every=self.downsample))
        # --- Rendering ---
        self.scene3d.disable_render = True
        self.clear_track_glyphs()
        self.clusterer.update_clusters()
        self.clusterer.draw_tracks()
        
        
class RegionPair(HasTraits):
    possible_regions = List
    region1 = Str
    region2 = Str
    selected_connection = Int

    def update_regions(self, clusterer):
        """Updates ``self.possible_regions`` based on the clusterer and
        sets an arbitrary region pair
        """
        print "updating possible regions"
        self.possible_regions = sorted([
            clusterer.region_labels[str(id1)]['dn_name'] \
             for id1 in clusterer.regions ])
        if not len(self.possible_regions): return
        self.region1 = self.possible_regions[0]
        self.region2 = self.possible_regions[0]


    traits_view = View(Group(
            Item("region1",editor=EnumEditor(name="possible_regions")),
            Item("region2",editor=EnumEditor(name="possible_regions")),
            orientation="vertical"
            ),
            kind="modal",
            buttons=[OKButton,CancelButton]
            )

class RegionPairs(HasTraits):
    region_pairs = List(Instance(RegionPair))
