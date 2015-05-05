#!/usr/bin/env python
import os
import numpy as np
from ..streamlines.track_dataset import RegionCluster, TrackDataset
from .cluster_ui import ClusterEditor, ClusterAdapter
from ..streamlines.track_math import tracks_to_endpoints
from ..database.track_datasource import TrackDataSource
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.tabular_adapter import TabularAdapter
import matplotlib.pyplot as plt
from dsi2.config import dsi2_data_path

from dipy.tracking import metrics as tm
from dipy.tracking import distances as td

from traits.api import HasTraits, Instance, Array, Enum, \
    Str, File, on_trait_change, Bool, Dict, Range, Color, List, Int, \
    Property, Button, DelegatesTo, on_trait_change, Str, Tuple
from traitsui.api import View, Group, Item, RangeEditor, EnumEditor, OKButton, CancelButton
from ..streamlines.track_math import region_pair_dict_from_roi_list
import networkx as nx

lausanne_scale_lookup = {
                  33:os.path.join(dsi2_data_path,
                    "lausanne2008", "resolution83", "resolution83.graphml"),
                  60:os.path.join(dsi2_data_path,
                    "lausanne2008", "resolution150", "resolution150.graphml"),
                  125:os.path.join(dsi2_data_path,
                    "lausanne2008", "resolution258", "resolution258.graphml"),
                  250:os.path.join(dsi2_data_path,
                    "lausanne2008", "resolution500", "resolution500.graphml"),
                  500:os.path.join(dsi2_data_path,
                    "lausanne2008", "resolution1015", "resolution1015.graphml")
                  }

class RegionPair(HasTraits):
    possible_regions = List
    region1 = Str
    region2 = Str
    selected_connection = Int

    def update_regions(self, aggregator):
        """Updates ``self.possible_regions`` based on the aggregator and
        sets an arbitrary region pair
        """
        print "updating possible regions"
        self.possible_regions = sorted([
            aggregator.region_labels[str(id1)]['dn_name'] \
             for id1 in aggregator.regions ])
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

class RegionLabelAggregator(ClusterEditor):
    track_source = Instance(TrackDataSource)
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
    # Buttons for the algorithm_widgets
    b_plot_connection_vector = Button(label="Connection Vectors")
    connection_vector_plot_type = Enum("lines","imshow")

    # Tools for querying a pair of regions from the atlas
    region_pair_query = Instance(RegionPair)
    b_query_region_pairs = Button(label="Query a Region Pair")
    b_change_postproc = Button(label="Change PostProcessor")

    # Graphml for the current streamline labels
    atlas_graphml    = File
    graphml_cache    = Dict

    def set_track_source(self,tsource):
        """
        Overwriting the
        """
        self.track_source = tsource
        self.track_source.set_render_tracks(self.render_tracks)
        # The track source contains label data, NOTE the track_source will
        # cache the label vectors for each subject
        self.atlas_parameters = self.track_source.load_label_data()
        
    def _b_plot_connection_vector_fired(self):
        """Listen for button clicks"""
        self.plot_connection_vectors()

    def _region_pair_query_default(self):
        return RegionPair()

    def _b_query_region_pairs_fired(self):
        """
        """
        self.select_region_pair()

    def update_atlas(self):
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
        * sets the queryable region labels in region_pair_query
        """
        print ""
        print "\t+ Updating Atlas"
        print "\t+ =============="

        # the atlas is not "None"
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

        # =====================================================================================
        # IF there is a graphml, load it to get the region names
        # only use the first part of the atlas name to get lausanne labels

        if not self.Lausanne2008_scale in self.graphml_cache:
            self.graphml_cache[self.Lausanne2008_scale] = self._load_graphml(self.Lausanne2008_scale)
        self.region_labels = self.graphml_cache[self.Lausanne2008_scale]['region_labels']
        self.region_pairs_to_index = self.graphml_cache[self.Lausanne2008_scale]['region_pairs_to_index']
        self.regions = self.graphml_cache[self.Lausanne2008_scale]['regions']
        self.index_to_region_pairs = self.graphml_cache[self.Lausanne2008_scale]['index_to_region_pairs']
        self.region_pair_strings_to_index = self.graphml_cache[self.Lausanne2008_scale]['region_pair_strings_to_index']
        print "\t\t *** updating options for the region-pair query"
        self.region_pair_query.update_regions(self)

        print "\t++ region labels updated"

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
        if self.auto_aggregate:
            self.update_clusters()
            
    def _load_graphml(self,scale_number):
        atlas_graphml = lausanne_scale_lookup[scale_number]
        print "\t\t+ loading regions from", atlas_graphml
        graph = nx.read_graphml(atlas_graphml)
        graphml_data = {'region_labels':{},
                        'region_pairs_to_index':{},
                        'index_to_region_pairs':{},
                        'region_pair_strings_to_index':{},
                        'regions':[]}
        for roi_id,roi_data in graph.nodes(data=True):
            graphml_data['region_labels'][roi_id] = roi_data
        graphml_data['regions'] = np.array(sorted(map( int,graph.nodes() )))

        graphml_data['region_pairs_to_index'] = region_pair_dict_from_roi_list(
                                                           graphml_data['regions'])
        
        # Which regionpairs map to which unique id in this dataset?
        graphml_data['index_to_region_pairs'] = dict(
            [
            (idxnum,(graphml_data['region_labels'][str(id1)]['dn_name'],
                       graphml_data['region_labels'][str(id2)]['dn_name']) ) \
             for (id1,id2), idxnum in graphml_data['region_pairs_to_index'].iteritems()
            ]
        )
        graphml_data['region_pair_strings_to_index'] = dict([
            (value,key) for key,value in graphml_data['index_to_region_pairs'].iteritems()
        ])
        return graphml_data

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
        1) A first-pass "aggregate" is run over all the track_sets
        2) The conection-vector matrix is built using all subjects
        3) IF a post-processing is selected
           3a) Statistics for each connection_id are used to determine
               which connection_ids should be visualized
           3b) aggregate is run again to collect stats for each dataset
               AFTER they've been subset to contain only selected
               connection_ids
        """
        ## First-pass: collect all the termination patterns
        _clusters = []
        self.label_lookup = []

        if not len(self.track_sets):
            print "\t+ No query results to apply aggregation to"
            return
        # First-pass aggregation
        for tds in self.track_sets:
            clusts, connection_id_map = self.aggregate(tds)
            self.label_lookup.append(connection_id_map)
            tds.set_clusters(clusts,
                             update_glyphs=(self.filter_operation=="None"))
            tds.labels = clusts
            _clusters += tds.clusters # collect the colorized version
        self.clusters = _clusters
        self.pre_filter_connections, self.pre_filter_matrix = self.connection_vector_matrix()
        if self.filter_operation == "None":
            self.post_filter_matrix, self.post_filter_connections,= \
                self.pre_filter_matrix, self.pre_filter_connections
            #print "%"*5, "No Postprocessing Filter"
            return

        # A post-processing filter is to be applied
        #print "%"*5, "Applying", self.filter_operation, "filter"
        OK_regions = self.post_processor.filter_clusters(
                  ( self.pre_filter_connections,self.pre_filter_matrix)  )
        # Second pass: subset the track_sets so that only OK regions
        #              are plotted
        filt_tracks = []
        _clusters = []
        self.label_lookup = []
        for tds in self.track_sets:
            ftds = tds.subset(tds.get_tracks_by_connection_id(OK_regions))
            clusts, connection_id_map = self.aggregate(ftds)
            self.label_lookup.append(connection_id_map)
            ftds.set_clusters(clusts)
            _clusters += ftds.clusters # collect the colorized version
            filt_tracks.append(ftds)
        self.clusters = _clusters
        self.set_track_sets(filt_tracks)
        self.post_filter_connections, self.post_filter_matrix = \
                                   self.connection_vector_matrix()
        
            
    def aggregate(self, ttracks):
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
                print "Region aggregator encountered an unexpected connection id: ", k
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
    
    def query_track_source_with_region_pair(self, region_id):                                                         
        """ User has quered a coordinates.
        1) Query the datasource for new streamlines
        2) Send them to the aggregator for aggregation                                                                
        3) Disable mayavi rendering
        4) Remove previous streamlines from the engine                                                                
        5) Add new streamlines to the engine
           -- if we're aggregation, then paint the streamlines                                                        
        6) re-enable mayavi rendering                                                                                 
        """
        if len(self.track_source) == 0:
            print "\t+ No datasets in the track_source"                                                               
            return
        # Set the pre-filtered tracks                                                                                 
        if self.scene3d:
            #print "\t+ disabling rendering"
            self.scene3d.disable_render = True
        #print "\t+ creating ``track_sets`` from results ..."                                                         
        self.set_track_sets(
            self.track_source.query_connection_id(region_id))
                                                 # every=self.downsample))                                            
        
        # Apply aggregation to the new ``track_sets`` if requested                                                    
        if self.auto_aggregate:
            #print "\t++ Applying aggregation to them ..."                                                            
            self.update_clusters()
        # Render their glyphs if the user wants                                                                       
        if self.render_tracks:
            #print "\t++ Rendering the new tracks."                                                                   
            self.draw_tracks()                                                                                        
        #print "\t++ Done"                                                                                            
        if self.scene3d:
            self.scene3d.disable_render = False                                                                       
            print "\t+ Re-enabling rendering"   
            
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

    def get_R_dat(self):
        """
        Creates a csv 
        """
        conn_ids, cvec = self.connection_vector_matrix()
        # if only a single subject
        if cvec.ndim < 2:
            cvec = cvec.reshape(1,-1)
        out = [ "subject, regionA, regionB, streamline_count" ]
        for subj, arr in zip(self.track_source.get_subjects(),cvec):
            for sl_count, reg_pair in zip(arr,conn_ids):
                regA, regB = self.index_to_region_pairs.get(
                    reg_pair, ("unlabeled","unlabeled"))
                out.append("%s, %s, %s, %d" % (subj, regA, regB, sl_count) )
        return "\n".join(out)

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



    def select_region_pair(self):
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
        region_id = self.get_region_pair_code(
            self.region_pair_query.region1,
            self.region_pair_query.region2)
        if region_id is None:
            print "region pair not found"
        else:
            self.query_track_source_with_region_pair(region_id)

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
              Item("auto_aggregate"),
              Item("render_clusters"),
              Item("render_tracks"),
            ),
            Group(
              Item(name="min_tracks",
                    editor=RangeEditor(mode="slider",
                    high = 100,low = 0,format = "%i")),
              # Specific to the dynamic atlas class
              Item("atlas_name", editor= EnumEditor(name="possible_atlases")),
              Group(*tuple(groups)),
              # Actions for altering internals
              Item("b_plot_connection_vector"),
              Item("b_query_region_pairs"),
              Item("b_change_postproc")
            ),
            Group(
              Item("clusters",
                    name='clusters',
                    editor=TabularEditor(
                             adapter=ClusterAdapter(),
                             editable=False),
              height=400, width=200, show_label=False),
              label="Aggregation Options",
              show_border=True)
            )
          )
        return traits_view


