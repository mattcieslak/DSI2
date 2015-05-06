#!/usr/bin/env python
import numpy as np
# Traits stuff
from traits.api import HasTraits, Instance, Array, on_trait_change, \
    Bool, Dict, Range, Color, List, Int, Property, Any, Function, DelegatesTo, Str
from traitsui.api import View, Item, VGroup, \
     HGroup, Group, RangeEditor, TableEditor, Handler, Include,HSplit, \
     CheckListEditor, ObjectColumn
from traitsui.group import ShadowGroup

from mayavi.core.ui.api import SceneEditor
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk

# Needed for Tabular adapter
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel

# Needs a TrackDataset to manipulate
from ..streamlines.track_dataset import TrackDataset, Cluster, RegionCluster
from .across_subject_postproc import AcrossSubjectPostproc
from ..database.track_datasource import TrackDataSource



class ClusterAdapter( TabularAdapter ):
    """
    Used to display the clusters as colored rows in a tabular editor
    """
    Cluster_bg_color = Property
    columns = [
        ("Data Source", "scan_id"),
        ("Count",       "ntracks"),
        ("ID",          "id_number"),
        ("Start",       "start_coordinate"),
        ("End",         "end_coordinate")
    ]
    font = "Courier 10"
    def __init__(self,**traits):
        super(ClusterAdapter,self).__init__(**traits)

    def _get_Cluster_bg_color(self):
        col = self.item.color
        return col


class AlgorithmParameterHandler(Handler):
    def setattr(self,info,obj,name,value):
        print "name:", name
        print "value:", value

cluster_editor_group = \
        VGroup( Group(
            Item("auto_aggregate"),
            #Item("render_clusters"),
            Item("render_tracks"),
                 ),
                 Include("algorithm_widgets"),
                     Group(Item("clusters",
                      name='clusters',
                      editor=TabularEditor(
                          adapter=ClusterAdapter(),
                          editable=False),
                      height=400, width=200, show_label=False),
                      label="Clusters/Segments List",
                      show_border=True)
                    )

class ClusterEditor(HasTraits):
    # Options for track aggregation
    auto_aggregate     = Bool(False)
    compute_prototypes = Bool(False)
    # XXX: this should be True for non-interactive use
    render_tracks       = Bool(False)
    tracks_drawn        = Bool(False)
    render_clusters     = Bool(False)
    clusters_drawn      = Bool(False)
    interactive         = Bool(False)
    clusters            = List(Instance(Cluster))
    parameters          = []

    # Data
    track_sets = List(Instance(TrackDataset))
    pre_filter_matrix = Array
    pre_filter_connections = Array
    post_filter_matrix = Array
    post_filter_connections = Array

    # 3d plotting
    scene3d = Instance(MlabSceneModel)

    # Collect labels?
    label_lookup = List()

    # static_color from browserbuilder
    subject_colors = List()

    # post-processing algorithm, defaults to "None"
    post_processor = Instance(AcrossSubjectPostproc,())
    filter_operation = DelegatesTo("post_processor")

    # local ui
    cluster_editor_group = cluster_editor_group

    # convenience for saving tracks or screenshots after a search
    save_name = Str

    def __init__(self, **traits):
        """ Creates a panel for editing cluster assignments.
        """
        super(ClusterEditor,self).__init__(**traits)
        self.cluster_editor_group = cluster_editor_group

    @on_trait_change('+parameter')
    def aggregation_param_changed(self,obj, name, old, new):
        print name, "parameter on aggregator changed"
        if name in self.parameters and self.auto_aggregate:
            self.update_clusters()

    def _auto_aggregate_changed(self):
        print "+ automatic aggregation changed:", self.auto_aggregate
        if self.auto_aggregate:
            self.update_clusters()

    def set_track_sets(self,tsets):
        """
        The entry point for streamline data to the aggregator.
        Some of the logic implemented here depends on whether a
        postprocessor will be used
        """
        # Remove the old track_sets
        self.clear_tracks()

        # set the new track_sets
        self.track_sets = tsets
        if self.render_tracks:
            self.draw_tracks()

    def set_track_source(self,tsource):
        pass


    def _render_tracks_changed(self):
        """
        When render_tracks gets changed, the render_tracks property
        on the items in ``track_source`` gets changed so that future
        subsets of them will inherit it.

        Additionally, the visibility of already-existing streamlines
        will get toggled.
        """
        # Apply the new "render_tracks" to 
        for tds in self.track_sets:
            tds.render_tracks = self.render_tracks
            
        print "+ render_tracks changed to", self.render_tracks
        #print "\t+ setting track_source's render_tracks attribute"
        #self.track_source.set_render_tracks(self.render_tracks)
        self.scene3d.disable_render = True
        # Tracks are drawn
        if self.tracks_drawn:
            print "\t+ toggling glyph visibility on previously drawn tracks"
            for tds in self.track_sets:
                tds.set_track_visibility(self.render_tracks)
        else:
            if self.render_tracks:
                self.draw_tracks()
        self.scene3d.disable_render = False
        print "+ Done."

    def _render_clusters_changed(self):
        print "+ render_clusters changed to", self.render_clusters
        self.scene3d.disable_render= True
        for tds in self.track_sets:
            tds.set_cluster_visibility(self.render_clusters)
        self.scene3d.disable_render = False

    #def update_clusters(self):
        #""" Creates new clusters when a aggregation parameter has been
        #changed. Will CREATE MayaVi objects if ``self.interactive``
        #and ``self.render_clusters`` are true.
        #"""

        #print "+ Updating cluster assignments."
        #_clusters = []
        #self.label_lookup = []
        ## If interactive, convert the tds to something that can render
        ##if self.render_tracks:
        ##    print "\t++ rendering tracks"
        ##    self.draw_tracks()
        ## Update the clusters in each TrackDataset through self.aggregate()
        ## NOTE: Someday this could be parallelized
        #for tnum, tds in enumerate(self.track_sets):
            #print "\t+ cluster %d of %d" % (tnum+1, len(self.track_sets))
            #clusts = self.aggregate(tds)
            #tds.set_clusters(clusts)
            #_clusters += tds.clusters # collect the colorized version
        #self.clusters = _clusters
        ## Take care of the graphics
        #if self.render_clusters:
            #print "\t++ rendering tracks"
            #self.draw_clusters()
        #print "+ Aggregation Complete"

    def update_clusters(self):
        """ Creates new clusters when a aggregation parameter has been
        changed. Will CREATE MayaVi objects if ``self.interactive``
        and ``self.render_clusters`` are true.
        """

        print "+ Updating cluster assignments."
        _clusters = []
        self.label_lookup = []
        for tnum, tds in enumerate(self.track_sets):
            labels = self.aggregate(tds)
            tds.labels = labels
            clusts = []
            # -----------------------------------------------------
            # Convert the labels array to a list of Cluster objects
            labels, occurrences = np.unique(labels,return_inverse=True)
            for labelnum, label in enumerate(labels):
                indices = np.flatnonzero(occurrences==labelnum)
                if self.compute_prototypes:
                    prototype = self.make_prototype(tds.tracks[indices])
                else: prototype = None
                
                clusts.append(
                    Cluster(
                        ntracks = len(indices),
                        id_number = label,
                        indices=indices,
                        scan_id = tds.scan_id
                        )
                    )
            # This grabs the colors from the rendered streamlines
            tds.set_clusters(clusts)
            _clusters += tds.clusters # collect the colorized version
        self.clusters = _clusters
        # Take care of the graphics
        if self.render_clusters:
            print "\t++ rendering tracks"
            self.draw_clusters()
        print "+ Aggregation Complete"
        
        
    ## Must be overwritten by a subclass
    def aggregate(self,track_datasets):
        raise NotImplementedError()
    
    def make_prototype(self,tracks):
        raise NotImplementedError()

    def draw_tracks(self):
        print "+ called draw_tracks"
        self.scene3d.disable_render = True
        if not self.render_tracks: return
        if self.tracks_drawn:
            print "\t+ Clearing pre-existing tracks"
            self.clear_tracks()

        print "\t+ Drawing the tracks"
        for tds in self.track_sets:
            tds.draw_tracks()
        self.tracks_drawn = True
        self.scene3d.disable_render = False
        print "++ Done"
        
    def add_tracks(self,tds):
        """ Adds a TrackDataset to the working set """
        tds.render_tracks = self.render_tracks
        self.track_sets.append(tds)
        if self.render_tracks:
            self.scene3d.disable_render = True
            self.track_sets[-1].draw_tracks()
            self.tracks_drawn = True        
            self.scene3d.disable_render = False
        if self.render_clusters:
            self.scene3d.disable_render = True
            self.track_sets[-1].draw_clusters()
            self.tracks_drawn = True        
            self.scene3d.disable_render = False

    def draw_clusters(self):
        if not self.render_clusters: return
        print "+ Drawing Clusters"
        self.scene3d.disable_render = True
        for tds in self.track_sets:
            tds.draw_clusters()
        self.scene3d.disable_render = False

    def clear_tracks(self):
        for trk in self.track_sets:
            trk.remove_glyphs()
        self.tracks_drawn = False

    def clear_clusters(self):
        pass



    cluster_table = TabularEditor(adapter=ClusterAdapter(),
                                  editable=False)

    cluster_rows = Group(Item(name='clusters', editor=cluster_table,
                                height=400, show_label=False),
                     label="Aggregation Options", show_border=True)
    algorithm_widgets = Group()

    def default_traits_view(self):
        return View( self.cluster_editor_group )

    browser_view = View(HSplit(
                     Item('scene3d',
                         editor=SceneEditor(scene_class=Scene),
                         height=500, width=500),
                     cluster_editor_group,
                     show_labels=False
                    ),
                )



if __name__=="__main__":
    ce = ClusterEditor(scene3d=MlabSceneModel())
    ce.configure_traits(view="browser_view",
                        handler = AlgorithmParameterHandler() )
