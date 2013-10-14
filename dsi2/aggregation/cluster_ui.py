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

track_dataset_source_table = TableEditor(
    columns =
    [   ObjectColumn(name="scan_id",editable=False),
        ObjectColumn(name="properties.study",editable=False),
        ObjectColumn(name="properties.scan_group",editable=False),
        ObjectColumn(name="properties.reconstruction",editable=False),
    ],
    auto_size  = True,
    edit_view="data_source_view"
    )

track_dataset_graphics_table = TableEditor(
    columns =
    [   ObjectColumn(name="scan_id",editable=False),
        ObjectColumn(name="properties.study",editable=False),
        ObjectColumn(name="properties.scan_group",editable=False),
        ObjectColumn(name="properties.reconstruction",editable=False),
    ],
    auto_size  = True,
    edit_view="graphics_view"
    )

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
            Item("render_clusters"),
            Item("render_tracks"),
                 ),
                 Include("algorithm_widgets"),
                     Group(Item("clusters",
                      name='clusters',
                      editor=TabularEditor(
                          adapter=ClusterAdapter(),
                          editable=False),
                      height=400, width=200, show_label=False),
                      label="Aggregation Options",
                      show_border=True)
                    )

class ClusterEditor(HasTraits):
    # Options for track aggregation
    auto_aggregate     = Bool(False)
    # XXX: this should be True for non-interactive use
    render_tracks       = Bool(False)
    tracks_drawn        = Bool(False)
    render_clusters     = Bool(False)
    clusters_drawn      = Bool(False)
    interactive         = Bool(False)
    clusters            = List(Instance(Cluster))
    parameters          = []

    # Data
    track_source = Instance(TrackDataSource)
    track_sets = List(Instance(TrackDataset))
    track_datasets = DelegatesTo("track_source")
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

    def query_track_source_with_coords(self,list_of_coord_tuples,downsample=0):
        """ User has changed the query coordinates.
        1) Query the datasource for new streamlines
        2) Send them to the aggregator for aggregation
        3) Disable mayavi rendering
        4) Remove previous streamlines from the engine
        5) Add new streamlines to the engine
           -- if we're aggregation, then paint the streamlines
        6) re-enable mayavi rendering
        """
        #print "+ Queried track sources with %i coordinates" % len(list_of_coord_tuples)
        #print "============================================"
        if len(self.track_source) == 0:
            print "\t+ No datasets in the track_source"
            return
        # Set the pre-filtered tracks
        if self.scene3d:
            #print "\t+ disabling rendering"
            self.scene3d.disable_render = True
        #print "\t+ creating ``track_sets`` from results ..."
        self.set_track_sets( # Note to self, this calls clear_tracks()
            self.track_source.query_ijk(list_of_coord_tuples,
                                        every=downsample))


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

    def repaint_streamlines(self):
        print "re-painting streamlines"
        self.scene3d.disable_render = True
        self.clear_clusters()
        self.update_clusters()
        self.scene3d.scene.disable_render = False

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
        self.track_source = tsource
        self.track_source.set_render_tracks(self.render_tracks)


    def _render_tracks_changed(self):
        """
        When render_tracks gets changed, the render_tracks property
        on the items in ``track_source`` gets changed so that future
        subsets of them will inherit it.

        Additionally, the visibility of already-existing streamlines
        will get toggled.
        """
        print "+ render_tracks changed to", self.render_tracks
        print "\t+ setting track_source's render_tracks attribute"
        self.track_source.set_render_tracks(self.render_tracks)
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

    def update_clusters(self):
        """ Creates new clusters when a aggregation parameter has been
        changed. Will CREATE MayaVi objects if ``self.interactive``
        and ``self.render_clusters`` are true.
        """

        print "+ Updating cluster assignments."
        _clusters = []
        self.label_lookup = []
        # If interactive, convert the tds to something that can render
        #if self.render_tracks:
        #    print "\t++ rendering tracks"
        #    self.draw_tracks()
        # Update the clusters in each TrackDataset through self.aggregate()
        # NOTE: Someday this could be parallelized
        for tnum, tds in enumerate(self.track_sets):
            print "\t+ cluster %d of %d" % (tnum+1, len(self.track_sets))
            clusts = self.aggregate(tds)
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

    # Provides access to
    objects_view = View(
        Group(
            Group(
                Item("track_datasets",
                      editor=track_dataset_source_table),
                orientation="horizontal",
                show_labels=False,
                show_border=True,
                label="Data Sources"
                ),
            Group(
                Item("track_sets", editor=track_dataset_graphics_table),
                orientation="horizontal",
                show_labels=False,
                show_border=True,
                label="Graphical Objects"
                ),
            orientation="vertical"
        ),
        resizable=True,
        width=900,
        height=500

    )


if __name__=="__main__":
    ce = ClusterEditor(scene3d=MlabSceneModel())
    ce.configure_traits(view="browser_view",
                        handler = AlgorithmParameterHandler() )
