#!/usr/bin/env python
import numpy as np
# Traits stuff
from traits.api import HasTraits, Instance, Array, on_trait_change, \
    Bool, Dict, Range, Color, List, Int, Property, Any, Function, DelegatesTo, Str
from traitsui.api import View, Item, VGroup, \
     HGroup, Group, RangeEditor, TableEditor, Handler, Include,HSplit, \
     CheckListEditor, ObjectColumn

from mayavi.core.ui.api import SceneEditor
from tvtk.pyface.scene import Scene

# Needed for Tabular adapter
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from mayavi.tools.mlab_scene_model import MlabSceneModel

# Needs a TrackDataset to manipulate
from ...streamlines.track_dataset import TrackDataset, Segment

from collections import defaultdict
def summarize_segments(segs):
    nsegs = defaultdict(int)
    streamlines = defaultdict(list)
    
    for ntrk, trk in enumerate(segs):
        seg_ids = np.unique(trk)
        for seg_id in seg_ids:
            streamlines[seg_id].append( ntrk )
            nsegs[seg_id] += np.sum(trk == seg_id)
    return nsegs, streamlines
    

class SegmentAdapter( TabularAdapter ):
    """
    Used to display the clusters as colored rows in a tabular editor
    """
    Segment_bg_color = Property
    columns = [
        ("Data Source", "scan_id"),
        ("segment ID",  "segment_id"),
        ("N Tracks",    "ntracks"),
        ("N Segs",  "ncoords"),
    ]
    font = "Courier 10"
    def __init__(self,**traits):
        super(SegmentAdapter,self).__init__(**traits)

    def _get_Segment_bg_color(self):
        col = self.item.color
        return col
    
from ..cluster_ui import ClusterEditor, ClusterAdapter

segment_editor_group = \
        VGroup( Group(
            Item("auto_aggregate"),
            Item("render_segments"),
            Item("render_tracks"),
                 ),
                 Include("algorithm_widgets"),
                     Group(Item("segments",
                      name='segments',
                      editor=TabularEditor(
                          adapter=SegmentAdapter(),
                          editable=False),
                      height=400, width=200, show_label=False),
                      label="Segmentation Options",
                      show_border=True)
                    )

class SegmentationEditor(ClusterEditor):
    # Options for track aggregation
    render_segments     = Bool(False)
    segments_drawn      = Bool(False)
    segments            = List(Instance(Segment))
    parameters          = []

    # Collect labels?
    segment_lookup = List()

    # local ui
    segment_editor_group = segment_editor_group

    def __init__(self, **traits):
        """ Creates a panel for editing cluster assignments.
        """
        super(SegmentationEditor,self).__init__(**traits)
        self.segment_editor_group = segment_editor_group
        
    def update_clusters(self):
        self.update_segments()

    def _auto_aggregate_changed(self):
        print "+ automatic aggregation changed:", self.auto_aggregate
        if self.auto_aggregate:
            self.update_segments()
            
    @on_trait_change('+parameter')
    def aggregation_param_changed(self,obj, name, old, new):
        print name, "parameter on aggregator changed"
        if name in self.parameters and self.auto_aggregate:
            self.update_segments()
            
    def _render_segments_changed(self):
        print "+ render_segments changed to", self.render_segments
        self.scene3d.disable_render= True
        for tds in self.track_sets:
            tds.set_segment_visibility(self.render_segments)
        self.scene3d.disable_render = False

    def update_segments(self):
        """ Creates new segments when a aggregation parameter has been
        changed. Will CREATE MayaVi objects if ``self.interactive``
        and ``self.render_segments`` are true.
        """

        print "+ Updating segment assignments."
        _segments = []
        for tnum, tds in enumerate(self.track_sets):
            __segments = []
            segments = self.segment(tds)
            number_of_labeled_segments, indices = summarize_segments(segments)
            labels = sorted(number_of_labeled_segments.keys())
            for labelnum, label in enumerate(labels):
                __segments.append( 
                    Segment(
                        ntracks = len(indices[label]),
                        segment_id = label,
                        indices=np.array(indices[label]),
                        scan_id = tds.scan_id,
                        segments = segments,
                        ncoords = number_of_labeled_segments[label]
                        )
                    )
                    
            # This grabs the colors from the rendered streamlines
            tds.set_segments( __segments)
            _segments += tds.segments # collect the colorized version
        self.segments = _segments
        # Take care of the graphics
        if self.render_segments:
            print "\t++ rendering tracks"
            self.draw_segments()
        print "+ Aggregation Complete"
        
    ## Must be overwritten by a subclass
    def segment(self,track_datasets):
        raise NotImplementedError()
    
    def draw_segments(self):
        if not self.render_segments: return
        print "+ Drawing Segments"
        self.scene3d.disable_render = True
        for tds in self.track_sets:
            tds.draw_segments()
        self.scene3d.disable_render = False


    segment_table = TabularEditor(adapter=SegmentAdapter(),
                                  editable=False)

    segment_rows = Group(Item(name='segments', editor=segment_table,
                                height=400, show_label=False),
                     label="Segmentation Options", show_border=True)
    algorithm_widgets = Group()

    def default_traits_view(self):
        return View( self.segment_editor_group )

    browser_view = View(HSplit(
                     Item('scene3d',
                         editor=SceneEditor(scene_class=Scene),
                         height=500, width=500),
                     segment_editor_group,
                     show_labels=False
                    ),
                )