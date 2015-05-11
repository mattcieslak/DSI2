#!/usr/bin/env python

from traits.api import HasTraits, Instance, Array, Bool, Dict, \
     on_trait_change, Delegate, List, Color, Any, Instance, Int, File, \
     Button, Enum, Str, DelegatesTo, Property
from traitsui.api import View, Item, HGroup, VGroup, \
     Group, Handler, HSplit, VSplit, RangeEditor, Include, Action, MenuBar, Menu, \
     TableEditor, ObjectColumn, Separator

from traitsui.extras.checkbox_column import CheckboxColumn

from dsi2.ui.volume_slicer import SlicerPanel
from dsi2.ui.screen_shooter import ScreenShooter
from dsi2.streamlines.track_dataset  import Cluster, TrackDataset, join_tracks
from dsi2.aggregation.cluster_ui import ClusterAdapter, ClusterEditor
from dsi2.aggregation.clustering_algorithms import QuickBundlesAggregator, FastKMeansAggregator
from dsi2.aggregation.region_clusters import RegionAggregator, RegionPair
from dsi2.database.track_datasource import TrackDataSource
from dsi2.aggregation.cluster_evaluator import AggregationEvaluator, flat_tep
from dsi2.aggregation.region_labeled_clusters import RegionLabelAggregator
from dsi2.volumes.roi import ROI
from dsi2.volumes.scalar_volume import ScalarVolumes
from dsi2.database.traited_query import Scan
from dsi2.volumes import get_MNI152_path

from dsi2.ui.ltpa_result import load_ltpa_results

from tvtk.pyface.scene import Scene
from mayavi.core.ui.api import SceneEditor

from traitsui.editors.tabular_editor import TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from traitsui.file_dialog import save_file, open_file
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk

from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel

import os
import numpy as np


class AlgorithmPicker(HasTraits):
    algorithm = Enum("None", "Brain Atlas", "k-Means", "QuickBundles", "ReebGen")

    def get_algorithm(self):
        if self.algorithm == "Brain Atlas":
            return RegionLabelAggregator()
        if self.algorithm == "k-Means":
            from dsi2.aggregation.clustering_algorithms import FastKMeansAggregator
            return FastKMeansAggregator()
        if self.algorithm == "QuickBundles":
            from dsi2.aggregation.clustering_algorithms import QuickBundlesAggregator
            return QuickBundlesAggregator()
        if self.algorithm == "ReebGen":
            from dsi2.aggregation.segmentation.segmentation_algorithms import ReebGraphSegmentation
            return ReebGraphSegmentation()
        else:
            return ClusterEditor()
        
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
        CheckboxColumn(name="render_tracks",editable=True),
        ObjectColumn(name="color_map",editable=True),
        CheckboxColumn(name="dynamic_color_clusters",editable=True),
        ObjectColumn(name="static_color",editable=True),
    ],
    auto_size  = True,
    edit_view="graphics_view"
    )
        
class StreamlineGraphics(HasTraits):
    track_source = Instance(TrackDataSource)
    track_sets = List(Instance(TrackDataset))
    
    
    # Provides access to
    graphics_view = View(
        Group(
            Group(
                Item("track_sets", editor=track_dataset_graphics_table),
                orientation="horizontal",
                show_labels=False,
                show_border=True,
                label="Graphical Objects"
                ),
            layout="tabbed"
        ),
            Group(
                Item("track_source",
                      editor=track_dataset_source_table),
                orientation="horizontal",
                show_labels=False,
                show_border=True,
                label="Data Sources"
                ),
        resizable=True,
        width=900,
        height=500,
        title="Streamline visualization options"

    )
class SphereBrowser(HasTraits):
    # Object that slices the MNI brain
    vslicer =    Instance(SlicerPanel)
    # Object that can update streamline labels
    aggregator =  Instance(ClusterEditor)
    # Holds a number of queryable datasources
    track_source = Instance(TrackDataSource)
    downsample = Int(0)

    # 3d MayaVi scene that will display slices and streamlines
    scene3d =    Delegate('vslicer')
    coordsupdated = Delegate('vslicer')

    # Object that evaluates how well the aggregation performed
    evaluator = Instance(AggregationEvaluator)

    # Widget for querying specific region pair streamlines
    roi_query = Instance(ROI)

    # Streamline Graphics Editor
    sl_editor = Instance(StreamlineGraphics)
    
    # Object that manages additional volume glyphs to be rendered
    additional_volumes = Instance(ScalarVolumes)

    # Object that enables the selection between aggregation 
    algorithm_picker = Instance(AlgorithmPicker)

    # For saving trk files and screencaps, keep track of the file
    # Path for saving
    save_path = File(os.getcwd())
    save_name = DelegatesTo("aggregator")

    # A list of file paths to
    reference_volume = DelegatesTo("vslicer")
    
    def __init__(self,**traits):
        """
        Event handler for a number of objects that all work together.
        A Sphere Browser lets the user
        1) view the MNI brain in a mayavi window, optionally with
           streamlines overlaid
        2) apply labels to the streamlines via aggregation or an atlas
           2a) if using a aggregation algorithm, the parameters can be
               updated through the cluster ui. and visualized in realtime
        3) A slice panel lets the user move slices and toggle their visibility
           in the 3d viewer.
        4) search coordinates may be specified via the sphere control sliders
           or loading a nifti file with a mask.

        Each of these subcomponents previously had their own event handlers
        which made running without a gui unstable (I'm not sure why).
        All event handlers have been moved from their models and placed here.

        """
        super(SphereBrowser,self).__init__(**traits)
        self.scene3d
        self.vslicer
        #self.aggregator.scene3d = self.scene3d
        #self.aggregator.set_track_source(self.track_source)

    # Default UI Items
    def _vslicer_default(self):
        return SlicerPanel()
    def _aggregator_default(self):
        #return RegionAggregator(scene3d=self.vslicer.scene3d)
        return ClusterEditor(scene3d=self.scene3d)
    def _track_source_default(self):
        #return TrackDataSource(scene3d=self.vslicer.scene3d)
        return TrackDataSource()
    def _evaluator_default(self):
        # Gives and evaluator access to this object's track_sets
        return AggregationEvaluator()
    def _roi_query_default(self):
        return ROI()
    def _additional_volumes_default(self):
        return ScalarVolumes(scene3d=self.scene3d)
    
    def _sl_editor_default(self):
        return StreamlineGraphics(
            track_sets = self.aggregator.track_sets,
            track_source = self.track_source
        )
    

    def set_track_source(self,track_source):
        """ Supply the aggregator/browser with a list of queryable
        streamline sources

        Parameters:
        -----------
          track_source:dsi2.database.track_datasource.TrackDataSource
        """
        track_source.interactive = True
        self.track_source = track_source
        self.aggregator.set_track_source(track_source)


    @on_trait_change('coordsupdated')
    def update_tracks_from_coords(self):
        """ User has changed the query coordinates.
        1) Query the datasource for new streamlines
        2) Send them to the aggregator for aggregation
        3) Disable mayavi rendering
        4) Remove previous streamlines from the engine
        5) Add new streamlines to the engine
           -- if we're aggregation, then paint the streamlines
        6) re-enable mayavi rendering

        """
        self.aggregator.clear_clusters()
        
        #print "============================================"
        if len(self.track_source) == 0:
            print "\t+ No datasets in the track_source"
            return
        self.scene3d.disable_render = True
        
        # Query the track source with our search coordinates
        new_tracks = self.track_source.query_ijk(
                         map(tuple,self.vslicer.sphere_coords),
                                   every=self.downsample)
        self.aggregator.set_track_sets(new_tracks)

        # Apply aggregation to the new ``track_sets`` if requested
        if self.aggregator.auto_aggregate:
            #print "\t++ Applying aggregation to them ..."
            self.aggregator.update_clusters()
        # Render their glyphs if the user wants
        if self.aggregator.render_tracks:
            #print "\t++ Rendering the new tracks."
            self.aggregator.draw_tracks()
        #print "\t++ Done"
        self.scene3d.disable_render = False
        print "\t+ Re-enabling rendering"
        self.save_name = "query.x%i.y%i.z%i.r%i"%(
                                   self.vslicer.sphere_x,
                                   self.vslicer.sphere_y,
                                   self.vslicer.sphere_z,
                                   self.vslicer.radius)

    def render_region_pairs(self,region_id):
        self.aggregator._b_query_region_pairs_fired()

    a_query_region_pair = Action( name = "Query a Region-Pair",
              action = "query_region_pair",
              enabled_when= "algorithm_picker.algorithm == 'Brain Atlas'")

    def clear_track_glyphs(self):
        self.scene3d.disable_render=True
        while len(self.scene3d.scene.mayavi_scene.children) > 2:
            self.scene3d.scene.mayavi_scene.children[-1].remove()
        self.scene3d.disable_render=False

    # Menu Items
    def take_screenshot(self):
        s=ScreenShooter(
            filepath=os.path.join(self.save_path,self.save_name+".png"))
        ui = s.edit_traits()
        if not ui.result:
            print "screenshot cancelled"
            return
        self.scene3d.mayavi_scene.scene.save(s.filepath)
    a_take_screenshot = Action( name = "Save screen shot",
                                action = "take_screenshot")
    
    def edit_volumes(self):
        self.additional_volumes.edit_traits()
    a_edit_volumes = Action(name = "Render additional volumes",
                            action = "edit_volumes")

    def set_coords_from_nifti(self):
        ui = self.roi_query.edit_traits()
        if not ui.result:
            print "search cancelled"
            return
        self.vslicer.arbitrary_voxel_query(self.roi_query.query_indices())
        self.save_name = self.roi_query.get_savename()
    a_set_coords_from_nifti = Action( name = "Search from NIfTI",
                                      action = "set_coords_from_nifti")

    def evaluate_aggregation(self):
        print "No longer Implemented"
        #self.evaluator.edit_traits()
    a_evaluate_aggregation = Action( name = "Evaluate Aggregation",
                                    action = "evaluate_aggregation" )

    def change_datasource(self):
        from dsi2.ui.browser_builder import BrowserBuilder
        bb = BrowserBuilder()
        #TODO: Check whether livemodal goes in viewdef or in conf call 
        ui = bb.edit_traits(view="dsource_view",kind="modal")
        if not ui.result:
            print "No new data source set"
            return
        
        self.set_track_source(bb.get_datasource())
        
    a_change_datasource = Action( name = "Select data source",
                                  action = "change_datasource")

    def _algorithm_picker_default(self):
        return AlgorithmPicker()
    
    @on_trait_change("algorithm_picker.algorithm")
    def algorighm_change(self):
        # A new algorithm got picked. Set up a new aggregator
        print "algorithm is ", self.algorithm_picker.algorithm
        new_agg = self.algorithm_picker.get_algorithm()
        # Brain atlas needs access to track_source
        if self.algorithm_picker.algorithm == "Brain Atlas":
            new_agg.set_track_source(self.track_source)
        if new_agg is None: return
        new_agg.set_track_sets(self.aggregator.track_sets)
        new_agg.scene3d = self.scene3d
        for attr in ("render_tracks", "auto_aggregate"):
            setattr(new_agg,attr,getattr(self.aggregator,attr))
        self.aggregator = new_agg
        print "Algorithm changed"
        
    a_change_aggregation_algorithm = Action( name="Change aggregation algorithm",
                                            action = "change_aggregation_algorithm")
    
    def change_reference_volume(self):
        fpath = open_file(
            title="Choose Reverence Volume File",
            filter=["*.img", "*.hdr", "*.nii.gz", "*.nii"],
            file_name = self.save_path
                          )
        if fpath == "": return
        self.reference_volume = fpath
        
    a_change_reference_volume = Action( name="Change refence volume",
                                            action = "change_reference_volume")
    def add_mni_152(self):
        self.reference_volume = get_MNI152_path()
        
    a_mni152_2mm_reference_volume = Action( name="Add MNI152 2mm T1",
                                            action = "add_mni_152")
    def save_streamlines(self):
        saver = StreamlineSaver(
                      track_sets=self.aggregator.track_sets,
                      subjects=self.track_source.track_dataset_properties,
                      file_prefix=os.path.join(self.save_path,self.save_name)
        )
        saver.edit_traits()
    a_save_streamlines = Action( name = "Save visible streamlines (trk)",
                                 action = "save_streamlines")
    def add_streamlines(self):
        """ Adds streamlines to the working set"""
        fpath = open_file(title="Select Streamline File",
                       filter=["*.txt","*.trk","*trk.gz","*.mat",
                               ],
            file_name = self.save_path
                       )
        if fpath == "": return
        try:
            tds = TrackDataset(fpath)
        except Exception,e:
            print "Unable to load", fpath, "because", e
            return
        
        self.aggregator.add_tracks(tds)
    a_add_streamlines = Action( name = "Add streamlines to working set",
                                 action = "add_streamlines")
    
    def add_ltpa_result(self):
        """ Adds streamlines to the working set"""
        fpath = open_file(title="Select Streamline File",
                       filter=["*.ltpa"],
            file_name = os.path.expanduser("~")
                       )
        if fpath == "": return
        try:
            lr = load_ltpa_results(fpath)
        except Exception,e:
            print "Unable to load", fpath, "because", e
            return
        
        lr.scene3d = self.scene3d
        lr.edit_traits()
        self.ltpa_result = lr
        
    a_load_ltpa = Action( name = "Load LTPA results",
                                 action = "add_ltpa_result")
    
    def edit_streamline_graphics(self):
        self.sl_editor.edit_traits(view="graphics_view")
        
    a_edit_streamline_viz = Action( name = "Edit streamline graphics",
                                 action = "edit_streamline_graphics")
    def save_csv(self):
        if not hasattr(self.aggregator, "get_R_dat"):
            print "Not implemented for this kind of aggregator"
            return
        
        r_txt = self.aggregator.get_R_dat()
        fpath = save_file()
        fop = open(fpath, "w")
        fop.write(r_txt)
        fop.close()
        
    a_save_csv = Action( name = "Save visible streamlines (csv)",
                                 action = "save_csv")

    def edit_scene3d(self):
        self.scene3d.edit_traits()
    a_edit_scene3d = Action( name = "3D Settings",
                             action = "edit_scene3d")

    traits_view = View(
                 VSplit(
                 HSplit(
                     VGroup(
                     Item("algorithm_picker",style="custom"),
                     Item("aggregator",style="custom"),
                     show_labels=False),
                     Item(
                         "scene3d",editor=SceneEditor(scene_class=Scene),
                           height=500, width=500),
                     show_labels=False
                         ),
                     Item("vslicer", style="custom"),
                     show_labels=False
                 ),
                 menubar = MenuBar(
                     Menu(
                          a_add_streamlines,
                          a_change_datasource,
                          Separator(),
                          a_change_reference_volume,
                          a_mni152_2mm_reference_volume,
                          Separator(),
                          a_load_ltpa,
                          name="Data"
                         ),
                     Menu(
                          a_set_coords_from_nifti,
                          name="Query"),
                     Menu(
                          a_query_region_pair,
                          a_evaluate_aggregation,
                          name="Aggregation"
                         ),
                     Menu(
                          a_edit_streamline_viz,
                          a_edit_scene3d,
                          a_edit_volumes,
                          a_take_screenshot,
                          name = "Graphics"
                        ),
                     Menu(
                          a_save_streamlines,
                          a_save_csv,
                          name="Streamlines"
                          )
                     ),
                 resizable=True,
                 title="Voxel Browser"
    )

label_table = TableEditor(
    columns = \
    [ ObjectColumn(name="scan_id",editable=False),
      ObjectColumn(name="scan_group",editable=False),
      ObjectColumn(name="label",editable=True),
    ],
    auto_size=True
    )

class StreamlineSaver(HasTraits):
    track_sets = List(Instance(TrackDataset))
    subjects = List(Instance(Scan)) # holds track_dataset_properties
    use_labels = Bool(False)
    labels = List
    # For defining groups of subjects to apply filters to
    split_factor = Enum("None", "scan_group","subject_id","scan_id")
    subject_labels = Array
    file_prefix = File
    b_save = Button(label="Save .trk")

    def _split_factor_changed(self):
        """
        assign a value to props.label for each of the
        info objects based on that info's value for
        ``self.split_factor``
        """
        if self.split_factor == "None":
            self.use_labels = False
            return
        self.use_labels = True
        labels = set()
        for info in self.subjects:
            labels.update([getattr(info, self.split_factor)])
        label_lut = dict([(name,code) for code,name in enumerate(labels)])
        self.labels = label_lut.keys()
        print label_lut
        lbl = []
        for info in self.subjects:
            _lbl = label_lut[getattr(info,self.split_factor)]
            info.label = _lbl
            lbl.append(_lbl)
        self.subject_labels = np.array(lbl)

    def _b_save_fired(self):

        """Somebody clicked the "Save" button"""
        # If we don't need to make separate .trk files for different labels
        if not self.use_labels:
            all_trks = join_tracks(self.track_sets)
            outf = self.file_prefix + ".trk"
            print "saving", outf
            all_trks.save(outf, use_mni_header=True)
            return
        # Save a separate trk file for each label type
        for label in self.labels:
            lbl_trks = join_tracks(
                [ trk for trk in self.track_sets \
                   if getattr(trk.properties,self.split_factor) == label ]
                )
            outf = self.file_prefix + "_%s.trk" % label
            print "saving", outf
            lbl_trks.save( outf,
                           use_mni_header=True)

    # widgets for editing algorithm parameters
    traits_view = View(Group(
                        Item("file_prefix"),
                        Item("split_factor"),
                        Item("subjects",editor=label_table),
                        Item("b_save"),
                         show_border=True,
                        ),
                         resizable=True,
                         height=400,
                         kind='nonmodal'
                       )
