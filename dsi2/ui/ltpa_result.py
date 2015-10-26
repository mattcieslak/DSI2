#!/usr/bin/env python

from traits.api import HasTraits, Instance, Array, Bool, Dict, \
     on_trait_change, Delegate, List, Color, Any, Instance, Int, File, \
     Button, Enum, Str, DelegatesTo, Property, CFloat,Range
from traitsui.api import View, Item, HGroup, VGroup, \
     Group, Handler, HSplit, VSplit, RangeEditor, Include, Action, MenuBar, Menu, \
     TableEditor, ObjectColumn, Separator

from traitsui.extras.checkbox_column import CheckboxColumn

from ..volumes.scalar_volume import ScalarVolumes

from tvtk.pyface.scene import Scene
from mayavi.core.ui.api import SceneEditor

from traitsui.color_column import ColorColumn
from mayavi.core.api import PipelineBase, Source
from mayavi import mlab

from traitsui.editors.tabular_editor import TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from traitsui.file_dialog import save_file, open_file
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk

from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
import os
import numpy as np
from dsi2.volumes.scalar_volume import ScalarVolume
from dsi2.streamlines.track_dataset import TrackDataset

import cPickle as pickle

ltpa_result_table = TableEditor(
    columns = 
    [
        ObjectColumn(name="name"),
        ObjectColumn(name="n_coordinates"),
        CheckboxColumn(name="visible"),
        ObjectColumn(name="coord_opacity"),
        ObjectColumn(name="tracksA_opacity"),
        CheckboxColumn(name="tracksA_visible"),
        ObjectColumn(name="tracksB_opacity"),
        CheckboxColumn(name="tracksB_visible"),
        ColorColumn(name="colorA", width=5),
        ColorColumn(name="colorB", width=5),
        ObjectColumn(name="coord_shape"),
        ObjectColumn(name="coord_radius"),
        
    ],
    auto_size=False,
)

class CoordinatesGraphic(HasTraits):
    # Data
    scalars = Array
    indices = Array
    
    radius = CFloat(0.5)

    # Holds the mayavi objects
    source = Instance(Source,transient=True)
    glyph = Instance(PipelineBase, transient=True)
    glyph_drawn = Bool(False, transient=True)
    splatter = Instance(PipelineBase,transient=True)
    glyph_opacity = Range(high=1.0,low=0.0,value=1.)
    

    # MayaVi data options
    color_map = Enum(
        [ "Blues", "Oranges", "pink", "Greens"] )
    render_type = Enum(["static_spheres","sized_cubes",
                        "static_cubes","splatter"])
    static_color = Color("green")
    visible = Bool(False)

            
    def set_visibility(self, visibility):
        if not visibility:
            if not self.glyph_drawn: return
        else:
            if not self.glyph_drawn:
                self.render()
                
        # Set visibility of all items
        for viz in [self.glyph, self.splatter]:
            if viz:
                viz.visible = visibility

    def render(self):
        if not self.visible: return
        try:
            color = self.static_color.toTuple()
        except:
            color = (self.static_color.red(),self.static_color.green(),self.static_color.blue())
        static_color = color[0]/255., color[1]/255., color[2]/255.
        
        if self.render_type == "sized_cubes":
            self.glyph = mlab.pipeline.glyph(
                self.source, colormap=self.color_map, mode="cube" )
        elif self.render_type == "splatter":
            self.splatter =  mlab.pipeline.gaussian_splatter(self.source)
            self.glyph = mlab.pipeline.volume(
                self.splatter,
                color=static_color)
        elif self.render_type == "static_cubes":
            self.source = mlab.pipeline.scalar_scatter(
                self.indices[:,0],self.indices[:,1],self.indices[:,2])
            self.glyph = mlab.pipeline.glyph(
                self.source, color=static_color, mode="cube" )
        elif self.render_type == "static_spheres":
            self.source = mlab.pipeline.scalar_scatter(
                self.indices[:,0],self.indices[:,1],self.indices[:,2])
            self.glyph = mlab.pipeline.glyph(
                self.source, color=static_color, 
                mode="sphere" )
            self.glyph.glyph.glyph_source.glyph_source.radius = self.radius
        self.glyph.actor.property.opacity = self.glyph_opacity
        self.glyph_drawn = True
            
    

    def _color_map_changed(self):
        self.clear()
        self.render()


    instance_view = View(
        Group(
        Item("filepath"),
        Group(Item("visible"),Item("glyph"),Item("splatter"),Item("source"),orientation="horizontal"),
        Item("static_color"),
        Item("b_render"),
        orientation="vertical")
        )


class LTPAResult(HasTraits):
    name=Str("LTPA Result")
    # 3d MayaVi scene that will display slices and streamlines
    scene3d = Instance(MlabSceneModel,transient=True)

    n_coordinates=Property(Int)
    def _get_n_coordinates(self):
        try:
            return self.result_coords.shape[0]
        except Exception, e:
            return 0

    # Data objects
    result_coords = Array
    result_coord_scalars = Array
    coords_apply_to = Enum("A","B")
    tracksA = Instance(TrackDataset)
    tracksB = Instance(TrackDataset)
    
    #graphics options
    coord_shape = Enum("sphere", "cube")
    coord_radius = CFloat(1.0)
    colorA = Color("red")
    colorB = Color("blue")
    showA_as = Enum("splatter","tracks")
    showB_as = Enum("splatter","tracks")
    coord_group = Enum("A","B")
    visible = Bool(False)
    tracksA_opacity = Range(0.0,1.0,0.05)
    tracksA_visible = Bool(False)
    tracksB_opacity = Range(0.0,1.0,0.05)
    tracksB_visible = Bool(False)
    
    
    # graphics objects
    coord_graphic = Instance(CoordinatesGraphic,transient=True)
    coord_opacity = Range(0.0,1.0,1.)
    
    def __init__(self,**traits):
        super(LTPAResult,self).__init__(**traits)
        # prepare track datasets for plotting        
        for tds in [self.tracksA, self.tracksB]:
            tds.render_tracks = True
            tds.tracks_drawn = False
            tds.dynamic_color_clusters = False
        self.tracksA.static_color = self.colorA
        self.tracksB.static_color = self.colorB
    
    def _coord_graphic_default(self):
        """ 
        Looks at the contents of this result object
        """
        if self.coords_apply_to == "A":
            c = self.colorA
        else:
            c = self.colorB
            
        return CoordinatesGraphic(
            indices = self.result_coords,
            static_color=c,
            scalars = self.result_coord_scalars,
            radius=self.coord_radius
        )
    
    def _coord_opacity_changed(self):
        self.coord_graphic.glyph.actor.property.opacity = self.coord_opacity
        
    def _visible_changed(self):
        """
        """
        for tds in [self.tracksA, self.tracksB]:
            tds.set_track_visibility(self.visible)
            self._tracksA_opacity_changed()
            self._tracksB_opacity_changed()
        self.coord_graphic.visible = self.visible
        self.coord_graphic.set_visibility(self.visible)
        
    def _tracksA_opacity_changed(self):
        if self.tracksA.tracks_drawn:
            self.tracksA.src.actor.property.opacity = self.tracksA_opacity
    def _tracksA_visible_changed(self):
        if self.tracksA.tracks_drawn:
            self.tracksA.set_track_visibility(self.tracksA_visible)
            
    def _tracksB_opacity_changed(self):
        if self.tracksB.tracks_drawn:
            self.tracksB.src.actor.property.opacity = self.tracksB_opacity
    def _tracksB_visible_changed(self):
        if self.tracksB.tracks_drawn:
            self.tracksB.set_track_visibility(self.tracksB_visible)
        

class LTPAResults(HasTraits):
    scene3d_inited = Bool(False)
    results = List(Instance(LTPAResult))
    scene3d = Instance(MlabSceneModel, (),transient=True)
    def __init__(self,**traits):
        super(LTPAResults,self).__init__(**traits)
        for res in self.results:
            res.scene3d = self.scene3d
    
    traits_view = View(
        Group(
            Item("results", editor=ltpa_result_table),
            show_labels=False
            ),
        resizable=True
        )
    test_view = View(
        Group(
            Item("scene3d",
                         editor=SceneEditor(scene_class=Scene),
                         height=500, width=500),
            Item("results", editor=ltpa_result_table),
            show_labels=False
            ),
        
        resizable=True
        )
    
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        if self.scene3d_inited: return
        #for res in self.results:
        #    res.visible = True
            

def load_ltpa_results(results_pth):
    if not os.path.exists(results_pth):
        raise ValueError("No such file " + results_pth)
    fop = open(results_pth,"rb")
    try:
        res = pickle.load(fop)
    except Exception, e:
        print "Unable to load", results_pth, "because of\n", e
        return LTPAResults()
    # When loading from a pickle, the __init__ isn't properly run.
    # so explicitly run the __init__ code here before returning the result
    #for result in res.results:
    #    for tds in [result.tracksA, result.tracksB]:
    #        tds.render_tracks = True
    #        tds.tracks_drawn = False
    #        tds.dynamic_color_clusters = False
    #    result.tracksA.static_color = result.colorA
    #    result.tracksB.static_color = result.colorB
    return res
    
    
    
