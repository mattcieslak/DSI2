#!/usr/bin/env python

from traits.api import \
     HasTraits, Str, Int, List, Button, File, Instance, Dict,Enum, \
     on_trait_change, Array, Bool, Color, Tuple, Button
from traitsui.api import Group, View, Handler, Item, \
     OKButton, CancelButton, EnumEditor, TableEditor, \
     CheckListEditor, ObjectColumn
import numpy as np
import os
import nibabel as nib

# Mayavi classes
from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk

class ScalarVolume(HasTraits):
    # Data
    filepath = File("")
    ijk   = Tuple
    scalars = Array

    # Holds the mayavi objects
    source = Instance(Source)
    glyph = Instance(PipelineBase)
    splatter = Instance(PipelineBase)

    # MayaVi data options
    color_map = Enum(
        [ "Blues", "Oranges", "pink", "Greens"] )
    render_type = Enum(["sized_cubes","static_cubes","splatter"])
    static_color = Color
    visible = Bool(True)

    b_render = Button(label="Render")

    def _b_render_fired(self):
        self.clear()
        self.render()

    def _filepath_changed(self):
        data = nib.load(self.filepath).get_data()
        self.indices = np.nonzero(data)
        self.scalars = data[self.indices]

    def _visible_changed(self):
        if self.glyph is not None:
            self.glyph.visible = self.visible

    def clear(self):
        if self.glyph is not None:
            try:
                self.glyph.remove()
            except Exception, e:
                print e
        if not self.splatter is None:
            try:
                self.splatter.remove()
            except Exception, e:
                print e

    def render(self):
        if not self.visible: return
        try:
            color = self.static_color.toTuple()
        except:
            color = self.static_color
        static_color = color[0]/255., color[1]/255., color[2]/255.

        self.source = mlab.pipeline.scalar_scatter(
            self.indices[0],self.indices[1],self.indices[2],self.scalars)
        if self.render_type == "sized_cubes":
            self.glyph = mlab.pipeline.glyph(
                self.source, colormap=self.color_map, mode="cube" )
        elif self.render_type == "splatter":
            self.splatter =  mlab.pipeline.gaussian_splatter(self.source)
            self.glyph = mlab.pipeline.volume(
                self.splatter,
                color=static_color)
        if self.render_type == "static_cubes":
            self.source = mlab.pipeline.scalar_scatter(
                self.indices[0],self.indices[1],self.indices[2])
            self.glyph = mlab.pipeline.glyph(
                self.source, color=static_color, mode="cube" )


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

volume_table = TableEditor(
    columns = [
        ObjectColumn(name="color_map", editable=True),
        ObjectColumn(name="static_color", editable=True),
        ObjectColumn(name="render_type", editable=True),
        ObjectColumn(name="visible",editable=True),
        ObjectColumn(name="filepath", editable=True),
    ],
    deletable  = True,
    auto_size  = True,
    show_toolbar = True,
    edit_view="instance_view",
    row_factory=ScalarVolume,
    orientation="vertical"
    )

class ScalarVolumes(HasTraits):
    volumes = List(Instance(ScalarVolume))
    scene3d = Instance(MlabSceneModel)

    def _scene3d_default(self):
        return MlabSceneModel()

    def render_regions(self):
        self.scene3d.disable_render = True
        for volume in self.volumes:
            volume.render()
        self.scene3d.disable_render = False

    test_view = View(
        Item("volumes",editor=volume_table),
        Group(
        Item("scene3d",
             editor=SceneEditor(scene_class=Scene),
             height=500, width=500),
        show_labels=False),
        resizable=True
        )
    traits_view = View(
        Group(
            Item("volumes",editor=volume_table),
            show_labels=False
            ),
        resizable=True
        )



if __name__ == "__main__":
    vpth = "../../scripts/pws_volumes/Left-Putamen.ctrl.MNI.nii.gz"
    vol = [ScalarVolume(filepath=vpth)]
    vols = ScalarVolumes(volumes=vol)
    vols.edit_traits(view="test_view")
