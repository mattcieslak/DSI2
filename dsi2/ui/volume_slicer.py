#!/usr/bin/env python

import numpy as np
import nibabel as nib
# Traits stuff
from traits.api import HasTraits, Instance, Array, \
    Bool, Dict, on_trait_change, Range, Color, Any, Int, DelegatesTo
from traitsui.api import View, Item, VGroup, \
    HGroup, Group, RangeEditor, ColorEditor, VSplit

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel

from tvtk.pyface.scene import Scene
from tvtk.api import tvtk

from chaco.chaco_plot_editor import ChacoPlotItem
from chaco.api import Plot, ArrayPlotData, gray
from enable.component_editor import ComponentEditor

from ..streamlines.track_math import sphere_around_ijk
from ..volumes.scalar_volume import ScalarVolume
from .chaco_slice import Slices
import os

# -- MNI152@2mm LAS --
nim = nib.load(os.getenv("DSI2_DATA") + "/MNI152_T1_2mm.nii.gz")
extents = nim.get_header().get_data_shape()


class SlicerPanel(HasTraits):
    # MNI_152 objects. data holds the np array, data_src is for mayavi
    data = Array
    data_src = Instance(Source)

    # --- Sphere configuration ---
    # position of the cursor
    # Radius of the sphere
    radius = Range(low=0,high=14,value=1)
    sphere_x = Range(low=0,high=extents[0],value=extents[0]/2)
    sphere_y = Range(low=0,high=extents[1],value=extents[1]/2)
    sphere_z = Range(low=0,high=extents[2],value=extents[2]/2)
    sphere_coords = Array
    sphere_color = Color((255,0,0,255))
    sphere_visible = Bool(True)
    coordsupdated = Int(0)
    # Spere's representation on the screen
    sphere_viz = Instance(PipelineBase)
    x_slice_plane = Instance(PipelineBase)
    y_slice_plane = Instance(PipelineBase)
    z_slice_plane = Instance(PipelineBase)



    # Slice plots
    slice_plots = Instance(Slices)
    x = DelegatesTo('slice_plots')
    y = DelegatesTo('slice_plots')
    z = DelegatesTo('slice_plots')

    # 3d image plane widget
    scene3d = Instance(MlabSceneModel, ())
    camera_initialized = False

    # Actual volume data
    mni_data_src = Instance(Source)

    def __init__(self, **traits):
        """ Creates a panel for viewing a 3d Volume.
        Parameters:
        ===========

        """
        super(SlicerPanel,self).__init__(**traits)
        self.data = nim.get_data().astype(float)
        self.sphere_coords
        self.scene3d
        self.sphere_viz
        #self.x_slice_plane
        self.x_slice_plane.ipw.sync_trait(
            "slice_position", self, alias="x")
        #self.x_slice_plane.ipw.sync_trait(
        #    "enabled", self.slice_plots, alias="x_slice_plane_visible")
        #self.y_slice_plane
        self.y_slice_plane.ipw.sync_trait(
            "slice_position", self, alias="y")
        #self.y_slice_plane.ipw.sync_trait(
        #    "enabled", self.slice_plots, alias="y_slice_plane_visible")
        #self.z_slice_plane
        self.z_slice_plane.ipw.sync_trait(
            "slice_position", self, alias="z")
        #self.z_slice_plane.ipw.sync_trait(
        #    "enabled", self.slice_plots, alias="z_slice_plane_visible")

    def _slice_plots_default(self):
        return Slices(volume_data=nim.get_data())

    def _sphere_viz_default(self):
        # different between wx and qt
        try:
            color_tuple = self.sphere_color.toTuple()
        except:
            color_tuple = self.sphere_color
        return mlab.points3d(
            self.sphere_coords[:,0],
            self.sphere_coords[:,1],
            self.sphere_coords[:,2],
            mode='cube',
            scale_factor=1,
            figure = self.scene3d.mayavi_scene,
            color  = (color_tuple[0]/255.,
                     color_tuple[1]/255.,
                     color_tuple[2]/255.)
        )

    def _sphere_coords_default(self):
        return np.array(sphere_around_ijk(
            self.radius, np.array([self.x, self.y, self.z])))

    def _sphere_visible_changed(self):
        self.sphere_viz.visible = self.sphere_visible

    def _sphere_color_changed(self):
        print "changing sphere color to", self.sphere_color
        # different between wx and qt
        try:
            color_tuple = self.sphere_color.toTuple()
        except:
            color_tuple = self.sphere_color
        self.sphere_viz.actor.property.color = (
                     color_tuple[0]/255.,
                     color_tuple[1]/255.,
                     color_tuple[2]/255.)

    def _data_src_default(self):
        return mlab.pipeline.scalar_field(self.data,
                            figure=self.scene3d.mayavi_scene,
                            name='Data',colormap="gray")

    def _x_slice_plane_default(self):
        ipw = mlab.pipeline.image_plane_widget(
            self.data_src,
            figure=self.scene3d.mayavi_scene,
            plane_orientation='x_axes',
            name='Cut x',colormap="gray"
            )
        ipw.ipw.slice_position=self.x
        return ipw

    def _y_slice_plane_default(self):
        ipw = mlab.pipeline.image_plane_widget(
            self.data_src, colormap='gray',
            figure=self.scene3d.mayavi_scene,
            plane_orientation='y_axes',
            name='Cut y')
        ipw.ipw.slice_position=self.y
        return ipw

    def _z_slice_plane_default(self):
        ipw = mlab.pipeline.image_plane_widget(
            self.data_src,colormap='gray',
            figure=self.scene3d.mayavi_scene,
            plane_orientation='z_axes',
            name='Cut z')
        ipw.ipw.slice_position=self.z
        return ipw

    @on_trait_change('sphere_x,sphere_y,sphere_z,radius')
    def _update_sphere(self):
        self.disable_render = True
        self.sphere_coords = np.array(sphere_around_ijk(
            self.radius, np.array([self.sphere_x,
                                   self.sphere_y,
                                   self.sphere_z])))
        self.sphere_viz.mlab_source.reset(
                 x=self.sphere_coords[:,0],
                 y=self.sphere_coords[:,1],
                 z=self.sphere_coords[:,2],
        )
        self.disable_render = False
        self.coordsupdated += 1

    def arbitrary_voxel_query(self,new_indices):
        self.disable_render = True
        self.sphere_coords = np.array(new_indices)
        self.sphere_viz.mlab_source.reset(
                 x=self.sphere_coords[:,0],
                 y=self.sphere_coords[:,1],
                 z=self.sphere_coords[:,2],
        )
        self.disable_render = False
        self.coordsupdated += 1

    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        self.scene3d.mlab.view(40, 50)
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
        #self.scene3d.mayavi_scene.scene.light_manager.light_mode = "vtk"
        self.x_slice_plane.ipw.interaction = 0
        self.y_slice_plane.ipw.interaction = 0
        self.z_slice_plane.ipw.interaction = 0

#    @on_trait_change('x_slice_plane_visible,y_slice_plane_visible,z_slice_plane_visible')
#    def update_slice_opacity(self,obj,name,old,new):
#        if name=='x_slice_plane_visible':
#            self.x_slice_plane.ipw.texture_visibility = new
#        if name=="y_slice_plane_visible":
#            self.y_slice_plane.ipw.texture_visibility = new
#        if name=="z_slice_plane_visible":
#            self.z_slice_plane.ipw.texture_visibility = new

    sphere_widgets = VGroup(
       Item(name="sphere_x",
            editor=RangeEditor(
                        auto_set=False,
                        mode="slider",
                        high = 91,
                        low  = 0,
                        format    = "%i")),
       Item(name="sphere_y",
            editor=RangeEditor(
                        auto_set=False,
                        mode="slider",
                        high = 109,
                        low  = 0,
                        format    = "%i")),
         Item(name="sphere_z",
            editor=RangeEditor(
                        auto_set=False,
                        mode="slider",
                        high = 91,
                        low  = 0,
                        format    = "%i")),
         Item(name="radius"),
         Item(name="sphere_color"),
         Item(name="sphere_visible"),
         label="Search Sphere",
         show_border=True
         )
    plot3d_group = Group(
                     Item('scene3d',
                     editor=SceneEditor(scene_class=Scene),
                     height=500, width=500),
                 show_labels=False)
    slice_panel_group = HGroup(sphere_widgets,
                               Item('slice_plots',style="custom"),
                               show_labels=False)

    # ----- Views -----
    browser_view = View(
        VSplit(
            plot3d_group,
            slice_panel_group
            )
        )
    traits_view = View(
          slice_panel_group
        )

if __name__=="__main__":
    sp = SlicerPanel()
    sp.configure_traits(view="browser_view")
