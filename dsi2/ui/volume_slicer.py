#!/usr/bin/env python

import numpy as np
import nibabel as nib
# Traits stuff
from traits.api import ( HasTraits, Instance, Array, 
    Bool, Dict, on_trait_change, Range, Color, Any, Int, 
    DelegatesTo, CInt, Property, File )
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
from ..volumes import get_MNI152

class SlicerPanel(HasTraits):
    # path to a nifti file that holds the data
    reference_volume = File
    
    scene3d_inited = Bool(False)
    # MNI_152 objects. data holds the np array, data_src is for mayavi
    data = Array(value=np.zeros((50,50,50)))
    data_src = Instance(Source)

    # --- Sphere configuration ---
    # position of the cursor
    # Radius of the sphere
    radius = Range(low=0,high=14,value=1)
    extent_x = Int(50)
    extent_y = Int(50)
    extent_z = Int(50)
    sphere_x = Range(low=0, high='extent_x')
    sphere_y = Range(low=0, high='extent_y')
    sphere_z = Range(low=0, high='extent_z')
    sphere_coords = Array
    sphere_color = Color((255,0,0,255))
    sphere_visible = Bool(True)
    coordsupdated = Int(0)
    # Spere's representation on the screen
    sphere_viz = Instance(PipelineBase)
    
    widgets_drawn = Bool(False)
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

    def __init__(self, **traits):
        """ Creates a panel for viewing a 3d Volume.
        Parameters:
        ===========

        """
        super(SlicerPanel,self).__init__(**traits)
        self.sphere_coords
        self.scene3d
        self.sphere_viz
        
    @on_trait_change("reference_volume")
    def render_volume(self):
        if not os.path.exists(self.reference_volume):
            print "No such file", self.reference_volume
            return
        print "Opening", self.reference_volume
        try:
            data = nib.load(self.reference_volume)
        except Exception, e:
            print "Unable to load data", e
            return
        
        # Remove imageplane widgets 
        self.scene3d.disable_render = True
        if self.widgets_drawn:
            self.x_slice_plane.remove()
            self.y_slice_plane.remove()
            self.z_slice_plane.remove()
        # Set data and update the data_src
        self.data = data.get_data()
        # Change the extents to match the new volume
        self.extent_x, self.extent_y, self.extent_z = self.data.shape
        # Send to mayavi
        self.data_src = mlab.pipeline.scalar_field(self.data,
                            figure=self.scene3d.mayavi_scene,
                            name='Data',colormap="gray")
        # Send the new data to the slices
        self.slice_plots.set_volume(self.data)
        # Update the sphere to be in the middle of this volume 
        self.sphere_x = self.extent_x / 2
        self.sphere_y = self.extent_y / 2
        self.sphere_z = self.extent_z / 2
        self.x_slice_plane = self.make_x_slice_plane()
        self.x_slice_plane.ipw.sync_trait(
            "slice_position", self, alias="x")
        self.x_slice_plane.ipw.sync_trait(
            "enabled", self.slice_plots, alias="x_slice_plane_visible")
        self.y_slice_plane = self.make_y_slice_plane()
        self.y_slice_plane.ipw.sync_trait(
            "slice_position", self, alias="y")
        self.y_slice_plane.ipw.sync_trait(
            "enabled", self.slice_plots, alias="y_slice_plane_visible")
        self.z_slice_plane = self.make_z_slice_plane()
        self.z_slice_plane.ipw.sync_trait(
            "slice_position", self, alias="z")
        self.z_slice_plane.ipw.sync_trait(
            "enabled", self.slice_plots, alias="z_slice_plane_visible")
        
        self.scene3d.disable_render = False
        
        self.widgets_drawn = True
        
    def _slice_plots_default(self):
        return Slices()

    def _sphere_viz_default(self):
        # different between wx and qt
        try:
            color_tuple = self.sphere_color.toTuple()
        except:
            color_tuple = self.sphere_color
            
        try:
            pts = mlab.points3d(
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
        except:
            pts = mlab.points3d(
                self.sphere_coords[:,0],
                self.sphere_coords[:,1],
                self.sphere_coords[:,2],
                mode='cube',
                scale_factor=1,
                figure = self.scene3d.mayavi_scene,
                color  = (1.,0.,0.)
                )
        return pts

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


    def make_x_slice_plane(self):
        ipw = mlab.pipeline.image_plane_widget(
            self.data_src,
            figure=self.scene3d.mayavi_scene,
            plane_orientation='x_axes',
            name='Cut x',colormap="gray"
            )
        ipw.ipw.slice_position=self.x
        ipw.ipw.interaction = 0
        return ipw

    def make_y_slice_plane(self):
        ipw = mlab.pipeline.image_plane_widget(
            self.data_src, colormap='gray',
            figure=self.scene3d.mayavi_scene,
            plane_orientation='y_axes',
            name='Cut y')
        ipw.ipw.slice_position=self.y
        ipw.ipw.interaction = 0
        return ipw

    def make_z_slice_plane(self):
        ipw = mlab.pipeline.image_plane_widget(
            self.data_src,colormap='gray',
            figure=self.scene3d.mayavi_scene,
            plane_orientation='z_axes',
            name='Cut z')
        ipw.ipw.slice_position=self.z
        ipw.ipw.interaction = 0
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
        if self.scene3d_inited: return
        self.scene3d.mlab.view(40, 50)
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
        #self.scene3d.mayavi_scene.scene.light_manager.light_mode = "vtk"
        self.scene3d_inited = True

    @on_trait_change('x_slice_plane_visible,y_slice_plane_visible,z_slice_plane_visible')
    def update_slice_opacity(self,obj,name,old,new):
        if name=='x_slice_plane_visible':
            self.x_slice_plane.ipw.texture_visibility = new
        if name=="y_slice_plane_visible":
            self.y_slice_plane.ipw.texture_visibility = new
        if name=="z_slice_plane_visible":
            self.z_slice_plane.ipw.texture_visibility = new

    sphere_widgets = VGroup(
       Item(name="sphere_x",
            editor=RangeEditor(
                        auto_set=False,
                        mode="slider",
                        low=0,
                        high_name="extent_x",
                        format    = "%i")),
       Item(name="sphere_y",
            editor=RangeEditor(
                        auto_set=False,
                        mode="slider",
                        low=0,
                        high_name='extent_y',
                        format    = "%i")),
         Item(name="sphere_z",
            editor=RangeEditor(
                        auto_set=False,
                        mode="slider",
                        low=0,
                        high_name='extent_z',
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