import numpy as np

from chaco.api import ArrayPlotData, Plot, gray
from enable.component_editor import ComponentEditor
from traits.api import Enum, HasTraits, Instance, Array, Bool, \
     DelegatesTo, Int,Range, CInt
from traitsui.api import Group, HGroup, Item, View, RangeEditor

# MNI
extents = { "x":(109,  91),
            "y":(91,   91),
            "z":(91,  109)
          }

class ChacoSlice(HasTraits):
    # Borrow the volume data from the parent
    volume_data = Array
    plot_data = ArrayPlotData
    plot = Instance(Plot)
    origin = Enum("bottom left", "top left", "bottom right", "top right")
    slice_number = Range(low=0, high=108, value=45)
    x = slice_number
    y = slice_number
    z = slice_number
    slice_plane_visible = Bool(True)
    slice_opacity = Range(0.0,1.0,1.)
    plane = Enum("x","y","z")

    def __init__(self,**traits):
        super(ChacoSlice,self).__init__(**traits)
        x,y = extents[self.plane]
        self.plot_data = ArrayPlotData(imagedata=self.data_source())
        self.plot = Plot(self.plot_data,
                padding=1,
                fixed_preferred_size=(50,50)
                )
        self.renderer = self.plot.img_plot(
            "imagedata", name="plot1",
            colormap=gray,
            aspect_ratio=float(x)/y)[0]

    def _slice_number_changed(self):
        #print self.plane, "slice changed to ",self.slice_number
        self.plot.data.set_data("imagedata",self.data_source())
        self.plot.request_redraw()

    def _origin_changed(self):
        self.renderer.origin = self.origin
        self.plot.request_redraw()

    def data_source(self):
        # Link the appropriate view changer to the slice number change
        if self.plane == "x":
            return self.volume_data[self.slice_number,:,:].T
        elif self.plane == "y":
            return self.volume_data[:,self.slice_number,:].T
        elif self.plane == "z":
            return self.volume_data[:,:,self.slice_number].T

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(),
                             height=100,width=100, show_label=False),
                        HGroup(
                            Item("slice_plane_visible"),
                            Item("slice_number", editor=RangeEditor(
                                    mode="slider",
                                    high   = 109,
                                    low    = 0,
                                    format = "%i")),
                            #Item("slice_opacity"),
                            #Item("origin"),
                            show_labels=False),
                    show_labels=False,
                        ),
                    #width=100, height=200,
                    resizable=True
                    )


class Slices(HasTraits):
    # x slice
    x_slice_window = Instance(ChacoSlice)
    x = CInt(45)
    x_slice_plane_visible = Bool(True)
    x_slice_opacity = DelegatesTo("x_slice_window.slice_opacity")
    # y slice
    y_slice_window = Instance(ChacoSlice)
    y = CInt(54)
    y_slice_plane_visible = Bool(True)
    y_slice_opacity = DelegatesTo("y_slice_window.slice_opacity")
    # z slice
    z_slice_window = Instance(ChacoSlice)
    z = CInt(45)
    z_slice_plane_visible = Bool(True)
    z_slice_opacity = DelegatesTo("z_slice_window.slice_opacity")

    # Plotting
    volume_data = Array

    def __init__(self, volume_data):
        self.volume_data = volume_data
        # Force creation of slice windows
        self.x_slice_window
        self.y_slice_window
        self.z_slice_window
        # Enable communication between x,y,z and their windows
        self.sync_trait("x", self.x_slice_window,
                        alias="slice_number",mutual=True)
        #self.sync_trait("x_slice_plane_visible", self.x_slice_window,
        #                alias="slice_plane_visible",mutual=True)
        self.sync_trait("y", self.y_slice_window,
                        alias="slice_number",mutual=True)
        #self.sync_trait("y_slice_plane_visible", self.y_slice_window,
        #                alias="slice_plane_visible",mutual=True)
        self.sync_trait("z", self.z_slice_window,
                        alias="slice_number",mutual=True)
        #self.sync_trait("z_slice_plane_visible", self.z_slice_window,
        #                alias="slice_plane_visible",mutual=True)

    def _x_slice_window_default(self):
        return ChacoSlice(
            volume_data=self.volume_data,
            plane = "x"
        )
    def _y_slice_window_default(self):
        return ChacoSlice(
            volume_data=self.volume_data,
            plane = "y",
            )
    def _z_slice_window_default(self):
        return ChacoSlice(
            volume_data=self.volume_data,
            plane = "z",
            )

    def _origin_changed(self):
        self.renderer.origin = self.origin
        self.plot.request_redraw()


    #@on_trait_change('x,y,z')
    #def draw_slices(self,obj,name,old,new):
        #if name=='x':
            #self.x_slice_plane.ipw.slice_position = new
        #if name=="y":
            #self.y_slice_plane.ipw.slice_position = new
        #if name=="z":
            #self.z_slice_plane.ipw.slice_position = new

    traits_view = View(
                    HGroup(
                        Item("x_slice_window",style="custom"),
                        Item("y_slice_window",style="custom"),
                        Item("z_slice_window",style="custom"),
                        show_labels=False
                         ),
                    resizable=True
                  )

if __name__ == "__main__":
    import nibabel as nib
    import numpy as np
    import os
    from dsi2.volumes import get_MNI152
    nim = get_MNI152()
    sl = Slices(volume_data=nim.get_data())
    sl.configure_traits()