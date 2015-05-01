import numpy as np

from chaco.api import ArrayPlotData, Plot, gray
from enable.component_editor import ComponentEditor
from traits.api import Enum, HasTraits, Instance, Array, Bool, \
     DelegatesTo, Int,Range, CInt, Property, Tuple, CBool, CArray
from traitsui.api import Group, HGroup, Item, View, RangeEditor


class ChacoSlice(HasTraits):
    # Borrow the volume data from the parent
    plot_data = ArrayPlotData
    plot = Instance(Plot)
    origin = Enum("bottom left", "top left", "bottom right", "top right")
    top_slice = Int
    slice_number = Range(low=0, high_name="top_slice", value=0)
    slice_plane_visible = Bool(True)
    slice_opacity = Range(0.0,1.0,1.)
    plane = Enum("x","y","z")

    def __init__(self,**traits):
        super(ChacoSlice,self).__init__(**traits)
        self.plot_data = ArrayPlotData(imagedata=self.data_source())
        self.plot = Plot(self.plot_data,
                padding=1,
                fixed_preferred_size=(50,50),
                bgcolor="black"
                )
        aspect_ratio = 1
        self.renderer = self.plot.img_plot(
            "imagedata", name="plot1",
            colormap=gray,
            aspect_ratio=aspect_ratio)[0]
        
    def _slice_number_changed(self):
        #print self.plane, "slice changed to ",self.slice_number
        self.plot.data.set_data("imagedata",self.data_source())
        self.plot.request_redraw()

    def _origin_changed(self):
        self.renderer.origin = self.origin
        self.plot.request_redraw()
        
    def reset_aspect_ratio(self):
        x,y = self.get_extents()
        self.renderer.aspect_ratio = float(x)/y
        
    def get_extents(self):
        i,j,k = self.parent.extents
        if self.plane == "x":
            return j,k
        if self.plane == "y":
            return i,k
        if self.plane == "z":
            return i,j
        
    def data_source(self):
        # Link the appropriate view changer to the slice number change
        if self.plane == "x":
            return self.parent.volume_data[self.slice_number,:,:].T
        elif self.plane == "y":
            return self.parent.volume_data[:,self.slice_number,:].T
        elif self.plane == "z":
            return self.parent.volume_data[:,:,self.slice_number].T

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(),
                             height=100,width=100, show_label=False),
                        HGroup(
                            Item("slice_plane_visible"),
                            Item("slice_number", editor=RangeEditor(
                                    mode="slider",
                                    high_name = 'top_slice',
                                    low    = 0,
                                    format = "%i")),
                            show_labels=False),
                    show_labels=False,
                        ),
                    #width=100, height=200,
                    resizable=True
                    )


class Slices(HasTraits):
    extents = Property(Tuple)
    # x slice
    x_slice_window = Instance(ChacoSlice)
    x = CInt
    x_slice_plane_visible = CBool(True)
    x_slice_opacity = DelegatesTo("x_slice_window.slice_opacity")
    # y slice
    y_slice_window = Instance(ChacoSlice)
    y = CInt
    y_slice_plane_visible = CBool(True)
    y_slice_opacity = DelegatesTo("y_slice_window.slice_opacity")
    # z slice
    z_slice_window = Instance(ChacoSlice)
    z = CInt
    z_slice_plane_visible = CBool(True)
    z_slice_opacity = DelegatesTo("z_slice_window.slice_opacity")

    # Plotting
    volume_data = Array(value=np.zeros((50,50,50)))

    def __init__(self):
        # Force creation of slice windows
        self.x_slice_window
        self.y_slice_window
        self.z_slice_window
        # Enable communication between x,y,z and their windows
        self.sync_trait("x", self.x_slice_window,
                        alias="slice_number",mutual=True)
        self.sync_trait("x_slice_plane_visible", self.x_slice_window,
                        alias="slice_plane_visible",mutual=True)
        self.sync_trait("y", self.y_slice_window,
                        alias="slice_number",mutual=True)
        self.sync_trait("y_slice_plane_visible", self.y_slice_window,
                        alias="slice_plane_visible",mutual=True)
        self.sync_trait("z", self.z_slice_window,
                        alias="slice_number",mutual=True)
        self.sync_trait("z_slice_plane_visible", self.z_slice_window,
                        alias="slice_plane_visible",mutual=True)

    def set_volume(self, volume):
        self.volume_data = volume
        for sw, dim in zip(["x","y","z"], volume.shape):
            # reset the slices to the middle of the volume
            setattr(self,sw,dim/2)
            win = getattr(self,sw + "_slice_window")
            win.top_slice = dim
            win.reset_aspect_ratio()
        
    def _x_default(self):
        return self.extents[0]/2
    def _y_default(self):
        return self.extents[1]/2
    def _z_default(self):
        return self.extents[2]/2
    
    def _get_extents(self):
        return self.volume_data.shape
    
    def _x_slice_window_default(self):
        return ChacoSlice(
            plane = "x",
            parent=self)        
    
    def _y_slice_window_default(self):
        return ChacoSlice(
            plane = "y",
            parent=self)
    
    def _z_slice_window_default(self):
        return ChacoSlice(
            plane = "z",
            parent=self)

    def _origin_changed(self):
        self.renderer.origin = self.origin
        self.plot.request_redraw()

    traits_view = View(
                    HGroup(
                        Item("x_slice_window",style="custom"),
                        Item("y_slice_window",style="custom"),
                        Item("z_slice_window",style="custom"),
                        show_labels=False
                         ),
                    resizable=True
                  )

ChacoSlice.add_class_trait("parent", Instance(Slices))