#!/usr/bin/env python
import sys, json, os
from time import time
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File, Int, Bool, cached_property,Event, Float
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn, SetEditor
from dsi2.ui.volume_slicer import SlicerPanel3D
from chaco.api import ArrayPlotData, Plot, VPlotContainer, jet
from chaco.tools.api import RangeSelectionOverlay, RangeSelection2D, ZoomTool
from enable.component_editor import ComponentEditor
from dsi2.streamlines.track_dataset import TrackDataset
import pandas as pd
from datetime import datetime
import numpy as np


class ParamRangeSelector(ZoomTool):
    xmin = Float(0)
    xmax = Float(0)
    ymin = Float(0)
    ymax = Float(0)
    params_selected = Int(0)
        
    def _end_select(self, event):
        self._screen_end = (event.x, event.y)
    
        start = np.array(self._screen_start)
        end = np.array(self._screen_end)
    
        if sum(abs(end - start)) < self.minimum_screen_delta:
            self._end_selecting(event)
            event.handled = True
            return
    
        low, high = self._map_coordinate_box(self._screen_start, self._screen_end)
    
        self.xmin, self.ymin = low
        self.xmax, self.ymax = high
        self.params_selected += 1
        return self._end_selecting(event)

class TrajectoryFilter(HasTraits):
    output_prefix = Str("")
    vslicer = Instance(SlicerPanel3D)
    plot_data = ArrayPlotData
    plot = Instance(Plot)
    plot_container = Instance(VPlotContainer)
    # User selected ranges
    selection_tool = Instance(ParamRangeSelector)
    params_selected = DelegatesTo("selection_tool")
    xmin = DelegatesTo("selection_tool")
    xmax = DelegatesTo("selection_tool")
    ymin = DelegatesTo("selection_tool")
    ymax = DelegatesTo("selection_tool")
    matching_streamlines = Int
    b_render_vol = Button(label="Render Volume")
    
    needs_classification = Bool(False)
    
    active_tracks = Instance(TrackDataset)
    
    # For classifying streamlines in the viewer
    b_plausible = Button(label="PLAUSIBLE")
    b_implausible = Button(label="IMPLAUSIBLE")
    
    n_plausible= Int
    n_implausible=Int
    
    scene3d = DelegatesTo("vslicer")
    reference_volume = File
    measurements = List()
    nbins=Int(100)
    x_measure = Str
    y_measure = Str
    b_pick_random = Button(label="Pick Random")
    n_streamlines = Int(50)
    traits_view = View(
        HSplit(Group(
            Group(
                Item("n_streamlines"), Item("b_pick_random"),
                Item("y_measure", editor=EnumEditor(name="measurements")),
                Item("x_measure", editor=EnumEditor(name="measurements")),
                orientation="vertical"
            ), Group(
                Item('plot_container', editor=ComponentEditor(),
                             height=200,width=200, show_label=False),
                show_labels=False),
            Group(Item("xmin"), Item("xmax"),orientation="horizontal"),            
            Group(Item("ymin"), Item("ymax"),orientation="horizontal"),
            Group(Item("n_plausible"), Item("n_implausible"),orientation="horizontal"),
            Group(Item("matching_streamlines")),
            
            orientation="vertical"
            ),
            Group(
                Group(
                    Item("b_plausible",enabled_when="needs_classification"), 
                    Item("b_implausible",enabled_when="needs_classification"),
                    show_border=True,
                    label="Streamline Classification",
                    show_labels=False,
                    orientation="horizontal",
                    springy=True
                    ),
                Item("vslicer",style="custom"),
                show_labels=False
                ),
            show_labels=False
        )
    )
    
    def __init__(self,streamline_properties,**traits):
        super(TrajectoryFilter,self).__init__(**traits)
        self.vslicer
        self.vslicer.reference_volume = self.reference_volume
        self.streamline_properties = streamline_properties
        self.measurements = sorted(self.streamline_properties.keys()) + [
            "center_of_mass_x", "center_of_mass_y", "center_of_mass_z"]
        del self.measurements[self.measurements.index("center_of_mass")]
        self.classified_streamlines = None
        
        
    def get_streamline_data(self,measurename):
        if measurename in self.streamline_properties:
            return self.streamline_properties[measurename]
        if measurename.startswith("center_of_mass"):
            dim = measurename[-1]
            if dim == "x":
                return self.streamline_properties["center_of_mass"][:,0]
            if dim == "y":
                return self.streamline_properties["center_of_mass"][:,1]
            if dim == "z":
                return self.streamline_properties["center_of_mass"][:,2]
            
        
    def add_to_classified(self,plausibility):
        self.needs_classification = False
        df = []
        for sl_id in self.active_tracks.original_track_indices:
            row = dict([(k,v[sl_id]) for k,v in self.streamline_properties.iteritems()])
            row["plausible"] = plausibility
            row["streamline_id"] = sl_id
            df.append(row)
        if self.classified_streamlines is None:
            self.classified_streamlines = pd.DataFrame(df)
        else:
            self.classified_streamlines = pd.concat(
                [self.classified_streamlines, pd.DataFrame(df)],
                                               ignore_index=True)
            
        self.n_plausible = (self.classified_streamlines.plausible==1).sum()
        self.n_implausible = (self.classified_streamlines.plausible==0).sum()
        
        
    def _b_plausible_fired(self):
        self.add_to_classified(1)
        
    def _b_implausible_fired(self):
        self.add_to_classified(0)
        
    @on_trait_change("params_selected")
    def select_streamlines(self):
        print "selection finished"
        self.needs_classification = True
        x_meas = self.get_streamline_data(self.x_measure)
        y_meas = self.get_streamline_data(self.y_measure)
        ok_indices = np.flatnonzero(
            np.logical_and(
                  np.logical_and(x_meas >= self.xmin, x_meas <= self.xmax),
                  np.logical_and(y_meas >= self.ymin, y_meas <= self.ymax)
            )
        )
        self.matching_streamlines = len(ok_indices)
        if self.matching_streamlines == 0:
            return

        # Choose some of the streamlines for plotting
        if self.matching_streamlines > self.n_streamlines:
            ok_indices = np.random.choice(ok_indices, self.n_streamlines)
        
       # clear the old glyphs 
        self.scene3d.disable_render=True
        while len(self.scene3d.scene.mayavi_scene.children) > 2:
            self.scene3d.scene.mayavi_scene.children[-1].remove()
        self.scene3d.disable_render=False
        
        # render the tracks
        self.active_tracks = self.track_dataset.subset(ok_indices)
        self.active_tracks.dynamic_color_clusters = False
        self.active_tracks.representation = "Tube"
        self.active_tracks.static_color = "magenta"
        self.active_tracks.render_tracks = True
        self.active_tracks.draw_tracks()
        
        
        
    def _plot_container_default(self):
        # for image plot
        img, xvals,yvals = self.get_plot_data()
        self.plot_data = ArrayPlotData(imagedata=img)
        self.plot = Plot(self.plot_data)
        self.renderer = self.plot.img_plot(
            "imagedata", colormap=jet, name="plot1",origin="bottom left")[0]
        self.selection_tool = ParamRangeSelector(
                                                 component=self.plot,
                                                  tool_mode="box", always_on=True)
        self.plot.overlays.append(self.selection_tool)
        self.memo = {}
        return VPlotContainer(self.plot,padding_left=10,
                                                           padding_right=10)
        
    def _vslicer_default(self):
        sp = SlicerPanel3D(sphere_visible=False)
        sp.reference_volume = self.reference_volume
        return sp
    
    def get_plot_data(self):
        if self.x_measure == "" or self.y_measure == "":
            return np.zeros((self.nbins,self.nbins)), (0,99), (0,99)
        hist,x,y = np.histogram2d(
            self.get_streamline_data(self.x_measure),
            self.get_streamline_data(self.y_measure),
            self.nbins)
        return hist,(x[0],x[-1]), (y[0],y[-1])
    
    @on_trait_change("x_measure,y_measure")
    def update_plot(self):
        img, xlim, ylim = self.get_plot_data()
        self.plot.data.set_data("imagedata",img.T)
        ranges = self.plot.range2d.sources[0]
        ranges.set_data(
                                  np.linspace(xlim[0],xlim[1], self.nbins), 
                                  np.linspace(ylim[0],ylim[1], self.nbins))
        self.plot.y_axis.title = self.y_measure
        self.plot.x_axis.title = self.x_measure
        self.plot.request_redraw()
        
    def _b_render_vol_fired(self):
        self.vslicer.render_volume()
    
    def set_track_dataset(self,tds):
        self.track_dataset = tds
        self.track_dataset.scene3d = self.scene3d
        
    def save(self):
        outfile = self.output_prefix + datetime.now().isoformat() + ".csv"
        self.classified_streamlines.to_csv(outfile)
        print "saved", outfile
                    
