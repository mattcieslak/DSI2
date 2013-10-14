#!/usr/bin/env python
import numpy as np
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, on_trait_change, Button
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action
from traitsui.group import ShadowGroup
# Enthought library imports
from enable.api import Component, ComponentEditor
from traits.api import Float, HasTraits, Int, Instance
from traitsui.api import Item, Group, View

# Chaco imports
from chaco.api import create_line_plot, add_default_axes, add_default_grids, \
        OverlayPlotContainer, PlotLabel, create_scatter_plot, Legend
from chaco.tools.api import PanTool, ZoomTool, LegendTool, TraitsTool, DragZoom
from chaco.example_support import COLOR_PALETTE

from .cluster_ui import ClusterEditor
from ..streamlines.track_math import tracks_to_endpoints

# For computing the silhouette coefficient
from sklearn import metrics

identity = lambda x : x

size = (400, 400)
title = "Simple Line Plot"

class OverlappingPlotContainer(OverlayPlotContainer):
    def __init__(self, *args, **kws):
        super(OverlayPlotContainer, self).__init__(*args, **kws)

class AggregationEvaluator(HasTraits):
    step_size=Int(8)
    reduction_function = Function
    # Function to reduce the tracks to the form they were clustered in
    clust_editor = Instance(ClusterEditor)
    parameters = List
    param1_name  = Str
    param2_name  = Str("None",)
    param3_name  = Str("None")
    n_params     = Int(1)
    search_params = List
    a_calc = Button()
    plot = Instance(OverlappingPlotContainer)

    def __init__(self,**traits):
        super(AggregationEvaluator,self).__init__(**traits)
        self.clust_editor
        self.param1_name
        #self._setup_plots()

    def _plot_default(self):
        return OverlappingPlotContainer(padding=50, fill_padding=True,
                                     bgcolor="lightgray", use_backbuffer=True)

    def _parameters_default(self):
        return ["None"] + self.clust_editor.parameters

    def _param1_name_default(self):
        return self.clust_editor.parameters[0]

    @on_trait_change("param+")
    def param_selected(self):
        self.search_params = [ self.param1_name ]
        if not self.param2_name == "None": self.search_params.append(self.param2_name)
        if not self.param3_name == "None": self.search_params.append(self.param3_name)

    def silhouette_coefficient(self,tds):
        """ Computes the silhouette coefficient for the current set
        of clusters.
        """
        labels = np.zeros(len(tds.tracks))
        for clust in tds.clusters:
            labels[clust.indices] = clust.id_number
        if np.sum(labels>0) < 2 :return -2
        return metrics.silhouette_score(
                  self.reduction_function(tds.tracks)[labels > 0],
                  labels[labels>0], metric='euclidean')

    def setup_plots(self,*a):
        print "calculate..."
        #if len(self.search_params) == 1:
        self._single_param_plot()

    def _a_calc_fired(self):
        self.setup_plots()

    def _single_param_plot(self):
        """Creates series of Bessel function plots"""
        for component in self.plot.components:
            self.plot.remove(component)
        self.plot.request_redraw()
        plots = {}
        self.clust_editor.interactive=False
        eval_points = np.floor(np.linspace(2,11,self.step_size))
        original_parameter = getattr(self.clust_editor, self.param1_name)
        for tnum, tds in enumerate(self.clust_editor.track_sets):
            tds.interactive = False
            # Compute the sihlouette coefficient for each parameter value
            eval_results = []
            for eval_param in eval_points:
                setattr(self.clust_editor, self.param1_name, int(eval_param))
                try:
                    eval_results.append(self.silhouette_coefficient(tds))
                except Exception, e:
                    print e
                    eval_results.append(-2)
            _plot = create_line_plot((eval_points, np.array(eval_results)),
                                    color=tuple(COLOR_PALETTE[tnum]),
                                    width=2.0)
            if tnum == 0:
                value_mapper, index_mapper, legend = \
                    self._setup_plot_tools(_plot)
            else:
                self._setup_mapper(_plot, value_mapper, index_mapper)

            self.plot.add(_plot)
            plots[tds.name] = _plot

        # Add lines to the legend
        legend.plots = plots

        # Add the title at the top
        self.plot.overlays.append(PlotLabel("Sihlouette Coefficient",
                                       component=self.plot,
                                       font="swiss 16",
                                       overlay_position="top"))
        # Traits inspector tool
        self.plot.tools.append(TraitsTool(self.plot))

        # Set the cluster editor back to its original glory
        self.clust_editor.interactive = True
        setattr(self.clust_editor,self.param1_name,original_parameter)

    def _setup_plot_tools(self, plot):
        """Sets up the background, and several tools on a plot"""
        # Make a white background with grids and axes
        plot.bgcolor = "white"
        add_default_grids(plot)
        add_default_axes(plot)

        # Allow white space around plot
        plot.index_range.tight_bounds = False
        plot.index_range.refresh()
        plot.value_range.tight_bounds = False
        plot.value_range.refresh()

        # The PanTool allows panning around the plot
        plot.tools.append(PanTool(plot))

        # The ZoomTool tool is stateful and allows drawing a zoom
        # box to select a zoom region.
        zoom = ZoomTool(plot, tool_mode="box", always_on=False)
        plot.overlays.append(zoom)

        # The DragZoom tool just zooms in and out as the user drags
        # the mouse vertically.
        dragzoom = DragZoom(plot, drag_button="right")
        plot.tools.append(dragzoom)

        # Add a legend in the upper right corner, and make it relocatable
        legend = Legend(component=plot, padding=10, align="ur")
        legend.tools.append(LegendTool(legend, drag_button="right"))
        plot.overlays.append(legend)

        return plot.value_mapper, plot.index_mapper, legend

    def _setup_mapper(self, plot, value_mapper, index_mapper):
        """Sets up a mapper for given plot"""
        plot.value_mapper = value_mapper
        value_mapper.range.add(plot.value)
        plot.index_mapper = index_mapper
        index_mapper.range.add(plot.index)


    # ----- UI Items
    traits_view = View(
                        HGroup(
                            Item("step_size"),
                            Item("param1_name",editor=EnumEditor(name="parameters")),
                            Item("param2_name",editor=EnumEditor(name="parameters")),
                            Item("param3_name",editor=EnumEditor(name="parameters")),
                            Item("a_calc",name="CALCULATE",show_label=False),
                            Item('plot', editor=ComponentEditor(size=size),
                                show_label=False),
                           orientation="vertical"
                        )
                )

    # Puts a ClusterEditor side-by-side to the AggregationEvaluator
    browser_view = View(
                    HSplit(
                        Item("clust_editor",style="custom",show_label=False),
                        VGroup(
                            Item("step_size"),
                            Item("param1_name",editor=EnumEditor(name="parameters")),
                            Item("param2_name",editor=EnumEditor(name="parameters")),
                            Item("param3_name",editor=EnumEditor(name="parameters")),
                            Item("a_calc",name="CALCULATE",show_label=False),
                            Item('plot', editor=ComponentEditor(size=size), show_label=False),
                            ),
                        show_labels=False
                         )
                    )

def flat_tep(trks):
    return tracks_to_endpoints(trks).reshape(trks.shape[0], 6)

if __name__ == "__main__":

    import cPickle as pickle
    from mayavi import mlab
    from dsi2.streamlines.track_dataset import track_dataset
    from dsi2.aggregation.cluster_ui import AlgorithmParameterHandler
    from mayavi.tools.mlab_scene_model import MlabSceneModel
    fop = open("data/example_trks.pkl",'r')
    trks1, trks2, trks3 = pickle.load(fop)
    fop.close()

    from dsi2.aggregation.clustering_algorithms import FastKMeansAggregator

    # Make a window for plotting, give it to a cluster editor
    sp  = FastKMeansAggregator(scene3d=MlabSceneModel(),
                                interactive=False)
    sp.set_track_sets( [track_dataset(trks1,name="1565A",
                            scene3d=sp.scene3d, interactive=sp.interactive),
                        TrackDataset(trks2,name="1565B",
                            scene3d=sp.scene3d, interactive=sp.interactive),
                        TrackDataset(trks3,name="1565C",
                            scene3d=sp.scene3d, interactive=sp.interactive)]
                     )


    ce = AggregationEvaluator(clust_editor=sp, reduction_function=flat_tep)
    ce.configure_traits(view="browser_view")