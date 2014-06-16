#!/usr/bin/env python
import numpy as np
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn
from traitsui.group import ShadowGroup
from ..database.track_datasource import TrackDataSource
import sys

#
from .ui_extras import colormaps
from .sphere_browser import SphereBrowser
#from ..database.local_data import local_qsdrdb as local_dsis_trackdb
from ..aggregation.clustering_algorithms import QuickBundlesAggregator, FastKMeansAggregator
#from ..aggregation.region_clusters import RegionAggregator
from ..aggregation.region_labeled_clusters import RegionLabelAggregator
from ..streamlines.track_dataset import TrackDataset
from ..database.traited_query import Scan, Query
from traitsui.extras.checkbox_column import CheckboxColumn

#local_trackdb = local_dsis_trackdb
scan_table = TableEditor(
    columns =
    [   ObjectColumn(name="scan_id",editable=False),
        ObjectColumn(name="study",editable=False),
        ObjectColumn(name="scan_group",editable=False),
        ObjectColumn(name="software",editable=False),
        ObjectColumn(name="color_map"),
        CheckboxColumn(name="dynamic_color_clusters"),
        ObjectColumn(name="static_color")
    ],
    deletable  = True,
    auto_size  = True,
    show_toolbar = True
    )

class BrowserBuilder(HasTraits):
    # Instance to display queryable traits
    query_parameters = Instance(Query, ())
    # Where will the data come from
    data_source = Enum("Local Data", "UCSB MongoDB")
    local_json = File
    # Holds the results from the query
    results = List(Instance(Scan))
    browsers = List(Instance(SphereBrowser))
    aggregator = Enum("K Means","QuickBundles","Region Labels")
    a_query = Button(label="Search for Datasets")
    a_browser_launch = Button(label="Launch Sphere Browser")

    def _a_query_fired(self):
        if self.data_source == "UCSB MongoDB":
            print "MongoDB not configured"
        if self.data_source == "Local Data":
            matchnum = 1
            for dataspec in local_trackdb:
                if self.query_parameters.local_matches(dataspec):
                    dataspec.color_map = colormaps[matchnum]
                    self.results.append(dataspec)
                    matchnum += 1

    def _a_browser_launch_fired(self):
        if self.aggregator == "K Means":
            cl = FastKMeansAggregator()
        elif self.aggregator == "QuickBundles":
            cl = QuickBundlesAggregator()
        elif self.aggregator == "Region Labels":
            cl = RegionLabelAggregator()
        sb = SphereBrowser(aggregator=cl)
        tdatasets = []
        for res in self.results:
            print "Loading", res.pkl_path
            tdatasets.append(res.get_track_dataset())
        track_source = TrackDataSource(track_datasets=tdatasets)
        # Should we pass a list of colors to the aggregator?
        def wx_color_convert(wxc):
            return wxc[0]/255.,wxc[1]/255.,wxc[2]/255.
        if all([not d.dynamic_color_clusters for d in track_source.track_dataset_properties]):
            cl.subject_colors = [wx_color_convert(c.static_color) for c in track_source.track_dataset_properties]
        sb.set_track_source(track_source)
        sb.configure_traits()
        self.browsers.append(sb)

    # ----- UI Items
    traits_view = View(
                    HSplit(
                            VGroup(
                                Group(
                        Item("data_source",label="Data Source"),
                        Item("aggregator",label="Aggregation Algorithm")),
                                Group(
                        Item("query_parameters", style="custom"),
                            show_labels=False),
                        ),
                            VGroup(
                        Item("a_query"),
                        Item("a_browser_launch"),
                        Item(name="results",
                             editor=scan_table),

                        orientation="vertical",
                        show_labels=False
                            )
                    )
                )

if __name__=="__main__":
    bb=BrowserBuilder()
    bb.configure_traits()
