#!/usr/bin/env python
import numpy as np
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn, OKButton, CancelButton
from traitsui.group import ShadowGroup
from ..database.track_datasource import TrackDataSource
from ..database.mongo_track_datasource import MongoTrackDataSource
from ..database.local_data import get_local_data
import sys

#
from .ui_extras import colormaps
from .sphere_browser import SphereBrowser
#from ..database.local_data import local_qsdrdb as local_dsis_trackdb
from ..aggregation.clustering_algorithms import QuickBundlesAggregator, FastKMeansAggregator
#from ..aggregation.region_clusters import RegionAggregator
from ..aggregation.region_labeled_clusters import RegionLabelAggregator
from ..streamlines.track_dataset import TrackDataset
from ..database.traited_query import Scan, MongoScan, Query
from traitsui.extras.checkbox_column import CheckboxColumn
import pymongo

scan_table = TableEditor(
    columns =
    [   ObjectColumn(name="scan_id",editable=False),
        ObjectColumn(name="study",editable=False),
        ObjectColumn(name="scan_group",editable=False),
        ObjectColumn(name="software",editable=False)
    ],
    deletable  = True,
    auto_size  = True,
    show_toolbar = True
    )

class BrowserBuilder(HasTraits):
    # Instance to display queryable traits
    query_parameters = Instance(Query, ())
    # Where will the data come from
    data_source = Enum("Local Data", "MongoDB")
    local_json = File
    local_scans = List([])
    # Connection parameters for mongodb
    client = Instance(pymongo.MongoClient)
    mongo_host = Str("127.0.0.1")
    mongo_port = Int(27017)
    db_name=Str("dsi2")
    # Holds the results from the query
    results = List(Instance(Scan))
    browsers = List(Instance(SphereBrowser))
    aggregator = Enum("K Means","QuickBundles","Region Labels")
    a_query = Button(label="Search for Datasets")
    a_browser_launch = Button(label="Launch Sphere Browser")

    def _a_query_fired(self):
        if self.data_source == "MongoDB":
            self.mongo_find_datasets()
        elif self.data_source == "Local Data":
            self.local_find_datasets()
            
    def _client_default(self):
        try:
            client = pymongo.MongoClient("mongodb://%s:%d/" %(
                 self.mongo_host, self.mongo_port))
            return client
        except Exception, e:
            try:
                print "Constructing vanilla client"
                return pymongo.MongoClient()
            except:
                return
        
    def mongo_find_datasets(self):
        """ Queries a mongodb instance to find scans that match
        the study
        """
        collection = self.client[self.db_name]["scans"]
        query = self.query_parameters.mongo_query()
        results = list(collection.find(query))
        print "found %d results" % len(results)
        self.results = [
            MongoScan(mongo_result=res) for res in results ]
        
            
    def local_find_datasets(self):
        self.local_scans = get_local_data(self.local_json)
        matchnum = 1
        matches = []
        if self.query_parameters.study == "":
            return self.local_scans
        for dataspec in self.local_scans:
            if self.query_parameters.local_matches(dataspec):
                dataspec.color_map = colormaps[matchnum]
                self.results.append(dataspec)
                matchnum += 1
                matches.append(dataspec)
        self.results = matches
        
    def get_datasource(self): 
        # Create a track source
        if self.data_source == "MongoDB":
            track_source = self.get_mongo_datasource()
        else:
            track_source = self.get_local_datasource()
        return track_source

    def _a_browser_launch_fired(self):
        # Create a track source
        track_source = self.get_datasource()
        # set it to a sphere browser
        sb = SphereBrowser()
        sb.set_track_source(track_source)
        sb.configure_traits()
        self.browsers.append(sb)
        
    def get_local_datasource(self):
        tdatasets = []
        for res in self.results:
            tdatasets.append(res.get_track_dataset())
        return TrackDataSource(track_datasets=tdatasets)
        
    def get_mongo_datasource(self):
        """ Instead of handling the 
        """
        scan_ids = [sc.scan_id for sc in self.results]
        print "only going to query for",", ".join(scan_ids)
        return MongoTrackDataSource(scan_ids=scan_ids,
                                    db_name=self.db_name,
                                    client=self.client)

    # ----- UI Items
    local_file_group = Group(
        Item("local_json"),
        show_border=True,
        label="Local data source",
        visible_when="data_source=='Local Data'"
    )
    mongo_file_group = Group(
        Item("mongo_host"),
        Item("mongo_port"),
        Item("db_name"),
        show_border=True,
        label="MongoDB Data source",
        visible_when="data_source=='MongoDB'"
    )
        
    dsource_view = View(
        HSplit(
                VGroup(
                    Group(
            Item("data_source",label="Data Source"),
            local_file_group,
            mongo_file_group,
            ),
                    Group(
            Item("query_parameters", style="custom"),
            Item("a_query"),
                show_labels=False),
            ),
            VGroup(
                Item(name="results",
                 editor=scan_table),
            show_labels=False
            )
        ),
        title="Select Data Source",
        kind="modal",
        buttons = [OKButton, CancelButton]
    )
    traits_view = View(
        HSplit(
                VGroup(
                    Group(
            Item("data_source",label="Data Source"),
            local_file_group,
            mongo_file_group,
            ),
                    Group(
            Item("query_parameters", style="custom"),
            Item("a_query"),
                show_labels=False),
            ),
            VGroup(
                Item("a_browser_launch"),
                Item(name="results",
                 editor=scan_table),
            show_labels=False
            )
        ),
        title="Select Data Source"
    )
if __name__=="__main__":
    bb=BrowserBuilder()
    bb.configure_traits()
