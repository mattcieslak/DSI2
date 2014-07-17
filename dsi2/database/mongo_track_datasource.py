#!/usr/bin/env python
from traits.api import HasTraits, List, Instance, Bool, Str, File, Dict, Int
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn
from ..streamlines.track_dataset import TrackDataset
from ..database.traited_query import MongoScan
from .track_datasource import TrackDataSource
from mayavi.tools.mlab_scene_model import MlabSceneModel
from .traited_query import Scan
#from .local_data import get_local_data
import numpy as np
import os
from collections import defaultdict
import operator
import pymongo
from bson.binary import Binary
import cPickle as pickle

import dsi2.config

#connection = pymongo.MongoClient()
#db = connection.dsi2

def dictmatch(qdict,ddict):
    return all([ ddict.get(key,"") == val for key,val in qdict.iteritems() ] )

# Inherit from TrackDataSource to work with 
class MongoTrackDataSource(TrackDataSource):
    #  Holds a list of objects sporting
    #  a .tracks_at_ijk() function.
    track_datasets = List
    track_dataset_properties = List(Instance(Scan))
    #scene3d=Instance(MlabSceneModel)
    interactive=Bool(False)
    atlas_name = Str("None")
    # if the atlas_name gets changed and new vectors need to be loaded,
    # only do it before a query is requested
    needs_atlas_update = Bool(False)
    json_source = File("")
    label_cache = List(List(Dict))
    render_tracks = Bool(False)
    atlas_id = None
    scan_ids = List
    # A cursor object
    client = Instance(pymongo.MongoClient,transient=True)
    mongo_host = Str("127.0.0.1")
    mongo_port = Int(27017)
    db_name=Str("dsi2")
    db = Instance(pymongo.database.Database,transient=True)
    
    def __init__(self,**traits):
        super(MongoTrackDataSource,self).__init__(**traits)
        """
        The critical pieces of information when constructing this class
        are the scan_ids, client, and db_name.
        
        Parameters:
        -----------
        scan_ids: list of str
          The "scan_id" field from the scans that are to be retrieved
          during the spatial query
        host:str
          Hostname for mongodb instance
          
        db_name:str
          The name of the database to use from ``client``.  Typically
          this is "dsi2", but for unit tests can be set to "dsi2_test"
        """
        # if track_datasets is not passed explicitly
        """
        Creates TrackDatasets that have no .tracks or .connections.
        """
        properties = []
        for scan_id in self.scan_ids:
            results = list(self.db.scans.find({"scan_id":scan_id}))
            if len(results) != 1:
                raise ValueError("scans collection does not contain exactly 1 " + scan+id)
            result = results[0]
            properties.append(MongoScan(mongo_result=result))
        self.track_dataset_properties = properties
        self.track_datasets = [
          TrackDataset(properties=props,header=props.header, 
                       connections=np.array([]),tracks=np.array([])) \
           for props in self.track_dataset_properties
        ]
        # A mapping to quickly assign stuff to TrackDataSets based on scan_id
        self.tds_lut = dict([
            (tds.properties.scan_id,tds) for tds in self.track_datasets ]
        )
    
    def _db_default(self):
        return self.client[self.db_name]
    
    def _client_default(self):
        return self.__get_client()
    
    def __get_client(self):
        try:
            client = pymongo.MongoClient("mongodb://%s:%d/" %(
                 self.mongo_host, self.mongo_port))
            return client
        except Exception, e:
            print "Constructing vanilla client"
            return pymongo.MongoClient()
        
    def new_mongo_connection(self):
        """
        To initialize a new connection to mongod, call this function
        """
        self.client = self.__get_client()
        self.db = self._db_default()

    def get_subjects(self):
        """ In this case, the user should supply a set of scan ids
        to be included in the coordinate search."""
        return self.scan_ids
    

    def query_ijk(self, ijk, every=0, fetch_streamlines=True):
        if every < 0:
            raise ValueError("every must be >= 0.")

        if every == 0:
            every = 1

        # Get the scan_ids that contain these coordinates
        coords = [str(c) for c in ijk]
        print "Query"
        print "======"        
        print "\t+ searching %d coordinates" % len(coords)
        coord_query = self.db.coordinates.aggregate( 
            # Find the coordinates for each subject
            [
                {
                  "$match":{
                    "scan_id":{"$in":self.scan_ids},
                    "ijk":{"$in":coords}
                  }
                },
                {"$project":{"scan_id":1,"sl_id":1}},
                {"$unwind":"$sl_id"},
                {"$group":{"_id":"$scan_id", "sl_ids":{"$addToSet":"$sl_id"}}}
            ]
        )
        
        # If the search failed, do nothing and exit
        if not coord_query["ok"] == 1.:
            print "\t+ WARNING: Coordinate query failed"
            return []
        
        # If we don't need streamlines, just return the
        # subsetted TrackDatasets that presumably have .connections
        if not fetch_streamlines:
            print "\t+ Streamlines are not required, so subsetting .connections"
            qresults = {}
            for result in coord_query['result']:
                qresults[result['_id']] = self.tds_lut[result["_id"]].subset(
                                        result["sl_ids"], every=every)
            print "+ Done."
            return [qresults[scan] if scan in qresults else self.tds_lut[scan].subset([]) for \
                       scan in self.scan_ids]
        
        # This gives a list of dictionaries. make a regular dictionary
        subject_sl_ids = dict([ (res["_id"], res["sl_ids"]) for res in coord_query["result"] ])
        return self.__datasets_from_connection_ids( subject_sl_ids )

    def __datasets_from_connection_ids(self, subj2id_map,every=1):
        """ takes a dictionary {subject_id:list of streamline_ids,...} and 
        returns a list of TrackDatasets in the correct order
        """
        # Build new TrackDatasets with streamlines included
        print "\t+ Querying the streamlines collection"
        query = {"$or":[ {"scan_id":scan_id, "sl_id":{"$in":sl_ids }} \
                         for scan_id,sl_ids in subj2id_map.iteritems() ]}
        print query
        
        # Collect a list of binary data streamlines from each subject
        strln_query = self.db.streamlines.aggregate(
            [
                {"$match":query},
                {"$group":
                  {
                   "_id":"$scan_id",
                   "sl_data":{"$push":"$data"}, 
                   "sl_ids":{"$push":"$sl_id"}
                  }
                }
            ]
        )
        # This gives a list of dictionaries. make a regular dictionary
        subject_strlns = dict([ (res["_id"], res) for res in strln_query["result"] ])

        qresults = []
        for scan_id in self.scan_ids:
            original_track_dataset = self.tds_lut[scan_id]
            # If the scan is not included in the results:
            if not scan_id in subject_strlns:
                qresults.append( 
                    TrackDataset(
                        tracks=np.array([]),
                        header=original_track_dataset.header,
                        properties=original_track_dataset.properties, 
                        connections=np.array([]), 
                        original_track_indices=np.array([]))
                )
                print "\t\t++ %s has %s streamlines" % (scan_id, len(sl_ids))
                continue
            
            # If the scan is included in the results
            subject_result = subject_strlns[scan_id]
            scan_id = subject_result['_id']
            sl_ids = subject_result["sl_ids"]
            print "\t\t++ %s has %s streamlines" % (scan_id, len(sl_ids))
            # get data from the result for this subject
            streamlines = subject_result["sl_data"]
            found_sl_ids = subject_result["sl_ids"]
            # If user requests downsampling, do it now
            if every > 1:
                streamlines = streamlines[::every]
                found_sl_ids = found_sl_ids[::every]
            # de-serialize the streamline data
            tracks = np.array([pickle.loads(s) for s in subject_result["sl_data"]])
            print "\t\t+++ query returned %d streamlines" % len(tracks)
            original_track_indices=np.array(found_sl_ids)
            connections = original_track_dataset.connections[original_track_indices] \
                if original_track_dataset.connections.size > 0 else np.array([])
                
            qresults.append( 
                TrackDataset(
                    tracks=tracks,
                    header=original_track_dataset.header,
                    properties=original_track_dataset.properties, 
                    connections=connections, 
                    original_track_indices=original_track_indices)
            )
            
        # Check that all scans are accounted for
        assert len(qresults) == len(self.scan_ids)
        
        return qresults

        
    def query_connection_id(self,connection_id,every=0):
        """
        Subsets the track datasets so that only streamlines labeled as
        `region_pair_id`
        """
        if every < 0:
            raise ValueError("every must be >= 0.")

        if every == 0:
            every = 1

        if type(connection_id) == int:
            connection_id = np.array([connection_id])
        elif type(connection_id) != np.ndarray:
            connection_id = np.array(connection_id)
        
        region_indices = {}
        for scan_id in self.scan_ids:
            original_track_dataset = self.tds_lut[scan_id]
            region_indices[scan_id] = np.flatnonzero(
                np.in1d(original_track_dataset.connections,connection_id)).tolist()

        return self.__datasets_from_connection_ids(region_indices)

    def change_atlas(self, query_specs):
        """ Sets the .connections for each TrackDataset to be loaded from the path
        specified in its properties.atlases
        """

        self.atlas_id = None
        atlas_name = query_specs["name"]
        qdict = dict(
                 [("name", atlas_name)] + 
                 [("parameters." + k, v) for k,v in query_specs.iteritems() \
                                     if k != "name"])

        result = [rec for rec in self.db.atlases.find( qdict )]
        if not len(result) == 1:
            raise ValueError("Query was unable to find a single matching atlas." \
                             "found %d" %len(result))
        
        self.atlas_id = rec["_id"]

        print "\t+ Setting new atlas in TrackDataSource"
        new_labels = []
        for scan in self.scan_ids:
            match = self.db.streamline_labels.find_one( { "scan_id": scan, "atlas_id": self.atlas_id }, [ "con_ids" ] )
            match = np.array(match["con_ids"])
            self.tds_lut[scan].set_connections(match)
            new_labels.append( match )

        return new_labels

    def load_label_data(self):
        # What are the unique atlas names?
        # Find all the unique atlas ids.
        result = self.db.scans.find( fields=[ "atlases" ] )
        atlas_ids = set([])
        for rec in result:
            atlas_ids.update([a for a in rec["atlases"]])
        atlas_ids = list(atlas_ids)
        # Map the ids to names
        result = self.db.atlases.find( { "_id": { "$in": atlas_ids } }, [ "name" ] )
        atlases = list(set([rec["name"] for rec in result]))
        print "\t+ Found %d unique atlases" % len(atlases)

        label_lut = {}
        for atlas_name in atlases:
            label_lut[atlas_name] = defaultdict(set)
            result = self.db.atlases.find( { "name": atlas_name } )
            for rec in result:
                for prop, propval in rec["parameters"].iteritems():
                    if prop == "notes": continue
                    label_lut[atlas_name][prop].update( (propval,) )

        varying_properties = {}
        for atlas_name in atlases:
            varying_properties[atlas_name] = {}
            for propname, propvals in label_lut[atlas_name].iteritems():
                # Are there multiple values this property can take on?
                if len(propvals) <= 1: continue
                varying_properties[atlas_name][propname] = sorted(list(propvals))
        print varying_properties
        return varying_properties
