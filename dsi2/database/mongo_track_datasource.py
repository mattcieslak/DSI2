#!/usr/bin/env python
from traits.api import HasTraits, List, Instance, Bool, Str, File, Dict
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
    client = Instance(pymongo.MongoClient)
    db_name = Str("dsi2")
    db = Instance(pymongo.database.Database)
    
    def __init__(self,**traits):
        super(MongoTrackDataSource,self).__init__(**traits)
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

    def get_subjects(self):
        """ In this case, the user should supply a set of scan ids
        to be included in the coordinate search."""
        #result = self.db.scans.find(fields=[ "subject_id" ])
        #return sorted(list(set(
        #    [rec["subject_id"] for rec in result])))
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
            return [qresults[scan] for scan in self.scan_ids]
        
        # Build new TrackDatasets with streamlines included
        print "\t+ Querying the streamlines collection"
        qresults = {}
        for sl_id_result in coord_query["result"]:
            scan_id = sl_id_result['_id']
            sl_ids = sl_id_result["sl_ids"]
            print "\t\t++ %s has %s streamlines" % (scan_id, len(sl_ids))
            original_track_dataset = self.tds_lut[scan_id]
            
            # Collect these with a simple .find() and loop through
            # the reslts to preserve the data and sl_id pairs. Someday,
            # accomplish this with the aggregate framework
            streamlines = []
            found_sl_ids = []
            for sl in self.db.streamlines.find(
                {"scan_id":scan_id, "sl_id":{"$in":sl_ids}}):
                streamlines.append(sl['data'])
                found_sl_ids.append(sl["sl_id"])
            tracks = np.array([pickle.loads(s) for s in streamlines])
            print "\t\t+++ query returned %d streamlines" % len(tracks)
            original_track_indices=np.array(found_sl_ids)
            survivors = tracks[every-1::every] if tracks.size > 0 else None
            connections = original_track_dataset.connections[original_track_indices][every-1::every] \
                if original_track_dataset.connections.size > 0 else np.array([])
                
            qresults[scan_id] = TrackDataset(
                tracks=survivors, 
                header=original_track_dataset.header,
                properties=original_track_dataset.properties, 
                connections=connections, 
                original_track_indices=original_track_indices)

        return [qresults[scan] for scan in self.scan_ids]

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

        # Get the scan_ids that contain these connections
        cons = [str(c) for c in connection_id]
        result = self.db.connections.find( { "con_id": { "$in": cons }, "atlas_id": self.atlas_id }, [ "scan_id" ] )
        scans = set([rec["scan_id"] for rec in result])

        # Get the streamlines for each scan and build a list of TrackDatasets
        # TrackDataSource returns a TrackDataset for each scan, even if there are no matching tracks, so do the same here.
        datasets = []
        result = self.db.scans.find( fields=[ "scan_id" ] )
        all_scans = [rec["scan_id"] for rec in result]
        for scan in all_scans:
            tracks = None
            streamlines = []
            connections = []
            if scan in scans:
                result = self.db.connections.find( 
                    { "con_id": { "$in": cons }, "scan_id": scan, "atlas_id": self.atlas_id }, 
                    [ "sl_ids", "con_id" ] )
                sl_cons = {}
                for rec in result:
                    for sl in rec["sl_ids"]:
                        streamlines.append(sl)
                        sl_cons[sl] = int(rec["con_id"])
                streamlines = sorted(list(set(streamlines)))

                # downsampling
                streamlines = streamlines[every-1::every]

                connections = [sl_cons[sl] for sl in streamlines]

                result = self.db.streamlines.find( { "sl_id": { "$in": streamlines }, "scan_id": scan }, [ "data" ] )
                tracks = [pickle.loads(rec["data"]) for rec in result]

            result = db.scans.find( { "scan_id": scan } )

            if result.count() > 1:
                logger.warning("Multiple records found for scan %s. Using first record.", scan)

            tds = self.build_track_dataset(result=result[0], tracks=tracks, 
                                           original_track_indices=np.array(streamlines), connections=np.array(connections))
            datasets.append(tds)

        return datasets

    def change_atlas(self, query_specs):
        """ Sets the .connections for each TrackDataset to be loaded from the path
        specified in its properties.atlases
        """

        self.atlas_id = None
        result = self.db.atlases.find( { "name": query_specs["name"] } )
        match_count = 0
        for rec in result:
            atlas_dict = rec["parameters"]
            atlas_dict["name"] = query_specs["name"]
            if dictmatch(query_specs, atlas_dict):
                self.atlas_id = rec["_id"]
                match_count += 1

        # Mimic TrackDataSource behavior for now.
        if match_count != 1:
            raise ValueError("Query did not return exactly one match")

        print "\t+ Setting new atlas in TrackDataSource"
        new_labels = []
        result = self.db.scans.find(fields=[ "scan_id" ])
        scans = [rec["scan_id"] for rec in result]
        for scan in scans:
            match = self.db.streamline_labels.find_one( { "scan_id": scan, "atlas_id": self.atlas_id }, [ "con_ids" ] )
            if match != None:
                match = [np.array(match["con_ids"])]
                new_labels += match

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
