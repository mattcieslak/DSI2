#!/usr/bin/env python
from traits.api import HasTraits, List, Instance, Bool, Str, File, Dict
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn
from ..streamlines.track_dataset import TrackDataset
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
        """
        The critical pieces of information when constructing this class
        are the scan_ids, client, and db_name.
        
        Parameters:
        -----------
        scan_ids: list of str
          The "scan_id" field from the scans that are to be retrieved
          during the spatial query
          
        client:pymongo.MongoClient
          A connection to the MongoDB instance
          
        db_name:str
          The name of the database to use from ``client``.  Typically
          this is "dsi2", but for unit tests can be set to "dsi2_test"
        """
        # if track_datasets is not passed explicitly
        #if not self.track_datasets:
            # Load from a json_source
            #if self.json_source:
            #self.track_datasets = [ d.get_track_dataset()  for d in \
                #get_local_data(self.json_source) ]
        # grab the properties from each loaded TrackDataset
        #self.track_dataset_properties = \
            #[tds.properties for tds in self.track_datasets]
    
    def _db_default(self):
        return self.client[self.db_name]

    def get_subjects(self):
        """ In this case, the user should supply a set of scan ids
        to be included in the coordinate search."""
        #result = self.db.scans.find(fields=[ "subject_id" ])
        #return sorted(list(set(
        #    [rec["subject_id"] for rec in result])))
        return self.scan_ids
    
    def set_render_tracks(self,visibility):
        self.render_tracks = visibility

    def __len__(self):
        """ Thi should reflect only the scan_ids inside 
        self.scan_ids.  
        """
        #return self.db.scans.count()
        return len(self.scan_ids)
    
    def build_track_dataset(self,result,tracks,original_track_indices,connections):
        header = pickle.loads(result["header"])

        properties = Scan()
        properties.scan_id = result["scan_id"]
        properties.subject_id = result["subject_id"]
        properties.scan_gender = result["gender"]
        properties.scan_age = result["age"]
        properties.study = result["study"]
        properties.scan_group = result["group"]
        properties.smoothing = result["smoothing"]
        properties.cutoff_angle = result["cutoff_angle"]
        properties.qa_threshold = result["qa_threshold"]
        properties.gfa_threshold = result["gfa_threshold"]
        properties.length_min = result["length_min"]
        properties.length_max = result["length_max"]
        properties.institution = result["institution"]
        properties.reconstruction = result["reconstruction"]
        properties.scanner = result["scanner"]
        properties.n_directions = result["n_directions"]
        properties.max_b_value = result["max_b_value"]
        properties.bvals = result["bvals"]
        properties.bvecs = result["bvecs"]
        properties.label = result["label"]
        properties.trk_space = result["trk_space"]

        tds = TrackDataset(tracks=tracks, header=header, original_track_indices=original_track_indices, 
                           properties=properties, connections=connections)
        tds.render_tracks = self.render_tracks
        return tds

    def query_ijk(self,ijk,every=0):
        if every < 0:
            raise ValueError("every must be >= 0.")

        if every == 0:
            every = 1

        # Get the scan_ids that contain these coordinates
        coords = [str(c) for c in ijk]
        result = self.db.coordinates.find( { "ijk": { "$in": coords },
                                        "scan_id":{"$in":self.scan_ids}
                                       }, 
                                      [ "scan_id" ] )
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
                result = self.db.coordinates.find( { "ijk": { "$in": coords }, "scan_id": scan }, [ "sl_id" ] )
                streamlines = sorted(list(set(reduce(operator.add, [rec["sl_id"] for rec in result]))))

                # downsampling
                streamlines = streamlines[every-1::every]

                # If we've already loaded an atlas, build a connections list.
                if self.atlas_id != None:
                    connections = self.db.connections2.find_one( { "scan_id": scan, "atlas_id": self.atlas_id }, [ "con_ids" ] )
                    connections = connections["con_ids"]
                    connections = [connections[sl] for sl in streamlines]

                result = self.db.streamlines.find( { "sl_id": { "$in": streamlines }, "scan_id": scan }, [ "data" ] )
                tracks = [pickle.loads(rec["data"]) for rec in result]

            result = self.db.scans.find( { "scan_id": scan } )

            if result.count() > 1:
                logger.warning("Multiple records found for scan %s. Using first record.", scan)

            tds = self.build_track_dataset(result=result[0], tracks=tracks, 
                                           original_track_indices=np.array(streamlines), connections=np.array(connections))
            datasets.append(tds)

        return datasets

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
            match = self.db.connections2.find_one( { "scan_id": scan, "atlas_id": self.atlas_id }, [ "con_ids" ] )
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

        # Make sure all the graphml paths are the same and
        #self.graphml_cache = {}
        #for atlas, vary_props in varying_properties.iteritems():
        #    self.graphml_cache[atlas] = {}
        #    for vprop,possible_values in vary_props.iteritems():
        #        self.graphml_cache[atlas][
        #        if not prop in self.graphml_cache[atlas].keys():
        #            self.graphml_cache[atlas][prop] =

        return varying_properties

# TODO: this causes problems, need to fix
#    traits_view = View(
#        Group(
#            Item("json_source"),
#            Group(
#                Item("track_datasets", editor=track_dataset_table),
#                orientation="horizontal",
#                show_labels=False
#                ),
#            orientation="vertical"
#        ),
#        resizable=True,
#        width=900,
#        height=500
#    )
