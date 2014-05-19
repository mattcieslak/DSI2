#!/usr/bin/env python
from traits.api import HasTraits, List, Instance, Bool, Str, File, Dict
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn
from ..streamlines.track_dataset import TrackDataset
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

connection = pymongo.MongoClient()
db = connection.dsi2


class MongoTrackDataSource(HasTraits):
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

    def __init__(self,**traits):
        super(MongoTrackDataSource,self).__init__(**traits)
        # if track_datasets is not passed explicitly
        #if not self.track_datasets:
            # Load from a json_source
            #if self.json_source:
                #self.track_datasets = [ d.get_track_dataset()  for d in \
                    #get_local_data(self.json_source) ]
        # grab the properties from each loaded TrackDataset
        #self.track_dataset_properties = \
            #[tds.properties for tds in self.track_datasets]

    def get_subjects(self):
        result = db.scans.find(fields={ "subject_id": 1 })
        return sorted(list(set(
            [rec["subject_id"] for rec in result])))

    def set_render_tracks(self,visibility):
        self.render_tracks = visibility

    def __len__(self):
        return db.scans.count()

    def build_track_dataset(self,result,tracks,original_track_indices):
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

        tds = TrackDataset(tracks=tracks, header=header, original_track_indices=original_track_indices, properties=properties)
        tds.render_tracks = self.render_tracks
        return tds

    def query_ijk(self,ijk,every=0):
        # Get the scan_ids that contain these coordinates
        coords = [str(c) for c in ijk]
        result = db.coordinates.find( { "ijk": { "$in": coords } }, { "scan_id": 1 } )
        scans = set([rec["scan_id"] for rec in result])

        # Get the streamlines for each scan and build a list of TrackDatasets
        datasets = []
        for scan in scans:
            result = db.coordinates.find( { "ijk": { "$in": coords }, "scan_id": scan }, { "sl_id": 1 } )
            streamlines = sorted(list(set(reduce(operator.add, [rec["sl_id"] for rec in result]))))
            result = db.streamlines.find( { "sl_id": { "$in": streamlines }, "scan_id": scan } )
            tracks = [pickle.loads(rec["data"]) for rec in result]
            result = db.scans.find( { "scan_id": scan } )

            if result.count() > 1:
                logger.warning("Multiple records found for scan %s. Using first record.", scan)

            tds = self.build_track_dataset(result=result[0], tracks=tracks, original_track_indices=np.array(streamlines))
            tds = tds.subset(range(tds.get_ntracks()), every=every)
            datasets.append(tds)
        
        return datasets

    def query_connection_id(self,connection_id,every=0):
        """
        Subsets the track datasets so that only streamlines labeled as
        `region_pair_id`
        """

        if type(connection_id) == int:
            connection_id = np.array([connection_id])
        elif type(connection_id) != np.ndarray:
            connection_id = np.array(connection_id)

        # Get the scan_ids that contain these connection(s)
        cons = [str(c) for c in connection_id]
        result = db.connections2.find( { "con_id": { "$in": cons } }, { "scan_id": 1 } )
        scans = set([rec["scan_id"] for rec in result])

        # Get the streamlines for each scan and build a list of TrackDatasets
        datasets = []
        for scan in scans:
            result = db.connections2.find( { "con_id": { "$in": cons }, "scan_id": scan }, { "sl_ids": 1 } )
            streamlines = []
            for rec in result:
                for sl in rec["sl_ids"]:
                    streamlines.append(sl)
            streamlines = sorted(list(set(streamlines)))
            result = db.streamlines.find( { "sl_id": { "$in": streamlines }, "scan_id": scan } )
            tracks = [pickle.loads(rec["data"]) for rec in result]
            result = db.scans.find( { "scan_id": scan } )

            if result.count() > 1:
                logger.warning("Multiple records found for scan %s. Using first record.", scan)

            tds = self.build_track_dataset(result=result[0], tracks=tracks, original_track_indices=np.array(streamlines))
            tds = tds.subset(range(tds.get_ntracks()), every=every)
            datasets.append(tds)

        return datasets

    def change_atlas(self, query_specs):
        """ Sets the .connections for each TrackDataset to be loaded from the path
        specified in its properties.atlases
        """
        print "\t+ Setting new atlas in TrackDataSource"
        new_labels = []
        for tds, cache in zip(self.track_datasets,self.label_cache):
            match = [lbl["data"] for lbl in cache if dictmatch(query_specs,lbl)]
            if not len(match) ==1:
                raise ValueError("Query did not return exactly one match")
            # ATTACHES LABELS TO THE `.connections` ATTRIBUTE OF EACH TDS
            tds.set_connections(match[0])
            new_labels += match

        return new_labels

    def load_label_data(self):
        # What are the unique atlas names?
        atlases = set([])
        for prop in self.track_dataset_properties:
            atlases.update([d.name for d in prop.track_label_items])
        atlases = list(atlases)
        print "\t+ Found %d unique atlases" % len(atlases)

        # HORRIBLE.
        label_lut = {}
        for atlas_name in atlases:
            label_lut[atlas_name] = defaultdict(set)
            for tds_props in self.track_dataset_properties:
                # loop over label sources
                for label_source in tds_props.track_label_items:
                    # If they're for this atlas, append the parameter value
                    if label_source.name == atlas_name:
                        # For each parameter
                        for prop, propval in label_source.parameters.iteritems():
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

        # Put in a look-up
        self.label_cache = []  # one row for each subject
        for props in self.track_dataset_properties:
            subj_cache = []
            # one row in the cache for each item
            for label_item in props.track_label_items:
                subj_lut = {"name":label_item.name}
                subj_lut.update(label_item.parameters)
                subj_lut['data'] = \
                            np.load(
                               os.path.join(props.pkl_dir,label_item.numpy_path)
                               ).astype(np.uint64)

                subj_cache.append(subj_lut)
            self.label_cache.append(subj_cache)

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

