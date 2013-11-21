#!/usr/bin/env python
from traits.api import HasTraits, List, Instance, Bool, Str, File, Dict
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn
from ..streamlines.track_dataset import TrackDataset
from ..streamlines.track_dataset import TrackDataset
from mayavi.tools.mlab_scene_model import MlabSceneModel
from .traited_query import Scan
from .local_data import get_local_data
import numpy as np
import os
from collections import defaultdict

import pymongo
from bson.binary import Binary
connection = pymongo.MongoClient()
db = connection.dsi2


class TrackDataSource(HasTraits):
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

    def __init__(self,**traits):
        super(TrackDataSource,self).__init__(**traits)
        # if track_datasets is not passed explicitly
        if not self.track_datasets:
            # Load from a json_source
            if self.json_source:
                self.track_datasets = [ d.get_track_dataset()  for d in \
                    get_local_data(self.json_source) ]
        # grab the properties from each loaded TrackDataset
        self.track_dataset_properties = \
            [tds.properties for tds in self.track_datasets]

    def get_subjects(self):
        return sorted(list(set(
            [ds.subject_id for ds in self.track_dataset_properties])))

    def set_render_tracks(self,visibility):
        for tds in self.track_datasets:
            tds.render_tracks = visibility

    def abc_splits(self):
        """ When the datasets loaded are from the triplicates,
        This function will generate `nperms` permutations of the
        dataset such that one scan from each subject is selected
        to be in the training set.
        """
        # Get all unique subject names
        subjects = self.get_subjects()
        scans = [ds.scan_id for ds in self.track_dataset_properties]
        scan_indices = defaultdict(list)
        # Loop over all scan id's and assign their index to the subject
        # they came from
        for scan_idx, scan in enumerate(scans):
            for subject in subjects:
                if scan.startswith(subject):
                    scan_indices[subject].append(scan_idx)
        indices = np.array([ scan_indices[subj] for subj in subjects ], dtype=int)
        splits = [
                ( indices[:,0], (indices[:,1], indices[:,2])),
                ( indices[:,1], (indices[:,0], indices[:,2])),
                ( indices[:,2], (indices[:,0], indices[:,1])),
                 ]
        return splits


    def __len__(self):
        return len(self.track_datasets)

    def query_ijk(self,ijk,every=0):


        return [ tds.subset(tds.get_tracks_by_ijks(ijk),every=every) \
                       for tds in self.track_datasets ]

    def query_connection_id(self,connection_id,every=0):
        """
        Subsets the track datasets so that only streamlines labeled as
        `region_pair_id`
        """
        if self.needs_atlas_update: self.update_atlas()
        return [ tds.subset(tds.get_tracks_by_connection_id(connection_id),every=every) \
                       for tds in self.track_datasets ]

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

    traits_view = View(
        Group(
            Item("json_source"),
            Group(
                Item("track_datasets", editor=track_dataset_table),
                orientation="horizontal",
                show_labels=False
                ),
            orientation="vertical"
        ),
        resizable=True,
        width=900,
        height=500
    )

