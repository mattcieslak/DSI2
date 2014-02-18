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

import dsi2.config

def dictmatch(qdict,ddict):
    return all([ ddict.get(key,"") == val for key,val in qdict.iteritems() ] )


track_dataset_table = TableEditor(
    columns =
    [   ObjectColumn(name="properties.scan_id",editable=True),
        ObjectColumn(name="properties.study",editable=True),
        ObjectColumn(name="properties.scan_group",editable=True),
        ObjectColumn(name="properties.reconstruction",editable=True),
    ],
    auto_size  = True,
    edit_view="import_view"
    )

class PartitionedDatasource(object):
    def __init__(self, indices, n_partitions,n_overlap=2,split_axis="y"):
        """
        Parameters:
        -----------
        indices:np.ndarray n x 3
            i, j, k indices for each searchable voxel.
        n_partitions:int
            how many parts to divide the indices into
        split_axes:str
            x,y or z axis to split the data on?
        n_overlap:int
            how many voxels should each partition overlap?


        |------|
             |------|
                  |------|
        """
        self.n_overlap = n_overlap
        self.n_partitions = n_partitions
        self.indices = indices
        axes = {"x":0,"y":1,"z":2}
        self.axis_num = axes[split_axis]
        # Get the range of indices along the split axis
        n_indices = indices.shape[0]
        split_value = n_indices/n_partitions # how many indices per partition?
        ax_values = indices[:, self.axis_num] # Actual
        imin,imax = ax_values.min(), ax_values.max()
        ax_range = np.arange(imin,imax+1)
        counts,bins = np.histogram(ax_values,bins=ax_range)
        # indices where this axis should be split for roughly equal
        self.split_indices = [imin] + np.flatnonzero(
           np.diff(np.floor(np.cumsum(counts)/split_value))).tolist()

    def get_partition(self,partition_number,data_source):
        """
        Takes a track_datasource and returns a partition of its
        combined ijk indices

        Parameters:
        -----------
        partition_number:int
            which partition to retrieve?
        data_source:TrackDataSource
            Huge datasource to turn subset

        Returns:
        --------
        queryable_indices, track_datasource
        """
        # minimum index accessible to this chunk
        index_min = self.split_indices[partition_number]
        if partition_number > 0:
            partition_min = index_min - self.n_overlap
        else:
            partition_min = index_min
        # Maximum index accessible to this chunk
        index_max = self.split_indices[partition_number+1]
        # if partition_number != self.n_partitions:
        partition_max = index_max + self.n_overlap # Can go over because we're using
                           # defaultdict(set). may need to fix w/ mongo?

        # create a new set of subsetted track datasets
        new_hashes = []
        indices = []
        for tds in data_source.track_datasets:
            # make an empty TrackDataset to stick tracks_at_ijk in
            new_tds = TrackDataset(
                          header=tds.header,
                          tracks=np.array([],dtype=object),
                          properties=tds.properties,
                          connections=tds.connections
                          )
            tracks_at_ijk = defaultdict(set)
            for coord, ids in tds.tracks_at_ijk.iteritems():
                c = coord[self.axis_num]
                if c >= partition_min and c <= partition_max:
                    # add the voxel to the lookup table
                    if c >= index_min and c >= index_max:
                        indices.append(coord)
                    tracks_at_ijk[coord] = ids
            # Set the minimal hash table for this partition
            new_tds.tracks_at_ijk = tracks_at_ijk
            new_hashes.append(new_tds)

        return indices, TrackDataSource(track_datasets=new_hashes)

    def get_partitions(self,data_source):
        """
        Takes a track_datasource and returns a partition of its
        combined ijk indices

        Parameters:
        -----------
        data_source:TrackDataSource
            Huge datasource to turn subset

        Returns:
        --------

        """
        # minimum index accessible to this chunk
        print "Splitting data source into", self.n_partitions, "partitions"
        print "original size of index", self.indices.shape[0]
        index_blocks = []
        # create a new set of subsetted track datasets
        new_datasources = []
        for partition_number in range(self.n_partitions):
            index_min = self.split_indices[partition_number]
            if partition_number > 0:
                partition_min = index_min - self.n_overlap
            else:
                partition_min = index_min
            # Maximum index accessible to this chunk
            index_max = self.split_indices[partition_number+1]
            # if partition_number != self.n_partitions:
            partition_max = index_max + self.n_overlap # Can go over because we're using
                               # defaultdict(set). may need to fix w/ mongo?
            # Append these indices to a list of the index partitions
            this_blocks_indices = self.indices[
                        (self.indices[:,self.axis_num] >= index_min) & \
                        (self.indices[:,self.axis_num] <= index_max) ]

            index_blocks.append(this_blocks_indices)
            print "partition",partition_number,"contains",this_blocks_indices.shape[0]

            new_hashes = []
            for tds in data_source.track_datasets:
                # make an empty TrackDataset to stick tracks_at_ijk in
                new_tds = TrackDataset(
                              header=tds.header,
                              tracks=np.array([],dtype=object),
                              properties=tds.properties,
                              connections=tds.connections
                              )
                tracks_at_ijk = defaultdict(set)
                for coord in map(tuple,this_blocks_indices):
                        tracks_at_ijk[coord] = tds.tracks_at_ijk[coord]
                # Set the minimal hash table for this partition
                new_tds.tracks_at_ijk = tracks_at_ijk
                new_hashes.append(new_tds)
            new_datasources.append(TrackDataSource(track_datasets=new_hashes))

        return list(zip(index_blocks,new_datasources))


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

    def non_stratified_triplicate_permutations(self,nperms):
        """ When the datasets loaded are from the triplicates,
        This function will generate `nperms` permutations of the
        dataset such that one scan from each subject is selected
        to be in the training set.
        """
        # Get all unique subject names
        subjects = sorted(list(set(
            [ds.subject_id for ds in self.track_dataset_properties])))
        scans = [ds.scan_id for ds in self.track_dataset_properties]
        scan_indices = defaultdict(list)
        # Loop over all scan id's and assign their index to the subject
        # they came from
        for scan_idx, scan in enumerate(scans):
            for subject in subjects:
                if scan.startswith(subject):
                    scan_indices[subject].append(scan_idx)
        permutations = set()
        n_attempts = 0
        # Attempt to generate `nperms` unique permutations of the training set
        max_attempts = nperms * nperms
        while len(permutations) < nperms:
            permutations.update(
                (tuple([np.random.randint(3) for subject in subjects]),)
                )
            n_attempts += 1
            if n_attempts > max_attempts:
                print "Reached maximum attempts at generating unique permutations"
                break
        # turn the {0,1,2} into indices into
        permutation_indices = []
        for perm in permutations:
            permutation_indices.append(
                   np.array([scan_indices[subject_id][indx] for subject_id, indx in \
                      zip(subjects,perm)], dtype=int)
                   )

        return subjects, permutation_indices, scan_indices

    def within_between_subjects_triangle_slice(self):
        props = self.track_dataset_properties
        withins = []
        betweens = []
        for a,b in np.array( np.triu_indices(len(self),1) ).T:
            if props[a].scan_id == props[b].scan_id: continue
            if props[a].subject_id == props[b].subject_id:
                withins.append((a,b))
            else:
                betweens.append((a,b))
        return tuple(np.array(withins).T), tuple(np.array(betweens).T)


    def __len__(self):
        return len(self.track_datasets)

    def query_ijk(self,ijk,every=0):
        if self.needs_atlas_update: self.update_atlas()
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
        #print varying_properties
        dsi2.config.logger.debug("%s", varying_properties)
        
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

