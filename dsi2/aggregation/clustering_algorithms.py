#!/usr/bin/env python
import numpy as np
from ..streamlines.track_dataset import Cluster
from .cluster_ui import ClusterEditor
from ..streamlines.track_math import tracks_to_endpoints

from dipy.tracking import metrics as tm
from dipy.tracking import distances as td

from traits.api import HasTraits, Instance, Array, \
    Bool, Dict, Range, Color, List, Int, Property #, DelegatesTo, on_trait_change, Str, Tuple
from traitsui.api import Group,Item, RangeEditor

class QuickBundlesAggregator(ClusterEditor):
    parameters = ["dthr", "min_tracks"]
    min_tracks = Range(low=0,high=100,value=5, auto_set=False,name="min_tracks",
                          desc="A cluster label must be assigned to at least this many tracks",
                          label="Minimum tracks",
                          parameter=True
                          )
    dthr = Range(low=1.0,high=200.0,value=5.0, auto_set=False,name="dthr",
                          desc="Distance threshold between cluster endpoint means",
                          label="Endpoint Distance Threshold",
                          parameter=True
                          )

    def aggregate(self, ttracks):
        """
        """
        # Pull out np object arrays from the TrackDataset
        tracks = ttracks.tracks
        # Holds the cluster assignments for each track
        clusters = []

        # Run DiPy's local skeleton aggregation
        tep = tracks_to_endpoints(tracks/2)
        C = td.local_skeleton_clustering(tep,self.dthr)

        # Populate the clusters list
        labels = np.zeros(len(tracks),dtype=np.int)
        clustnum = 0
        for k in C.keys():
            # I think the coordinates are added to the 'hidden' field
            if C[k]['N'] < self.min_tracks: 
                label = -1
            else:
                label = clustnum
                clustnum += 1
            indices = C[k]['indices']
            labels[indices] = clustnum
        return labels

    # widgets for editing algorithm parameters
    algorithm_widgets = Group(
                         Item(name="min_tracks",
                              editor=RangeEditor(mode="slider", high = 100,low = 0,format = "%i")),
                         Item(name="dthr",
                              editor=RangeEditor(mode="slider", high = 100,low = 0,format = "%i"))
                              )

from sklearn.cluster import MiniBatchKMeans
class FastKMeansAggregator(ClusterEditor):
    parameters = ["k", "min_tracks"]
    min_tracks = Range(low=0,high=100,value=5, auto_set=False,name="min_tracks",
                          desc="A cluster label must be assigned to at least this many tracks",
                          label="Minimum tracks",
                          parameter=True
                          )
    k = Range(low=2,high=20,value=5, auto_set=False,name="k",
                          desc="K parameter for k-means",
                          label="Number of clusters",
                          parameter=True
                          )

    def aggregate(self, ttracks):
        """
        """
        # Holds the cluster assignments for each track
        clusters = []
        # Pull out np object arrays from the TrackDataset
        tracks = ttracks.tracks

        tep = tracks_to_endpoints(tracks/2).reshape(tracks.shape[0],6)
        # API change in scikit-learn
        try:
            mbk = MiniBatchKMeans(k=self.k)
        except Exception,e :
            mbk = MiniBatchKMeans(n_clusters=self.k)
            
        mbk.fit(tep)
        labels = mbk.labels_
        return labels        

    # widgets for editing algorithm parameters
    algorithm_widgets = Group(
                         Item(name="min_tracks",
                              editor=RangeEditor(mode="slider", high = 100,
                                                 low = 0,format = "%i")),
                         Item(name="k",
                              editor=RangeEditor(mode="slider", high = 20,
                                                 low = 0,format = "%i"))
                              )


#from sklearn.cluster import dbscan
#class DBSCANAggregator(ClusterEditor):
    #epsilon = Float(0.5,
                   #label="Epsilon",
                   #desc="The maximum distance between two samples " \
                   #"for them to be considered as in the same neighborhood.",
                   #parameter=True)

    #min_samples = Int(5,
                   #desc="The maximum distance between two samples for them " \
                   #"to be considered as in the same neighborhood.",
                   #parameter=True)

    #algorithm = Enum(("auto","ball_tree","kd_tree","brute"),
                   #desc="Algorithm used to compute pointwise distances",
                   #parameter=True)
                   
    #downsample_to = Int(8,
                    #desc="How many points should each streamline be " \
                    #"downsampled to?",
                    #parameter=True)
    
    #algorithm_widgets = Group(
                               #Item("epsilon"),
                               #Item("min_samples"),
                               #Item("algorithm")
                               #)
    #def aggregate(self, track_dataset):
        ## Pull out the numpy array of streamlines
        #tracks = track_dataset.tracks
        ## Downsample the tracks 
        #downsampled_tracks = np.array(
            #[downsample(track, n_pols=self.downsample_to).flatten() \
                #for track in tracks]
        #)
        ## Populate the clusters list
        #clusters = []
        #clustnum = 0
        #for k in C.keys():
            ## I think the coordinates are added to the 'hidden' field
            #if C[k]['N'] < self.min_tracks: continue
            #clustnum += 1
            #indices = C[k]['indices']
            #xyz = (C[k]['hidden']/C[k]['N'])
            #clusters.append(
                #Cluster(
                     #start_coordinate =  xyz[0],
                     #end_coordinate =    xyz[1],
                     #ntracks = len(indices),
                     #id_number = clustnum,
                     #indices = indices,
                     #scan_id = ttracks.scan_id
                #)
            #)

        #return clusters
