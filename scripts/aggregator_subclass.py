#!/usr/bin/env python
import time, sys, os
from dsi2.database.mongo_track_datasource import MongoTrackDataSource
from dsi2.ui.sphere_browser import SphereBrowser
from dsi2.aggregation.cluster_ui import ClusterEditor
from dipy.tracking.metrics import downsample
from traits.api import Int
from traitsui.api import Item, Group
from sklearn.cluster import MiniBatchKMeans
import numpy as np

class KMeansAggregator(ClusterEditor):
    k = Int(3,
            label="K",
            desc="How many groups should the streamlines be grouped into? ",
            parameter=True)        
    algorithm_widgets = Group(
                          Item("k")
                          )        
    def aggregate(self, track_dataset):
        """
        An example implementation of the k-means algorithm implemented in 
        DSI Studio.  This function is automatically applied to all 
        TrackDatasets returned from a query.
  
        Parameters:
        -----------
        track_dataset:dsi2.streamlines.track_dataset.TrackDataset
        """
        # extract the streamline data
        tracks = track_dataset.tracks
        
        # Make a matrix of downsampled streamlines
        points = np.array([ downsample(trk, 3).flatten() \
                                    for trk in tracks])
  
        # Calculate the length of each streamline
        lengths = np.array([len(trk) for trk in tracks]).reshape(-1,1)
        
        # Concatenate the points and the track lengths
        features = np.hstack((points, lengths))
        
        # Initialize the k-means algorithm
        kmeans = MiniBatchKMeans(n_clusters=self.k, compute_labels=True)
        kmeans.fit(features)
  
        # Return the labels
        return kmeans.labels_      
    
all_subjects = [ "0377A" ]
    
    
# If you want to only do a subset of these, select them here.
# Otherwise, set test_subject=all_subjects
test_subjects = all_subjects


data_source = MongoTrackDataSource(
    scan_ids = test_subjects,
    mongo_host = "127.0.0.1",
    mongo_port = 27017,
    db_name="dsi2_test"
)
kmeans_agg = KMeansAggregator()

browser = SphereBrowser(track_source=data_source, aggregator=kmeans_agg)
browser.edit_traits()