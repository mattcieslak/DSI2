#!/usr/bin/env python
import sys
import os
import nibabel as nib
import numpy as np
import cPickle as pickle

import paths

import dsi2.config
dsi2.config.local_trackdb_path = paths.test_input_data

from dsi2.ui.local_data_importer import (LocalDataImporter, b0_to_qsdr_map,
                                         create_missing_files)
from dsi2.database.local_data import get_local_data
from dsi2.streamlines.track_dataset import TrackDataset
from dsi2.aggregation.clustering_algorithms import FastKMeansAggregator, QuickBundlesAggregator


scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
toy_dataset = TrackDataset(header={"n_scalars":0}, streams = (),properties=scans[0])

cluster1 = [np.zeros((3,50)).T for x in range(20)]
cluster2 = [np.zeros((3,55)).T for x in range(20)]
cluster3 = [np.zeros((3,60)).T for x in range(20)]
for x in cluster1:
    x[:,0] = np.linspace(-100,100,50)
for x in cluster2:
    x[:,1] = np.linspace(-100,100,55)
for x in cluster3:
    x[:,2] = np.linspace(-100,100,60)

toy_dataset.tracks = np.array(cluster1 + cluster2 + cluster3,dtype=object)

def test_fast_kmeans():
    fkm = FastKMeansAggregator(k=3,min_tracks=0)
    clusters = fkm.aggregate(toy_dataset)
    assert len(clusters) == 3


    for cluster in clusters:
        assert cluster.ntracks == 20
        assert (np.diff(np.sort(cluster.indices)) == np.ones((19,))).all()


    #assert (clusters[0].start_coordinate == np.array([-50., 0., 0.])).all()
    #assert (clusters[0].end_coordinate == np.array([50., 0., 0.])).all()
    #assert (clusters[1].start_coordinate == np.array([0., -50., 0.])).all()
    #assert (clusters[1].end_coordinate == np.array([0., 50., 0.,])).all()
    #assert (clusters[2].start_coordinate == np.array([0., 0., -50.])).all()
    #assert (clusters[2].end_coordinate == np.array([0., 0., 50.])).all()

def test_quickbundles():
    qb = QuickBundlesAggregator()
    clusters = qb.aggregate(toy_dataset)
    assert len(clusters) == 3


    for cluster in clusters:
        assert cluster.ntracks == 20
        assert (np.diff(np.sort(cluster.indices)) == np.ones((19,))).all()

    #assert (clusters[0].start_coordinate == np.array([-50., 0., 0.])).all()
    #assert (clusters[0].end_coordinate == np.array([50., 0., 0.])).all()
    #assert (clusters[1].start_coordinate == np.array([0., -50., 0.])).all()
    #assert (clusters[1].end_coordinate == np.array([0., 50., 0.,])).all()
    #assert (clusters[2].start_coordinate == np.array([0., 0., -50.])).all()
    #assert (clusters[2].end_coordinate == np.array([0., 0., 50.])).all()
