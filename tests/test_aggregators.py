#!/usr/bin/env python
import sys
import os
test_input_data = os.getenv("TEST_INPUT_DATA")
test_output_data = os.getenv("TEST_OUTPUT_DATA")
test_input_data = "/home/cieslak/testing_data/testing_input"
test_output_data = "/home/cieslak/testing_data/testing_output"
import nibabel as nib
import numpy as np
import cPickle as pickle

from dsi2.ui.local_data_importer import (LocalDataImporter, b0_to_qsdr_map,
                                         create_missing_files)
from dsi2.database.local_data import get_local_data
from dsi2.streamlines.track_dataset import TrackDataset
from dsi2.aggregation.clustering_algorithms import FastKMeansAggregator, QuickBundlesAggregator



scans = get_local_data(os.path.join(test_output_data,"example_data.json"))
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

def test_mini_batch_kmeans():
    global scans
    global tds
    mbk = MiniBatchKMeans()
    mbk.aggregate(toy_dataset)
    assert
    assert len(tds.track_dataset) == len(scans)

def test_quickbundles():
    query_coords = sphere_around_ijk(3,(33,54,45))
    assert len(coords) == 123
    assert type(tds) == TrackDataSource
    search_results = tds.query_ijk(coords)
    # There must be 2 results
    assert len(search_results) == 2
    # the known number of streamlines is 1844, 1480
    assert search_results[0].get_ntracks() == 1844
    assert search_results[1].get_ntracks() == 1480




