#!/usr/bin/env python
import os
import numpy as np
import paths
import dsi2.config
dsi2.config.local_trackdb_path = paths.test_output_data
from dsi2.database.local_data import get_local_data
from dsi2.database.track_datasource import TrackDataSource
from dsi2.streamlines.track_math import sphere_around_ijk
from dsi2.database.mongo_track_datasource import MongoTrackDataSource

# load the traditional track datasource for comparison
scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])

test_coordinates = sphere_around_ijk(3,(33,54,45))

def test_query_ijk():
    mtds = MongoTrackDataSource()
    mongo_results = mtds.query_ijk(test_coordinates)
    results = tds.query_ijk(test_coordinates)

    assert len(mongo_results) == len(results)
    for x, y in zip(mongo_results, results):
        assert x.get_ntracks() == y.get_ntracks()
        assert len(x.original_track_indices) == len(set(x.original_track_indices))
        assert len(y.original_track_indices) == len(set(y.original_track_indices))
        assert len(x.original_track_indices) == len(y.original_track_indices)
        assert len(x.original_track_indices) == x.get_ntracks()

        for mongoidx, idx in enumerate(x.original_track_indices):
            itemindex = np.where(y.original_track_indices == idx)[0]
            assert len(itemindex) == 1
            itemindex = itemindex[0]
            assert (x.tracks[mongoidx] == y.tracks[itemindex]).all()

def test_len():
    mtds = MongoTrackDataSource()
    assert len(mtds) == len(tds)

def test_get_subjects():
    mtds = MongoTrackDataSource()
    assert mtds.get_subjects() == tds.get_subjects()

