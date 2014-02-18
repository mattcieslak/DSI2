#!/usr/bin/env python
import sys
import os
import nibabel as nib
import numpy as np
import cPickle as pickle

import paths

import dsi2.config
dsi2.config.local_trackdb_path = paths.test_output_data

from dsi2.ui.local_data_importer import (LocalDataImporter, b0_to_qsdr_map,
                                         create_missing_files)
from dsi2.database.local_data import get_local_data
from dsi2.database.track_datasource import TrackDataSource
from dsi2.streamlines.track_math import sphere_around_ijk


scans = None
tds =None

def test_loading():
    global scans
    global tds
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])
    assert len(tds.track_datasets) == len(scans)

def test_querying():
    query_coords = sphere_around_ijk(3,(33,54,45))
    assert len(query_coords) == 123
    assert type(tds) == TrackDataSource
    search_results = tds.query_ijk(query_coords)
    # There must be 2 results
    assert len(search_results) == 2
    # the known number of streamlines is 1844, 1480
    assert search_results[0].get_ntracks() == 1844
    assert search_results[1].get_ntracks() == 1480

def test_get_subjects():
    subjects = tds.get_subjects()
    # There are two subjects
    assert len(subjects) == 2
    # The ids are 0377 and 2843
    assert subjects[0] == "0377"
    assert subjects[1] == "2843"

def test_set_render_tracks():
    tds.set_render_tracks(False)
    for dataset in tds.track_datasets:
        assert dataset.render_tracks == False

    tds.set_render_tracks(True)
    for dataset in tds.track_datasets:
        assert dataset.render_tracks == True

def test_len():
    assert len(tds) == 2

def test_load_label_data():
    results = tds.load_label_data()
    assert results["Lausanne2008"]["scale"] == [33, 60, 125, 250, 500]

