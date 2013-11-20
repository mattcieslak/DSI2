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
from dsi2.database.track_datasource import TrackDataSource
from dsi2.streamlines.track_math import sphere_around_ijk


scans = None
tds =None

def test_loading():
    global scans
    global tds
    scans = get_local_data(os.path.join(test_output_data,"example_data.json"))
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




