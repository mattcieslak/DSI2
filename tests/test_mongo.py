#!/usr/bin/env python
import sys
import os
import paths

import dsi2.config
dsi2.config.local_trackdb_path = paths.test_output_data

import nibabel as nib
import numpy as np
import cPickle as pickle

import random

import pymongo
from bson.binary import Binary
connection = pymongo.MongoClient()
db = connection.dsi2

from dsi2.ui.local_data_importer import (LocalDataImporter, b0_to_qsdr_map,
                                         create_missing_files)

from dsi2.database.local_data import get_local_data

local_scans = get_local_data(paths.test_output_data + "/example_data.json")
scan = local_scans[0]
trackds = scan.get_track_dataset()

def test_compare_random_streamline():
   
    sl_id = random.randint(0, 99999)

    result = db.streamlines.find({"sl_id": sl_id, "scan_id": "0377A"})

    assert result.count() == 1

    sl = pickle.loads(result[0]["data"])

    assert (sl == trackds.tracks[sl_id]).all()

def test_scans_collection():

    result = db.scans.find()

    assert result.count() == 2

    result = db.scans.find({"scan_id": "0377A"})

    assert result.count() == 1

    result = db.scans.find({"scan_id": "2843A"})

    assert result.count() == 1

def test_compare_tracks_at_ijk():

    result = db.coordinates.find({"ijk": "33_54_45", "scan_id": "0377A"})

    assert result.count() == 1

    tracks = trackds.tracks_at_ijk[(33, 54, 45)]

    # fix
    assert result[0]["sl_id"] == tracks

