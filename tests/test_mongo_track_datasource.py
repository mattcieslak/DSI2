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

test_coordinates = sphere_around_ijk(3,(33,54,45))
test_con_ids = [3456, 3455]

def compare_results(mongo_results, local_results):
    assert len(mongo_results) == len(local_results)

    # TODO: is it always the case that both sets (mongo and local) of 
    # TrackDatasets are in the same order? The following assumes that they are.
    
    for mongo, local in zip(mongo_results, local_results):
        assert mongo.get_ntracks() == local.get_ntracks()
        assert (mongo.original_track_indices == local.original_track_indices).all()
        assert len(mongo.original_track_indices) == mongo.get_ntracks()
        
        assert mongo.render_tracks == local.render_tracks

        assert (mongo.connections == local.connections).all()

        assert mongo.properties.scan_id == local.properties.scan_id
        assert mongo.properties.subject_id == local.properties.subject_id
        assert mongo.properties.scan_gender == local.properties.scan_gender
        assert mongo.properties.scan_age == local.properties.scan_age
        assert mongo.properties.study == local.properties.study
        assert mongo.properties.scan_group == local.properties.scan_group
        assert mongo.properties.smoothing == local.properties.smoothing
        assert mongo.properties.cutoff_angle == local.properties.cutoff_angle
        assert mongo.properties.qa_threshold == local.properties.qa_threshold
        assert mongo.properties.gfa_threshold == local.properties.gfa_threshold
        assert mongo.properties.length_min == local.properties.length_min
        assert mongo.properties.length_max == local.properties.length_max
        assert mongo.properties.institution == local.properties.institution
        assert mongo.properties.reconstruction == local.properties.reconstruction
        assert mongo.properties.scanner == local.properties.scanner
        assert mongo.properties.n_directions == local.properties.n_directions
        assert mongo.properties.max_b_value == local.properties.max_b_value
        assert mongo.properties.bvals == local.properties.bvals
        assert mongo.properties.bvecs == local.properties.bvecs
        assert mongo.properties.label == local.properties.label
        assert mongo.properties.trk_space == local.properties.trk_space

        for mongo_track, local_track in zip(mongo.tracks, local.tracks):
            assert (mongo_track == local_track).all()

def test_load_label_data():
    # load the traditional track datasource for comparison
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])
    
    mtds = MongoTrackDataSource()

    mongo_results = mtds.load_label_data()
    local_results = tds.load_label_data()

    assert mongo_results == local_results

def test_query_ijk():

    # load the traditional track datasource for comparison
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])
    
    mtds = MongoTrackDataSource()
    
    mongo_results = mtds.query_ijk(test_coordinates)
    local_results = tds.query_ijk(test_coordinates)

    compare_results(mongo_results, local_results)

    # Try some different "every" values
    every = [0, 2, 5, 7]
    for e in every:
        mongo_results = mtds.query_ijk(test_coordinates, e)
        local_results = tds.query_ijk(test_coordinates, e)

        compare_results(mongo_results, local_results)

    every = -1
    mongo_exception = ""
    local_exception = ""
    try:
        mongo_results = mtds.query_ijk(test_coordinates, every)
    except ValueError, e:
        mongo_exception = str(e)
    try:
        local_results = tds.query_ijk(test_coordinates, every)
    except ValueError, e:
        local_exception = str(e)
    assert mongo_exception == local_exception != ""

    # This should result in no matches with the test data
    no_match = [(0, 0, 0)]
    mongo_results = mtds.query_ijk(no_match)
    local_results = tds.query_ijk(no_match)

    compare_results(mongo_results, local_results)

    # what if an atlas is loaded?
    mtds.load_label_data()
    tds.load_label_data()

    query = { "name": "Lausanne2008", "scale": 33 }
    
    mtds.change_atlas(query)
    tds.change_atlas(query)

    mongo_results = mtds.query_ijk(test_coordinates)
    local_results = tds.query_ijk(test_coordinates)

    compare_results(mongo_results, local_results)
 
def test_query_connection_id():

    # load the traditional track datasource for comparison
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])

    mtds = MongoTrackDataSource()
 
    tds.load_label_data()
    mtds.load_label_data()

    # Test each variation of the Lausanne2008 atlas

    scales = [33, 60, 125, 250, 500]    
    query = { "name": "Lausanne2008" }

    for scale in scales:
        query["scale"] = scale

        tds.change_atlas(query)
        mtds.change_atlas(query)

        mongo_results = mtds.query_connection_id(test_con_ids)
        local_results = tds.query_connection_id(test_con_ids)

        compare_results(mongo_results, local_results)

    # Try some different "every" values with Lausanne2008-33
    query["scale"] = 33
    every = [0, 2, 5, 7]
    for e in every:
        mongo_results = mtds.query_connection_id(test_con_ids, e)
        local_results = tds.query_connection_id(test_con_ids, e)

        compare_results(mongo_results, local_results)

    every = -1
    mongo_exception = ""
    local_exception = ""
    try:
        mongo_results = mtds.query_connection_id(test_con_ids, every)
    except ValueError, e:
        mongo_exception = str(e)
    try:
        local_results = tds.query_connection_id(test_con_ids, every)
    except ValueError, e:
        local_exception = str(e)
    assert mongo_exception == local_exception != ""


    # bogus atlas
    query["scale"] = 700

    local_exception = ""
    mongo_exception = ""

    try:
        tds.change_atlas(query)
    except ValueError, e:
        local_exception = str(e)

    try:
        mtds.change_atlas(query)
    except ValueError, e:
        mongo_exception = str(e)

    assert local_exception == mongo_exception != ""

def test_len():

    # load the traditional track datasource for comparison
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])
 
    mtds = MongoTrackDataSource()

    assert len(mtds) == len(tds)

def test_get_subjects():

    # load the traditional track datasource for comparison
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])
 
    mtds = MongoTrackDataSource()
    
    assert mtds.get_subjects() == tds.get_subjects()

def test_set_render_tracks():

     # load the traditional track datasource for comparison
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])
     
    mtds = MongoTrackDataSource()
    mongo_results = mtds.query_ijk(test_coordinates)
    local_results = tds.query_ijk(test_coordinates)
    for mongo, local in zip(mongo_results, local_results):
        assert mongo.render_tracks == local.render_tracks

    mtds.set_render_tracks(True)
    tds.set_render_tracks(True)

    mongo_results = mtds.query_ijk(test_coordinates)
    local_results = tds.query_ijk(test_coordinates)
    for mongo, local in zip(mongo_results, local_results):
        assert mongo.render_tracks == True
        assert mongo.render_tracks == local.render_tracks

    mtds.set_render_tracks(False)
    tds.set_render_tracks(False)

    mongo_results = mtds.query_ijk(test_coordinates)
    local_results = tds.query_ijk(test_coordinates)
    for mongo, local in zip(mongo_results, local_results):
        assert mongo.render_tracks == False
        assert mongo.render_tracks == local.render_tracks

