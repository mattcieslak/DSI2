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
    
    for mongo, local in zip(mongo_results, local_results):
        assert mongo.get_ntracks() == local.get_ntracks()
        assert len(mongo.original_track_indices) == len(set(mongo.original_track_indices))
        assert len(local.original_track_indices) == len(set(local.original_track_indices))
        assert len(mongo.original_track_indices) == len(local.original_track_indices)
        assert len(mongo.original_track_indices) == mongo.get_ntracks()
        
        assert mongo.render_tracks == local.render_tracks

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

        # streamlines will probably not be in the same order, but they
        # should match based on their original index
        for mongoindex, index in enumerate(mongo.original_track_indices):
            localindex = np.where(local.original_track_indices == index)[0]
            assert len(localindex) == 1
            localindex = localindex[0]
            assert (mongo.tracks[mongoindex] == local.tracks[localindex]).all()

def test_query_ijk():

    # load the traditional track datasource for comparison
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])
    
    mtds = MongoTrackDataSource()
    
    mongo_results = mtds.query_ijk(test_coordinates)
    local_results = tds.query_ijk(test_coordinates)

    compare_results(mongo_results, local_results)

def test_query_connection_id():
    # load the traditional track datasource for comparison
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])

    # load connection data for the traditional track datasource
    for trackds in tds.track_datasets:
        atlas_labels = trackds.properties.track_label_items[0].load_array(paths.test_output_data)
        trackds.set_connections(atlas_labels)
    
    mtds = MongoTrackDataSource()

    mongo_results = mtds.query_connection_id(test_con_ids)
    local_results = tds.query_connection_id(test_con_ids)

    compare_results(mongo_results, local_results)

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

def test_load_label_data():
    # load the traditional track datasource for comparison
    scans = get_local_data(os.path.join(paths.test_output_data,"example_data.json"))
    tds = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])
    
    mtds = MongoTrackDataSource()

    mongo_results = mtds.load_label_data()
    local_results = tds.load_label_data()

    assert mongo_results == local_results
