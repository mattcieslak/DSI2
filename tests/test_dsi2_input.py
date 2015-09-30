import pytest
import os, shutil
import numpy as np
from create_testing_data import (create_test_data, 
        tds1, tds1_scale33, tds1_scale60,
        tds2, tds2_scale33, tds2_scale60,
        SCALE60_LPC_1_TO_BS, SCALE60_LPC_2_TO_BS,
        SCALE60_RPC_1_TO_BS, SCALE60_RPC_2_TO_BS,
        SCALE33_LPC_TO_BS, SCALE33_RPC_TO_BS)
from paths import ( 
    test_input_data, test_output_data, 
    input_data_json, local_trackdb_dir,
    local_mongodb_dir, local_mongodb_log )
from dsi2.ui.local_data_importer import LocalDataImporter, create_missing_files, \
     graphml_from_label_source, MongoCreator
from dsi2.volumes import get_region_ints_from_graphml
from dsi2.volumes.mask_dataset import MaskDataset
from dsi2.streamlines.track_math import connection_ids_from_tracks
from dsi2.streamlines.track_dataset import TrackDataset
from dsi2.database.track_datasource import TrackDataSource
from dsi2.database.mongo_track_datasource import MongoTrackDataSource
from dsi2.database.local_data import get_local_data
import cPickle as pickle
import pymongo

@pytest.fixture(scope="session")
def mongo_db(request):
    """
    Opens a vanilla mongod client and starts it running
    returns a dsi2.ui.local_data_importer.MongoCreator instance
    """
    mongo_conn = MongoCreator(
        database_dir= local_mongodb_dir,
        restrict_ips = True,
        numactl_interleave=False,
        log_path=local_mongodb_log,
        port="27999"
        )
    assert mongo_conn._b_start_fired()
    print 
    def shutdown_mongod():
        print "shutting down mongod"
        mongo_conn.shutdown()
        os.system("rm -rf " + local_mongodb_dir)
    request.addfinalizer(shutdown_mongod)
    return mongo_conn
    
    
@pytest.fixture(scope="session")
def data_importer(request, create_test_data):
    """
    automatically calls create_test_data, makes sure all files
    exist to run a full-fledged session on local data.
    
    Directory is automatically cleaned at the end of the session
    """
    if os.path.exists(local_trackdb_dir):
        print "Removing local trackdb dir:", local_trackdb_dir
        os.system("rm -rf " + local_trackdb_dir)
    os.makedirs(local_trackdb_dir)
    ldi = LocalDataImporter(json_file=create_test_data)
    assert len(ldi.datasets) == 2
    for scan in ldi.datasets:
        assert create_missing_files(scan)
        # Check the pickle file
        assert scan.scan_id in ("s1", "s2")
        
    def finalize():
        print "Finally: removing local trackdb dir:", local_trackdb_dir
        os.system("rm -rf " + local_trackdb_dir)
    request.addfinalizer(finalize)
    return ldi

@pytest.fixture(scope="session")
def mni_wm_coords(data_importer):
    # gets all coordinates covered by streamlines
    scan = data_importer.datasets[0]
    fop = open(scan.pkl_path,"rb")
    mni_tds1 = pickle.load(fop)
    fop.close()
    scan = data_importer.datasets[1]
    fop = open(scan.pkl_path,"rb")
    mni_tds2 = pickle.load(fop)
    fop.close()
    all_coords = set(mni_tds1.tracks_at_ijk.keys()) | set(mni_tds2.tracks_at_ijk.keys()) 
    return list(all_coords)

def test_data_importer_save(data_importer,tmpdir):
    """ 
    tests that LocalDataImporter can save a file that is loadable
    as a LocalDataImporter
    """
    new_json = str(tmpdir.join("test.json"))
    data_importer.json_file = new_json
    data_importer._save_fired()
    new_ldi = LocalDataImporter(json_file=new_json)
    assert len(new_ldi.datasets)
    
def test_mongo_upload(data_importer,mongo_db):
    data_importer.mongo_creator = mongo_db
    data_importer._upload_to_mongodb_fired()
    
@pytest.fixture(scope="session")
def track_datasource(create_test_data):
    return TrackDataSource(
        track_datasets = [
            scan.get_track_dataset() for scan in \
            get_local_data(create_test_data)] )
    
    
@pytest.fixture(scope="session")
def mongo_track_datasource(mongo_db):
    return MongoTrackDataSource(
                scan_ids=["s1","s2"],
                dbname="dsi2",
                client=mongo_db.get_connection())
    
    
    
def test_data_sources(mongo_track_datasource,track_datasource):
    """
    Compare TrackDataSource to MongoTrackDataSource
    """
    mds = mongo_track_datasource
    tds = track_datasource
    assert len(mds.track_datasets) == len(tds.track_datasets) == 2
    assert mds.get_subjects() == tds.get_subjects()
    
from dsi2.aggregation import make_aggregator   
def test_region_aggregator(mongo_track_datasource,track_datasource,mni_wm_coords):
    mds = mongo_track_datasource
    tds = track_datasource
    # Check that scale 60 is the same
    mds_agg_60 = make_aggregator( 
        algorithm="region labels",
        atlas_name="Lausanne2008",
        atlas_scale=60, data_source=mds)
    tds_agg_60 = make_aggregator( 
        algorithm="region labels",
        atlas_name="Lausanne2008",
        atlas_scale=60, data_source=tds)
    for conn_id in (SCALE60_LPC_1_TO_BS, SCALE60_LPC_2_TO_BS,
                    SCALE60_RPC_1_TO_BS, SCALE60_RPC_2_TO_BS):
        mds_agg_60.set_track_sets(mds.query_connection_id(conn_id))
        tds_agg_60.set_track_sets(mds.query_connection_id(conn_id))
        assert all(
            [np.all(m==t) for m,t in zip(
                mds_agg_60.track_sets[0].tracks, tds_agg_60.track_sets[0].tracks)])
        assert all(
            [np.all(m==t) for m,t in zip(
                mds_agg_60.track_sets[1].tracks, tds_agg_60.track_sets[1].tracks)])
    # also scale 33
    mds_agg_33 = make_aggregator( 
        algorithm="region labels",
        atlas_name="Lausanne2008",
        atlas_scale=33, data_source=mds)
    tds_agg_33 = make_aggregator( 
        algorithm="region labels",
        atlas_name="Lausanne2008",
        atlas_scale=33, data_source=tds)
    for conn_id in (SCALE33_LPC_TO_BS, SCALE33_RPC_TO_BS):
        mds_agg_33.set_track_sets(mds.query_connection_id(conn_id))
        tds_agg_33.set_track_sets(mds.query_connection_id(conn_id))
        assert all(
            [np.all(m==t) for m,t in zip(
                mds_agg_33.track_sets[0].tracks, tds_agg_33.track_sets[0].tracks)])
        assert all(
            [np.all(m==t) for m,t in zip(
                mds_agg_33.track_sets[1].tracks, tds_agg_33.track_sets[1].tracks)])
    
        
    # test individual sphere queries
    print "TESTING COORDINATE QUERIES"
    for ncoord, coord in enumerate(mni_wm_coords):
        if ncoord % 100 == 0:
            print ncoord, "/", len(mni_wm_coords)
        mds_agg_60.set_track_sets(mds.query_ijk((coord,)))
        mds_agg_60.update_clusters()
        c1,v1 = mds_agg_60.connection_vector_matrix()
        tds_agg_60.set_track_sets(mds.query_ijk((coord,)))
        tds_agg_60.update_clusters()
        c2,v2 = mds_agg_60.connection_vector_matrix()
        assert c1 == c2
        assert np.all(v1 == v2)
        mds_agg_33.set_track_sets(mds.query_ijk((coord,)))
        mds_agg_33.update_clusters()
        c1,v1 = mds_agg_33.connection_vector_matrix()
        tds_agg_33.set_track_sets(mds.query_ijk((coord,)))
        tds_agg_33.update_clusters()
        c2,v2 = mds_agg_33.connection_vector_matrix()
        assert c1 == c2
        assert np.all(v1 == v2)
    
    
    
    

    
    
    
    
    
        
