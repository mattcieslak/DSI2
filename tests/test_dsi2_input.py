import pytest
import os, shutil
import numpy as np
from create_testing_data import (create_test_data, 
        tds1, tds1_scale33, tds1_scale60,
        tds2, tds2_scale33, tds2_scale60 )
from paths import test_input_data, test_output_data, input_data_json, local_trackdb_dir
from dsi2.ui.local_data_importer import LocalDataImporter, create_missing_files, \
     get_region_ints_from_graphml, graphml_from_label_source
from dsi2.volumes.mask_dataset import MaskDataset
from dsi2.streamlines.track_math import connection_ids_from_tracks
from dsi2.streamlines.track_dataset import TrackDataset
import cPickle as pickle

class TestDSI2:
    def test_data_importer(self):
        create_test_data()
        if os.path.exists(local_trackdb_dir):
            print "Removing local trackdb dir:", local_trackdb_dir
            shutil.rmtree(local_trackdb_dir)
        os.makedirs(local_trackdb_dir)
        ldi = LocalDataImporter(json_file=input_data_json)
        assert len(ldi.datasets) == 2
        for scan in ldi.datasets:
            assert create_missing_files(scan)
            # Check the pickle file
            fop = open(scan.pkl_path,"rb")
            test_pkl = pickle.load(fop)
            fop.close()
            assert type(test_pkl) is TrackDataset
            assert scan.scan_id in ("s1", "s2")
        
            
        # Check the label assignments of s1, scale33
        ds1 = ldi.datasets[0]
        ds1_s33_lbl = ds1.track_label_items[0]
        assert np.all(np.load(ds1_s33_lbl.numpy_path) == tds1_scale33)
        mds = MaskDataset(ds1_s33_lbl.qsdr_volume_path)
        graphml = graphml_from_label_source(ds1_s33_lbl)
        regions = get_region_ints_from_graphml(graphml)
        conn_ids = connection_ids_from_tracks(mds, tds1,
                        save_npy=False,
                        scale_coords=tds1.header['voxel_size'],
                        region_ints=regions,
                        correct_labels=tds1_scale33)
        # Check the label assignments of s1, scale60
        ds1_s60_lbl = ds1.track_label_items[1]
        assert np.all(np.load(ds1_s60_lbl.numpy_path) == tds1_scale60)
        mds = MaskDataset(ds1_s60_lbl.qsdr_volume_path)
        graphml = graphml_from_label_source(ds1_s60_lbl)
        regions = get_region_ints_from_graphml(graphml)
        conn_ids = connection_ids_from_tracks(mds, tds1,
                        save_npy=False,
                        scale_coords=tds1.header['voxel_size'],
                        region_ints=regions,
                        correct_labels=tds1_scale60)
        # Check the label assignments of s2, scale33
        ds2 = ldi.datasets[1]
        ds2_s33_lbl = ds2.track_label_items[0]
        assert np.all(np.load(ds2_s33_lbl.numpy_path) == tds2_scale33)
        mds = MaskDataset(ds2_s33_lbl.qsdr_volume_path)
        graphml = graphml_from_label_source(ds2_s33_lbl)
        regions = get_region_ints_from_graphml(graphml)
        conn_ids = connection_ids_from_tracks(mds, tds2,
                        save_npy=False,
                        scale_coords=tds2.header['voxel_size'],
                        region_ints=regions,
                        correct_labels=tds2_scale33)
        # Check the label assignments of s2, scale60
        ds2_s60_lbl = ds2.track_label_items[1]
        mds = MaskDataset(ds2_s60_lbl.qsdr_volume_path)
        assert np.all(np.load(ds2_s60_lbl.numpy_path) == tds2_scale60)
        graphml = graphml_from_label_source(ds2_s60_lbl)
        regions = get_region_ints_from_graphml(graphml)
        conn_ids = connection_ids_from_tracks(mds, tds2,
                        save_npy=False,
                        scale_coords=tds2.header['voxel_size'],
                        region_ints=regions,
                        correct_labels=tds2_scale60)
            
    def test_mongo_uploader(self):
        pass
    