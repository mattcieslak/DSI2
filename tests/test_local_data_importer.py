#!/usr/bin/env python
import sys
import os
import os.path as op
from paths import test_input_data, test_output_data, input_data_json, local_trackdb_dir
import nibabel as nib
import numpy as np
import cPickle as pickle
from create_testing_data import (tds1, tds1_scale33, tds1_scale60, 
                                 tds2, tds2_scale33, tds2_scale60)

from dsi2.ui.local_data_importer import (LocalDataImporter, b0_to_qsdr_map,
                                         create_missing_files)

def test_b0_to_qsdr_map():
    """
    Tests that the mapping from b0 space to QSDR space works
    correctly.  We use the ROI and fib volumes generated for 
    testing.  The trick here is that the fib file contains a 
    self-mapping.  Therefore the data in b0-space should be the
    same as the data after mapping to qsdr space.
    """
    b0_vol = op.join(test_input_data,"s1",
            "s1.scale33.nii.gz")
    fib_file =op.join(test_input_data,"s1",
            "s1.fib.gz")
    qsdr_out = op.join(test_output_data,"tmp.QSDR_out.nii.gz")
    # Perform the mapping
    b0_to_qsdr_map(fib_file,b0_vol,qsdr_out)
    # check that data is unchanged
    correct_vol = nib.load(b0_vol).get_data()
    test_vol = nib.load(qsdr_out).get_data()
    assert (test_vol==correct_vol).all()


def test_create_missing_files():
    json_file = input_data_json
    os.makedirs(local_trackdb_dir)
    ldi = LocalDataImporter(json_file=input_data_json)
    assert len(ldi.datasets) == 2
    for scan in ldi.datasets:
        assert create_missing_files(scan)
        # Check the pickle file
        fop = open(scan.pkl_path,"rb")
        test_pkl = pickle.load(fop)
        fop.close()
        
        assert scan.scan_id in ("s1", "s2")
    # test that the numpy files match the correct streamlines    
    lbl0 = ldi.datasets[0].track_label_items
    assert np.all(np.load(lbl0[0].numpy_path) == tds1_scale33)
    assert np.all(np.load(lbl0[1].numpy_path) == tds1_scale60)