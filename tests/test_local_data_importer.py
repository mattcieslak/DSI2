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
def test_b0_to_qsdr_map(tmpdir):
    td =  "/" + tmpdir.relto("/")
    print type(td)
    b0_vol = os.path.join(test_input_data,"0377A",
            "B0.scale33.thick2.nii.gz")
    fib_file = os.path.join(test_input_data,"0377A",
            "0377A.src.gz.odf8.f3.reg1.qsdr.1.25.2mm.map.fib.gz")
    qsdr_out = os.path.join(td,"QSDR_out.nii.gz")
    correct_file = os.path.join(test_output_data,
            "0377A","QSDR.scale33.thick2.nii.gz")
    # Call the test function
    b0_to_qsdr_map(fib_file,b0_vol,qsdr_out)
    correct_vol = nib.load(correct_file).get_data()
    test_vol = nib.load(qsdr_out).get_data()
    assert (test_vol==correct_vol).all()


def test_create_missing_files(tmpdir):
    new_localdb = "/" + tmpdir.relto("/")
    json_file = os.path.join(test_input_data,"example_data.json")
    ldi=LocalDataImporter(json_file=json_file,
                          output_directory=new_localdb)
    assert len(ldi.datasets)
    scan0 = ldi.datasets[0]
    create_missing_files(scan0,
            input_dir=os.path.dirname(ldi.json_file),
            output_dir=new_localdb)

    # Check the pickle file
    fop = open(os.path.join(new_localdb,scan0.pkl_path),"rb")
    test_pkl = pickle.load(fop)
    fop.close()

    fop = open(os.path.join(test_output_data,scan0.pkl_path),"rb")
    ok_pkl = pickle.load(fop)
    fop.close()
    # All the same coordinates covered?
    assert set(ok_pkl.tracks_at_ijk.keys()) == set(test_pkl.tracks_at_ijk.keys())
    # Do they point to the same streamline ids?
    for coord in ok_pkl.tracks_at_ijk.keys():
        assert test_pkl.tracks_at_ijk[coord] == ok_pkl.tracks_at_ijk[coord]

    # Check the labeling output
    for label_source in scan0.track_label_items:
        # File containing the corresponding label vector
        ok_npy_path = os.path.join(test_output_data,label_source.numpy_path)
        test_npy_path = os.path.join(new_localdb,label_source.numpy_path)
        assert (np.load(ok_npy_path) == np.load(test_npy_path)).all()


