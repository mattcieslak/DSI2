#!/usr/bin/env python
import sys
import os

import paths

import dsi2.config
dsi2.config.local_trackdb_path = paths.test_input_data

import dsi2.streamlines.track_dataset as track_dataset

import nibabel as nib
import numpy as np
import cPickle as pickle

def test_join_tracks():
    fop = open(os.path.join(paths.test_output_data, "0377A/0377A.src.gz.odf8.f3.reg1.qsdr.1.25.2mm.map.fib.gz.nsm.MNI.pkl"), "rb")
    tds1 = pickle.load(fop)
    fop.close()
    fop = open(os.path.join(paths.test_output_data, "2843A/2843A.src.gz.odf8.f3.reg1.qsdr.1.25.2mm.map.fib.gz.nsm.MNI.pkl"), "rb")
    tds2 = pickle.load(fop)
    fop.close()
  
    tds3 = track_dataset.join_tracks([tds1, tds2])

    assert len(tds3.tracks) == len(tds1.tracks) + len(tds2.tracks)
    # TODO: check that each track in tds3 comes from tds1 and tds2

def test_subset():
    fop = open(os.path.join(paths.test_output_data, "0377A/0377A.src.gz.odf8.f3.reg1.qsdr.1.25.2mm.map.fib.gz.nsm.MNI.pkl"), "rb"    )
    tds1 = pickle.load(fop)
    fop.close()

    indices = [12, 78, 99, 107]

    tds_subset = tds1.subset(indices)
    assert len(tds_subset.tracks) == 4
    assert tds_subset.original_track_indices == indices

    tds_subset = tds1.subset(indices, every=2)
    assert len(tds_subset.tracks) == 2
    assert tds_subset.original_track_indices == indices[1::2]
    
    tds_subset = tds1.subset(indices, inverse=True)
    assert len(tds_subset.tracks) == 99996

    tds_subset = tds1.subset(indices, inverse=True, every=2)
    assert len(tds_subset.tracks) == 49998
    
    tds_subset = tds1.subset(indices, inverse=True, every=1)
    assert len(tds_subset.tracks) == 99996

def test_length_filter():
    fop = open(os.path.join(paths.test_output_data, "0377A/0377A.src.gz.odf8.f3.reg1.qsdr.1.25.2mm.map.fib.gz.nsm.MNI.pkl"), "rb"    )
    tds1 = pickle.load(fop)
    fop.close()
    
    # this should do nothing
    tds1.length_filter()
    assert len(tds1.tracks) == 100000

    # TODO: why do these fail?
    tds_filtered = tds1.length_filter(minlength=100, new=True)

    tds1.length_filter(minlength=100)
