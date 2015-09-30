#!/usr/bin/env python

"""
Creates a simple testing dataset that will be 
easily verifiable after import

Contents:
----------
 * Two fake Lausanne atlases, consisting of labeled cubes
   * scale60.nii.gz
   * scale33.nii.gz
"""

import numpy as np
import nibabel as nib
from scipy.io.matlab import savemat
from gzip import open as gzopen
from dsi2.streamlines.track_dataset import TrackDataset
from dsi2.volumes import get_NTU90
import os
import json
import os.path as op
from dsi2.volumes import QSDR_SHAPE, QSDR_AFFINE, find_graphml_from_filename, load_lausanne_graphml
from dsi2.database.traited_query import Scan, TrackLabelSource, TrackScalarSource
from dsi2.aggregation import make_aggregator
import shutil
from copy import deepcopy
import pytest
from dsi2.streamlines import qsdr_2mm_trk_header as qsdr_trk_header



scale33_graphml = find_graphml_from_filename("scale33")
scale33_data = load_lausanne_graphml(scale33_graphml)
scale60_graphml = find_graphml_from_filename("scale60")
scale60_data = load_lausanne_graphml(scale60_graphml)

ORIG_SHAPE = (79,95,69) # Shape of original data
BRAINSTEM_ID = 83
L_PRECEN_33 = 51
R_PRECEN_33 = 10
L_PRECEN_1_60 = 80
L_PRECEN_2_60 = 81
R_PRECEN_1_60 = 16
R_PRECEN_2_60 = 17

# Connection IDS
SCALE33_RPC_TO_BS = scale33_data['region_pairs_to_index'][(R_PRECEN_33,BRAINSTEM_ID)]
SCALE33_LPC_TO_BS = scale33_data['region_pairs_to_index'][(L_PRECEN_33,BRAINSTEM_ID)]
SCALE60_RPC_1_TO_BS = scale60_data['region_pairs_to_index'][(R_PRECEN_1_60,BRAINSTEM_ID)]
SCALE60_LPC_1_TO_BS = scale60_data['region_pairs_to_index'][(L_PRECEN_1_60,BRAINSTEM_ID)]
SCALE60_RPC_2_TO_BS = scale60_data['region_pairs_to_index'][(R_PRECEN_2_60,BRAINSTEM_ID)]
SCALE60_LPC_2_TO_BS = scale60_data['region_pairs_to_index'][(L_PRECEN_2_60,BRAINSTEM_ID)]


def test_region_label_names():
    assert scale33_data['region_labels'][str(BRAINSTEM_ID)]['dn_name'] == "Brain-Stem"
    assert scale33_data['region_labels'][str(R_PRECEN_33)]['dn_name'] == "rh.precentral"
    assert scale33_data['region_labels'][str(L_PRECEN_33)]['dn_name'] == "lh.precentral"
    assert scale60_data['region_labels'][str(R_PRECEN_1_60)]['dn_name'] == "rh.precentral_1"
    assert scale60_data['region_labels'][str(L_PRECEN_1_60)]['dn_name'] == "lh.precentral_1"
    assert scale60_data['region_labels'][str(R_PRECEN_2_60)]['dn_name'] == "rh.precentral_2"
    assert scale60_data['region_labels'][str(L_PRECEN_2_60)]['dn_name'] == "lh.precentral_2"
    

    
# First, create the QSDR atlases
q_30 = np.zeros(QSDR_SHAPE)
q_60 = np.zeros(QSDR_SHAPE)

def make_coords(xmin,xmax,ymin,ymax,zmin,zmax):
    return np.array(
    [d.flatten() for d in np.mgrid[
        xmin:(xmax+1), ymin:(ymax+1), zmin:(zmax+1)]])

def save_fake_fib(fname):
    """returns a dict to get saved"""
    inds = np.arange(QSDR_SHAPE[0] * QSDR_SHAPE[1] * QSDR_SHAPE[2])
    mx, my, mz = np.unravel_index(inds,QSDR_SHAPE,order="F")
    fop = gzopen(fname,"wb")
    savemat(fop,
            {"dimension":np.array(QSDR_SHAPE),
                     "mx":mx,"my":my,"mz":mz},
            format='4'
            )
    fop.close()
    


# Create scale 33 precentral, left and right
L_PRE_CEN_MIN_X=50
L_PRE_CEN_MAX_X=58
L_PRE_CEN_MIN_Y=40
L_PRE_CEN_MAX_Y=51
L_PRE_CEN_MIN_Z=45
L_PRE_CEN_MAX_Z=50

R_PRE_CEN_MIN_X=20
R_PRE_CEN_MAX_X=28
R_PRE_CEN_MIN_Y=40
R_PRE_CEN_MAX_Y=51
R_PRE_CEN_MIN_Z=45
R_PRE_CEN_MAX_Z=50

BS_MIN_X=36
BS_MAX_X=40
BS_MIN_Y=46
BS_MAX_Y=50
BS_MIN_Z=14
BS_MAX_Z=20

# Create the brainstem "bs" region, same for both scales
bs_i, bs_j, bs_k = make_coords(
    BS_MIN_X, BS_MAX_X,
    BS_MIN_Y, BS_MAX_Y,
    BS_MIN_Z, BS_MAX_Z)
bs_coord = np.array([bs_i,bs_j,bs_k]).T

left_pre_cen33_i,left_pre_cen33_j,left_pre_cen33_k = make_coords(
    L_PRE_CEN_MIN_X,L_PRE_CEN_MAX_X, 
    L_PRE_CEN_MIN_Y,L_PRE_CEN_MAX_Y,
    L_PRE_CEN_MIN_Z,L_PRE_CEN_MAX_Z)
right_pre_cen33_i,right_pre_cen33_j,right_pre_cen33_k = make_coords(
    R_PRE_CEN_MIN_X,R_PRE_CEN_MAX_X, 
    R_PRE_CEN_MIN_Y,R_PRE_CEN_MAX_Y,
    R_PRE_CEN_MIN_Z,R_PRE_CEN_MAX_Z)

# Create scale 60 precentral, left and right
# Use the midpoints in y to divide the original regions
L_PRE_CEN_MID = (L_PRE_CEN_MAX_Y + L_PRE_CEN_MIN_Y) // 2
R_PRE_CEN_MID = (R_PRE_CEN_MAX_Y + R_PRE_CEN_MIN_Y) // 2

left_pre_cen60_1_i,left_pre_cen60_1_j,left_pre_cen60_1_k = make_coords(
    L_PRE_CEN_MIN_X,L_PRE_CEN_MAX_X, 
    L_PRE_CEN_MIN_Y,L_PRE_CEN_MID,
    L_PRE_CEN_MIN_Z,L_PRE_CEN_MAX_Z)
lpc60_1_coord = np.array(
    [left_pre_cen60_1_i,left_pre_cen60_1_j,left_pre_cen60_1_k]).T
left_pre_cen60_2_i,left_pre_cen60_2_j,left_pre_cen60_2_k = make_coords(
    L_PRE_CEN_MIN_X,L_PRE_CEN_MAX_X, 
    L_PRE_CEN_MID+1,L_PRE_CEN_MAX_Y,
    L_PRE_CEN_MIN_Z,L_PRE_CEN_MAX_Z)
lpc60_2_coord = np.array(
    [left_pre_cen60_2_i,left_pre_cen60_2_j,left_pre_cen60_2_k]).T
                
right_pre_cen60_1_i,right_pre_cen60_1_j,right_pre_cen60_1_k = make_coords(
    R_PRE_CEN_MIN_X,R_PRE_CEN_MAX_X, 
    R_PRE_CEN_MIN_Y,R_PRE_CEN_MID,
    R_PRE_CEN_MIN_Z,R_PRE_CEN_MAX_Z)
rpc60_1_coord = np.array(
    [right_pre_cen60_1_i,right_pre_cen60_1_j,right_pre_cen60_1_k]).T
right_pre_cen60_2_i,right_pre_cen60_2_j,right_pre_cen60_2_k = make_coords(
    R_PRE_CEN_MIN_X,R_PRE_CEN_MAX_X, 
    R_PRE_CEN_MID+1,R_PRE_CEN_MAX_Y,
    R_PRE_CEN_MIN_Z,R_PRE_CEN_MAX_Z)
rpc60_2_coord = np.array(
    [right_pre_cen60_2_i,right_pre_cen60_2_j,right_pre_cen60_2_k]).T

def get_scale33_nim():
    newdata = np.zeros(QSDR_SHAPE)
    
    newdata[bs_i, bs_j, bs_k] = BRAINSTEM_ID
    newdata[left_pre_cen33_i,
            left_pre_cen33_j,
            left_pre_cen33_k] = L_PRECEN_33
    
    newdata[right_pre_cen33_i,
            right_pre_cen33_j,
            right_pre_cen33_k] = R_PRECEN_33
    return nib.Nifti1Image(newdata,QSDR_AFFINE)
    
def get_scale60_nim():
    newdata = np.zeros(QSDR_SHAPE)
    
    newdata[bs_i, bs_j, bs_k] = BRAINSTEM_ID
    newdata[left_pre_cen60_1_i,
            left_pre_cen60_1_j,
            left_pre_cen60_1_k] = L_PRECEN_1_60
    newdata[left_pre_cen60_2_i,
            left_pre_cen60_2_j,
            left_pre_cen60_2_k] = L_PRECEN_2_60
    
    newdata[right_pre_cen60_1_i,
            right_pre_cen60_1_j,
            right_pre_cen60_1_k] = R_PRECEN_1_60
    newdata[right_pre_cen60_2_i,
            right_pre_cen60_2_j,
            right_pre_cen60_2_k] = R_PRECEN_2_60
    return nib.Nifti1Image(newdata,QSDR_AFFINE)
    

def create_streamlines(from_coords, to_coords):
    streamlines = []
    for from_coord, to_coord in zip(from_coords,to_coords):
        fx,fy,fz = from_coord
        tx,ty,tz = to_coord
        distance_mm = int(
            np.sqrt(np.sum((from_coord-to_coord)**2)))
        
        streamlines.append( 
            np.array([
                np.linspace(fx,tx,distance_mm,endpoint=True)*2,
                np.linspace(fy,ty,distance_mm,endpoint=True)*2,
                np.linspace(fz,tz,distance_mm,endpoint=True)*2
            ]).T 
        )
    return streamlines

def create_fake_fib_file(fname, 
                                             volume_shape = np.array([50,50,50]),
                                             voxel_size=np.array([2.0,2.0,2.0]),
                                             n_odfs=10,
                                             odf_resolution=3
                                             ):
    """
    Creates a small fib.gz file
    """
    import gzip
    from scipy.io.matlab import savemat
    from dipy.core.subdivide_octahedron import create_unit_sphere
    
    sphere = create_unit_sphere(odf_resolution)
    fa0 = np.zeros(np.prod(volume_shape))
    index0 = np.zeros(np.prod(volume_shape))
    fa0[:n_odfs] = 1
    index0[:n_odfs] = 1
    
    fop = gzip.open(fname,"wb")
    savemat(fop, {
            "dimension": volume_shape,
            "fa0": fa0,
            "odf_vertices": sphere.vertices,
            "odf_faces": sphere.faces,
            "index0":index0,
            "voxel_size":voxel_size
            },
        format="4"
        )
    fop.close()
    
    
def get_streamlines1():
    """simulated control dataset"""
    scale33 = []
    scale60 = []
    
    np.random.seed(0)
    sl_bs_lpc1 = bs_coord[np.random.choice(len(bs_i),20)]
    scale33 += [ SCALE33_LPC_TO_BS ] * 20
    np.random.seed(0)
    sl_lpc1_bs = lpc60_1_coord[np.random.choice(len(left_pre_cen60_1_i),20)]
    scale60 += [ SCALE60_LPC_1_TO_BS ] * 20
    
    np.random.seed(1)
    sl_bs_lpc2 = bs_coord[np.random.choice(len(bs_i),20)]
    scale33 += [ SCALE33_LPC_TO_BS ] * 20
    np.random.seed(1)
    sl_lpc2_bs = lpc60_2_coord[np.random.choice(len(left_pre_cen60_2_i),20)]
    scale60 += [ SCALE60_LPC_2_TO_BS ] * 20
    
    np.random.seed(2)
    sl_bs_rpc1 = bs_coord[np.random.choice(len(bs_i),20)]
    scale33 += [ SCALE33_RPC_TO_BS ] * 20
    np.random.seed(2)
    sl_rpc1_bs = rpc60_1_coord[np.random.choice(len(right_pre_cen60_1_i),20)]
    scale60 += [ SCALE60_RPC_1_TO_BS ] * 20
    
    np.random.seed(3)
    sl_bs_rpc2 = bs_coord[np.random.choice(len(bs_i),20)]
    scale33 += [ SCALE33_RPC_TO_BS ] * 20
    np.random.seed(3)
    sl_rpc2_bs = rpc60_2_coord[np.random.choice(len(right_pre_cen60_2_i),20)]
    scale60 += [ SCALE60_RPC_2_TO_BS ] * 20
    
    # create the streamlines from left pre central
    streamlines = []
    streamlines += create_streamlines(sl_bs_lpc1, sl_lpc1_bs)
    streamlines += create_streamlines(sl_bs_lpc2, sl_lpc2_bs)
    streamlines += create_streamlines(sl_bs_rpc1, sl_rpc1_bs)
    streamlines += create_streamlines(sl_bs_rpc2, sl_rpc2_bs)
    
    hdr = deepcopy(qsdr_trk_header)
    hdr['n_count'] = len(streamlines)
    tds1 = TrackDataset(header=hdr, tracks=streamlines)

    return tds1, np.array(scale33), np.array(scale60)
tds1, tds1_scale33, tds1_scale60 = get_streamlines1()

def get_streamlines2():
    """ Simulated "damaged" dataset
    """
    scale33 = []
    scale60 = []
    # limited y on lpc1 
    lpc_1_small_i,lpc_1_small_j,lpc_1_small_k = make_coords(
        L_PRE_CEN_MIN_X,L_PRE_CEN_MAX_X, 
        L_PRE_CEN_MIN_Y+4,L_PRE_CEN_MID,
        L_PRE_CEN_MIN_Z,L_PRE_CEN_MAX_Z)
    lpc_1_small_coord = np.array([lpc_1_small_i,lpc_1_small_j,lpc_1_small_k]).T
    np.random.seed(4)
    sl_bs_lpc1_small = lpc_1_small_coord[np.random.choice(len(lpc_1_small_i),10)]
    np.random.seed(4)
    bs_small_i,bs_small_j,bs_small_k = make_coords( 
        BS_MIN_X, BS_MAX_X,
        BS_MIN_Y+4, BS_MAX_Y,
        BS_MIN_Z, BS_MAX_Z)
    bs_small_coord = np.array([ bs_small_i, bs_small_j, bs_small_k ]).T
    sl_lpc1_bs_small = bs_small_coord[np.random.choice(len(bs_small_coord),10)]
    scale60 += [ SCALE60_LPC_1_TO_BS ] * 10
    scale33 += [ SCALE33_LPC_TO_BS ] * 10
    
    np.random.seed(5)
    sl_bs_lpc2 = bs_coord[np.random.choice(len(bs_i),20)]
    scale33 += [ SCALE33_LPC_TO_BS ] * 20
    np.random.seed(5)
    sl_lpc2_bs = lpc60_2_coord[np.random.choice(len(left_pre_cen60_2_i),20)]
    scale60 += [ SCALE60_LPC_2_TO_BS ] * 20
    
    np.random.seed(6)
    sl_bs_rpc1 = bs_coord[np.random.choice(len(bs_i),20)]
    scale33 += [ SCALE33_RPC_TO_BS ] * 20
    np.random.seed(6)
    sl_rpc1_bs = rpc60_1_coord[np.random.choice(len(right_pre_cen60_1_i),20)]
    scale60 += [ SCALE60_RPC_1_TO_BS ] * 20
    
    np.random.seed(7)
    sl_bs_rpc2 = bs_coord[np.random.choice(len(bs_i),30)]
    scale33 += [ SCALE33_RPC_TO_BS ] * 30
    np.random.seed(7)
    sl_rpc2_bs = rpc60_2_coord[np.random.choice(len(right_pre_cen60_2_i),30)]
    scale60 += [ SCALE60_RPC_2_TO_BS ] * 30
    
    # create the streamlines from left pre central
    streamlines = []
    streamlines += create_streamlines(sl_bs_lpc1_small, sl_lpc1_bs_small)
    streamlines += create_streamlines(sl_bs_lpc2, sl_lpc2_bs)
    streamlines += create_streamlines(sl_bs_rpc1, sl_rpc1_bs)
    streamlines += create_streamlines(sl_bs_rpc2, sl_rpc2_bs)
    
    hdr = deepcopy(qsdr_trk_header)
    hdr['n_count'] = len(streamlines)
    tds1 = TrackDataset(header=hdr, tracks=streamlines)
    return tds1, np.array(scale33), np.array(scale60)

tds2, tds2_scale33, tds2_scale60 = get_streamlines2()

@pytest.fixture(scope="session")
def create_test_data(request):
    from paths import test_input_data, input_data_json, test_output_data
    for directory in [test_input_data,test_output_data]:
        if os.path.exists(directory):
            print "removing previous testing data"
            os.system("rm -rf " + directory)
    print "creating new testing directory"
    os.makedirs(test_input_data)
    droot = test_input_data
    print "saving data to", droot, "sumary at", input_data_json
    datasets = []
    def save_subject(subjname, tracks):
        subjdir = op.join(droot,subjname)
        trk_pth = op.join(subjdir,subjname+".trk")
        pkl_trk_pth = op.join(subjdir,subjname+".pkl.trk")
        scale33 = op.join(subjdir,subjname+".scale33.nii.gz")
        scale33_q = op.join(subjdir,subjname+".scale33.QSDR.nii.gz")
        scale60 = op.join(subjdir,subjname+".scale60.nii.gz")
        scale60_q = op.join(subjdir,subjname+".scale60.QSDR.nii.gz")
        fib = op.join(subjdir,subjname+".fib.gz")
        os.makedirs(subjdir)
        tracks.save(trk_pth,use_qsdr_header=True)
        get_scale33_nim().to_filename(scale33)
        get_scale60_nim().to_filename(scale60)
        get_scale33_nim().to_filename(scale33_q)
        get_scale60_nim().to_filename(scale60_q)
        save_fake_fib(fib)
        sc =  Scan(
                scan_id=subjname,
                subject_id=subjname,
                study="testing",
                scan_group="fake_brain",
                scan_gender="F",
                streamline_space="qsdr",
                software="DSI Studio",
                reconstruction="qsdr",
                pkl_path=op.join(subjdir,subjname+".pkl"),
                fib_file=fib,
                trk_file=trk_pth,
                pkl_trk_path=pkl_trk_pth)
        sc.track_label_items = [
                    TrackLabelSource(
                        name="Lausanne2008",
                        parameters={"scale":33,"thick":1},
                        numpy_path=op.join(subjdir,"scale33.npy"),
                        b0_volume_path=scale33,
                        qsdr_volume_path=scale33_q
                        ),
                    TrackLabelSource(
                        name="Lausanne2008",
                        parameters={"scale":60,"thick":1},
                        numpy_path=op.join(subjdir,"scale60.npy"),
                        b0_volume_path=scale60,
                        qsdr_volume_path=scale60_q
                        )
                    ]
        datasets.append(sc)
                
    save_subject("s1", tds1)
    save_subject("s2", tds2)

    json_data = [scan.to_json() for scan in datasets]
    with open(input_data_json,"w") as outfile:
        json.dump(json_data,outfile,indent=4)
    print "Saved", input_data_json
    def teardown():
        print "Teardown of test input data"
        shutil.rmtree(test_input_data)
        shutil.rmtree(test_output_data)
    if not request is None:
        request.addfinalizer(teardown)
    
    return input_data_json

def mlab_show_test_dataset():
    from mayavi import mlab
    mlab.points3d(bs_i, bs_j, bs_k,mode="cube",color=(1,1,0))
    # Left pre-central
    mlab.points3d(left_pre_cen60_1_i, left_pre_cen60_1_j,
                  left_pre_cen60_1_k,mode="cube",color=(0,1,1),
                  opacity=0.5)
    mlab.points3d(left_pre_cen60_2_i, left_pre_cen60_2_j,
                  left_pre_cen60_2_k,mode="cube",color=(1,0,1),
                  opacity=0.5)
    # Right pre-central 
    mlab.points3d(right_pre_cen60_1_i, right_pre_cen60_1_j,
                  right_pre_cen60_1_k,mode="cube",color=(0,1,0),
                  opacity=0.5)
    mlab.points3d(right_pre_cen60_2_i, right_pre_cen60_2_j,
                  right_pre_cen60_2_k,mode="cube",color=(0,0,1),
                  opacity=0.5)
    tds1.render_tracks = True
    tds1.dynamic_color_clusters = False
    tds1.static_color = "red"
    tds1.draw_tracks()
    tds2.render_tracks = True
    tds2.dynamic_color_clusters = False
    tds2.static_color = "blue"
    tds2.draw_tracks()
