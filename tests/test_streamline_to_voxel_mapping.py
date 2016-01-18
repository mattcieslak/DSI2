#!/usr/bin/env python
import nibabel as nib
from dipy.tracking import utils
import numpy as np
from dsi2.streamlines.track_math  import (trackvis_header_from_info,streamlines_to_ijk,
                streamlines_to_itk_world_coordinates, voxels_to_streamlines,
                region_pair_dict_from_roi_list,connection_ids_from_voxel_coordinates)
from create_testing_data import create_fake_fib_file
import os
from dipy.tracking.streamline import transform_streamlines
from dsi2.volumes import find_graphml_from_filename, load_lausanne_graphml
from scipy.io.matlab import loadmat
from create_testing_data2 import simulated_atlas



VOLUME_SHAPE = np.array([50, 50, 50])
VOXEL_SIZE = np.array([2.0, 2.0, 2.0])
STREAMLINE_ORI = "LPS"
FAKE_FIB_FILE="fake.fib.gz"
FAKE_TRK_FILE="fake.trk"
REFERENCE_VOL=FAKE_FIB_FILE + ".fa0.nii.gz"
TDI_OUTPUT=FAKE_TRK_FILE + ".tdi.nii.gz"
NUM_SIMULATIONS=1000
TDI_CSV=TDI_OUTPUT+".csv"
TEMP_ATLAS="temp_atlas.nii.gz"
CONN_TRK="fake_conn.trk"
FRANK_CONN="fake.fib.gz.temp_atlas.count.end.connectivity.mat"

all_rm = [FAKE_FIB_FILE,FAKE_TRK_FILE,TDI_OUTPUT,
         REFERENCE_VOL, TDI_CSV]
sim_rm = [FAKE_TRK_FILE,TDI_OUTPUT, TDI_CSV]
con_rm = [CONN_TRK, TEMP_ATLAS,FRANK_CONN]


def rm(to_rm):
    for pth in to_rm:
        if os.path.exists(pth):
            os.remove(pth)

rm(all_rm)

# Generate an output from DSI Studio
create_fake_fib_file(FAKE_FIB_FILE, VOLUME_SHAPE, VOXEL_SIZE)
os.system("dsi_studio " 
          "--action=exp --export=fa0 " 
          "--source=" +  FAKE_FIB_FILE )
ref_output = nib.load(REFERENCE_VOL)
ref_affine = ref_output.get_affine()
ref_shape = ref_output.shape
ref_voxel_size = ref_output.get_header().get_zooms()

def get_coordinate_from_ants(volume):
    os.system("ImageMath 3 " + \
          TDI_CSV+" LabelStats " + volume )
    coord = np.loadtxt(TDI_CSV, skiprows=1,delimiter=",")
    os.remove(TDI_CSV)
    return coord[:3]


def test_coordinate_transforms():
    rm(sim_rm)
    # Get a header for the test data and create a random coordinate
    header = trackvis_header_from_info( STREAMLINE_ORI, 
                                        VOLUME_SHAPE, VOXEL_SIZE)
    extents = VOLUME_SHAPE * VOXEL_SIZE
    test_coord = extents * (0.1 + 0.8*np.random.rand(3))
    streamlines = [(np.array([test_coord]*10),None,None) for x in xrange(30)]
    voxmm_streamlines = [sl[0] for sl in streamlines]
    nib.trackvis.write(FAKE_TRK_FILE,streamlines,header)
    
    # Use DSI Studio to calculate a track density map.
    os.system("dsi_studio " 
              "--action=ana --export=tdi " 
              "--source=" +  FAKE_FIB_FILE + " "
              "--tract=" + FAKE_TRK_FILE)
    from_frank = nib.load(TDI_OUTPUT)
    franks_voxel = np.unravel_index(from_frank.get_data().argmax(),from_frank.shape)
    
    ijk_streamlines = streamlines_to_ijk(voxmm_streamlines,
                                   from_frank, trackvis_header=header,return_coordinates="voxel_index")
    my_voxel = ijk_streamlines[0][0]

    assert np.all(my_voxel == franks_voxel)
    vox2vmm = voxels_to_streamlines(ijk_streamlines,ref_output,STREAMLINE_ORI)
    
    assert np.all( np.abs(test_coord - vox2vmm[0][0]) < ref_voxel_size)
    
    # Check that DSI Studio uses the same affine for TDI and fa0
    assert np.allclose(ref_affine,from_frank.get_affine())
    # Get ANTs's coordinate. We just verified that the TDI_OUTPUT coordinate
    # is the same as streamlines_to_ijk's output
    ants_coord = get_coordinate_from_ants(TDI_OUTPUT)
    
    my_coord = streamlines_to_itk_world_coordinates(
                          voxmm_streamlines,from_frank,trackvis_header=header)[0][0]
    assert np.all( np.abs(ants_coord - my_coord) < ref_voxel_size)


def simulate_connection(nib_atlas_img, regions, 
                        n_streamlines=5, curvy=False, points_per_streamline=10):
    """
    pick two regions to create a streamlines between
    """
    rm(con_rm)

    label_cube = nib_atlas_img.get_data()
    target_regions = np.random.choice(regions,size=2,replace=False)
    test_conn_id = region_pair_dict_from_roi_list(regions)[tuple(sorted(target_regions))]
    regA_voxels = [np.array(np.nonzero(label_cube == target_regions[0])).T]
    regA_streamline_coords = voxels_to_streamlines(regA_voxels,nib_atlas_img,STREAMLINE_ORI)[0]
    regA_voxels = [np.array(np.nonzero(label_cube == target_regions[0])).T]
    regA_streamline_coords = voxels_to_streamlines(regA_voxels,nib_atlas_img,STREAMLINE_ORI)[0]
    regB_voxels = [np.array(np.nonzero(label_cube == target_regions[1])).T]
    regB_streamline_coords = voxels_to_streamlines(regB_voxels,nib_atlas_img,STREAMLINE_ORI)[0]

    np.random.seed(0)
    streamlines = []
    regA_choices = np.random.choice(np.arange(len(regA_streamline_coords)),n_streamlines)
    regB_choices = np.random.choice(np.arange(len(regB_streamline_coords)),n_streamlines)
    for coord_indexA, coord_indexB in  zip(regA_choices,regB_choices):
        coordA = regA_streamline_coords[coord_indexA]
        coordB = regB_streamline_coords[coord_indexB]
        if curvy:
            raise NotImplementedError("someday...")
        else:
            streamlines.append(
                np.array([
                    np.linspace(coordA[0],coordB[0],points_per_streamline),
                    np.linspace(coordA[1],coordB[1],points_per_streamline),
                    np.linspace(coordA[2],coordB[2],points_per_streamline)
                    ]).T
            )                
    header = trackvis_header_from_info( STREAMLINE_ORI, 
                                        VOLUME_SHAPE, VOXEL_SIZE)

    nib.trackvis.write(CONN_TRK, [(sl,None,None) for sl in streamlines], hdr_mapping=header)    
    nib_atlas_img.to_filename(TEMP_ATLAS)


    os.system("dsi_studio " 
              "--action=ana --tract=" + CONN_TRK + " " 
              "--source=" +  FAKE_FIB_FILE + " --connectivity=" + TEMP_ATLAS + " "
              "--connectivity_type=end")          
    m = loadmat(FRANK_CONN)
    assert m['connectivity'].max() == n_streamlines
    frank_maxes = np.unravel_index(m['connectivity'].argmax(),m['connectivity'].shape)
    # Convert 'names'
    names = "".join(m['name'].view("S2")[0]).split("\n")
    max_rois = [int(names[i].lstrip("region")) for i in frank_maxes]
    assert set(max_rois) == set(target_regions)

def test_connectivity_matrix():
    for scale in [33, 60,125,250]:
        fake_atlas,regions = simulated_atlas(scale=scale,
                                             volume_affine=ref_affine,volume_shape=VOLUME_SHAPE)
        simulate_connection(fake_atlas, regions)


    # Check that connection_ids_from_voxel_coordinates agrees
    
    
#def test_connectivity_matrix(n_streamlines=5):
    #for scale in [33, 60,125,250]:
        #fake_atlas,regions = simulated_atlas(scale=scale,
                                             #volume_shape=VOLUME_SHAPE,
                                             #volume_affine=ref_affine)
        #fake_atlas.to_filename(TEMP_ATLAS)
        #target_regions, streamlines = simulated_connection(fake_atlas, 
                                              #output_trk_filename=CONN_TRK,
                                              #n_streamlines=n_streamlines,
                                              #return_streamlines=True)
        #os.system("dsi_studio " 
              #"--action=ana --tract=" + CONN_TRK + " " 
              #"--source=" +  FAKE_FIB_FILE + " --end=" + TEMP_ATLAS + " "
              #"--export=connectivity")          
        #m = loadmat(FRANK_CONN)
        #assert m['connectivity'].max() == n_streamlines
        #frank_maxes = np.unravel_index(m['connectivity'].argmax(),m['connectivity'].shape)
        ## Convert 'names'
        #names = "".join(m['name'].view("S2")[0]).split("\n")
        #max_rois = [int(names[i].lstrip("region")) for i in frank_maxes]
        #assert set(max_rois) == set(target_regions)
        
        ## Check that connection_ids_from_voxel_coordinates agrees
        #ijk_streamlines = streamlines_to_ijk(streamlines, fake_atlas, trackvis_header=header,
                                             #return_coordinates="voxel_index")
        #conn_ids = connection_ids_from_voxel_coordinates(ijk_streamlines,fake_atlas.get_data(), 
                                              #atlas_label_int_array=regions)
        #assert np.all(np.diff(conn_ids) == 0)
        #assert conn_ids[0] == test_conn_id
