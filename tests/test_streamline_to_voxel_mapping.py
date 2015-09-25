#!/usr/bin/env python
import nibabel as nib
from dipy.tracking import utils
import numpy as np
from dsi2.streamlines.track_math  import (trackvis_header_from_info,streamlines_to_ijk,
                                          streamlines_to_itk_world_coordinates)
from create_testing_data import create_fake_fib_file
import os
from dipy.tracking.streamline import transform_streamlines    

VOLUME_SHAPE = np.array([50, 50, 50])
VOXEL_SIZE = np.array([2.0, 2.0, 2.0])
STREAMLINE_ORI = "LPS"
FAKE_FIB_FILE="fake.fib.gz"
FAKE_TRK_FILE="fake.trk"
REFERENCE_VOL=FAKE_FIB_FILE + ".fa0.nii.gz"
TDI_OUTPUT=FAKE_TRK_FILE + ".tdi.nii.gz"
NUM_SIMULATIONS=1000
TDI_CSV=TDI_OUTPUT+".csv"

all_rm = [FAKE_FIB_FILE,FAKE_TRK_FILE,TDI_OUTPUT,
         REFERENCE_VOL, TDI_CSV]
sim_rm = [FAKE_TRK_FILE,TDI_OUTPUT, TDI_CSV]


def rm(to_rm):
    for pth in to_rm:
        if os.path.exists(pth):
            os.remove(pth)

rm(all_rm)

# Generate an output from DSI Studio
create_fake_fib_file(FAKE_FIB_FILE, VOLUME_SHAPE, VOXEL_SIZE)
os.system("/home/cieslak/projects/bin/dsi_studio " 
          "--action=exp --export=fa0 " 
          "--source=" +  FAKE_FIB_FILE )
ref_output = nib.load(REFERENCE_VOL)
ref_affine = ref_output.get_affine()
ref_shape = ref_output.get_shape()

def get_coordinate_from_ants(volume):
    os.system("ImageMath 3 " + \
          TDI_CSV+" LabelStats " + volume )
    coord = np.loadtxt(TDI_CSV, skiprows=1,delimiter=",")
    os.remove(TDI_CSV)
    return coord[:3]


def simulate(test_ants_coord=True):
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
    os.system("/home/cieslak/projects/bin/dsi_studio " 
              "--action=ana --export=tdi " 
              "--source=" +  FAKE_FIB_FILE + " "
              "--tract=" + FAKE_TRK_FILE)
    from_frank = nib.load(TDI_OUTPUT)
    franks_voxel = np.unravel_index(from_frank.get_data().argmax(),from_frank.get_shape())
    
    ijk_streamlines = streamlines_to_ijk(voxmm_streamlines,
                                   from_frank, trackvis_header=header)
    my_voxel = ijk_streamlines[0][0]

    assert np.all(my_voxel == franks_voxel)
    
    # Check that DSI Studio uses the same affine for TDI and fa0
    assert np.allclose(ref_affine,from_frank.get_affine())
    # Get ANTs's coordinate. We just verified that the TDI_OUTPUT coordinate
    # is the same as streamlines_to_ijk's output
    ants_coord = get_coordinate_from_ants(TDI_OUTPUT)
    
    my_coord = streamlines_to_itk_world_coordinates(
                          voxmm_streamlines,from_frank,trackvis_header=header)
