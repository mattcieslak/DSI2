"""
A test that the b0 to qsdr mapping works
"""
import os
import nibabel as nib
import numpy as np
dsi_studio="/home/cieslak/projects/bin/dsi_studio"

# Load in an exported 4dnii from dwi simulation in DSI Studio
#simulation_qa = nib.load("qsdr_test_data/simulated_template.src.nii.gz")
#sim_data = sim.get_data()

# Load the actual QSDR template but fill it with QA from the phantom
dtemp = nib.load("qsdr_test_data/HCP488_QA.nii.gz")
template_sim = np.zeros([
    dtemp.shape[0], dtemp.shape[1],
    dtemp.shape[2]])
simulated_qa0 = nib.load("qsdr_test_data/orig_qa0.nii.gz")
sim_qa = simulated_qa0.get_data()


R_offset = template_sim.shape[0] // 2 -10
A_offset = template_sim.shape[1] // 2 - 10
S_offset = template_sim.shape[2] // 2 - 10 

template_sim[
    R_offset:(R_offset+sim_qa.shape[0]),
    A_offset:(A_offset+sim_qa.shape[1]),
    S_offset:(S_offset+sim_qa.shape[2])
    ] = sim_qa[::-1,::-1,:]
nib.Nifti1Image(template_sim,dtemp.get_affine()).to_filename(
    "qsdr_test_data/qa_sim_template.nii.gz")


os.system( dsi_studio + \
            " --action=rec --method=7 "
            "--output_map=1 "
            "--source=qsdr_test_data/simulated_template_snr70_fa7_dif1.25_n2.src "
            "--template=" + os.path.abspath("qsdr_test_data/qa_sim_template.nii.gz") +
            " --param0=1.25 --param1=2" )

from glob import glob
qsdr_fib = glob("qsdr_test_data/simulated*qsdr*fib.gz")[0]

def check_r1l2_orientation(img_path):
    img = nib.load(img_path)
    wd = img.get_data()
    lh_roi_center    = np.array(np.nonzero(wd ==2)).mean(1)
    rh_roi_center   = np.array(np.nonzero(wd ==1)).mean(1)
    rh_greater = img.get_affine()[0,0] > 0
    if rh_greater:
        return lh_roi_center[0] < rh_roi_center[0]
    return lh_roi_center[0] > rh_roi_center[0]

def test_b0_to_qsdr_map():
    from dsi2.volumes import b0_to_qsdr_map
    LATERALITY_INPUT="qsdr_test_data/native_space_R1_L2.nii.gz"
    LATERALITY_OUTPUT="qsdr_test_data/warped_regions.nii.gz"
    assert check_r1l2_orientation(LATERALITY_INPUT) 
    b0_to_qsdr_map(qsdr_fib,LATERALITY_INPUT,LATERALITY_OUTPUT)
    assert check_r1l2_orientation(LATERALITY_OUTPUT) 
    
                   