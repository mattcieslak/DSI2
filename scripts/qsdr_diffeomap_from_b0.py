import nibabel as nib
import gzip
from scipy.io.matlab import loadmat
import numpy as np
import sys, os

usage = """
qsdr_diffeomap_from_b0.py fib.gz b0_atlas.nii.gz output_qsdr_vol.nii.gz
"""
fib_file = sys.argv[1]
b0_atlas = sys.argv[2]
output_v = sys.argv[3]

# Load the mapping from the fib file
fibf = gzip.open(fib_file,"rb")
m = loadmat(fibf)
fibf.close()
volume_dimension = m['dimension'].squeeze().astype(int)
mx = m['mx'].squeeze().astype(int)
my = m['my'].squeeze().astype(int)
mz = m['mz'].squeeze().astype(int)

# Load the QSDR template volume from DSI studio
QSDR_vol = os.path.join("/storage2/cieslak/bin/dsi_studio64/dsi_studio_64/NTU90_QA.nii.gz")
QSDR_nim = nib.load(QSDR_vol)
QSDR_data = QSDR_nim.get_data()

# Labels in b0 space
old_atlas = nib.load(b0_atlas).get_data()

# Fill up the output atlas with labels from b0,collected through the fib mappings
new_atlas = old_atlas[mx,my,mz].reshape(volume_dimension,order="F")
aff = QSDR_nim.get_affine()
aff[(0,1,2),(0,1,2)]*=2
onim = nib.Nifti1Image(new_atlas,aff)
onim.to_filename(output_v)

