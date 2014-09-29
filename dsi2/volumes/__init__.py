import nibabel as nib
from pkg_resources import Requirement, resource_filename

def get_MNI152():
    return nib.load(resource_filename(
                   Requirement.parse("dsi2"),
                   "example_data/MNI152_T1_2mm.nii.gz")
            )
