import nibabel as nib
import os
import numpy as np
from dsi2.config import dsi2_data_path

def get_MNI152_path():
    return os.path.join(dsi2_data_path,
                   "MNI152_T1_2mm.nii.gz")

def get_MNI152():
    return nib.load(get_MNI152_path())


def save_coords_to_volume(coords,outpath):
    """
    Parameters:
    -----------------
    coords: List of 3-tuples
    outpath:path to output (.nii[.gz])
    """
    img = get_MNI152()
    data = np.zeros_like(img.get_data())
    hdr = img.get_header()
    affine = img.get_affine()
    ix, jx, kx = np.array(coords).T
    
    data[ix,jx,kx] = 1
    
    new_img = nib.Nifti1Image(data, affine=affine, header=hdr)
    new_img.to_filename(outpath)
    

def get_NTU90():
    return nib.load(
                   os.path.join(dsi2_data_dir,
                   "NTU90_QA.nii.gz")
            )

def find_graphml_from_b0(b0_path):
    if not b0_path: raise ValueError("Must provide a b0 volume path")
    aname = os.path.split(b0_path)[-1].lower()
    atlas_lut = {
        "scale33":"dsi2/example_data/lausanne2008/resolution83/resolution83.graphml",
        "scale60":"dsi2/example_data/lausanne2008/resolution150/resolution150.graphml",
        "scale125":"dsi2/example_data/lausanne2008/resolution258/resolution258.graphml",
        "scale250":"dsi2/example_data/lausanne2008/resolution500/resolution500.graphml",
        "scale500":"dsi2/example_data/lausanne2008/resolution1015/resolution1015.graphml",
        "resolution1015":"dsi2/example_data/lausanne2008/resolution1015/resolution1015.graphml",
        "resolution500":"dsi2/example_data/lausanne2008/resolution500/resolution500.graphml",
        "resolution258":"dsi2/example_data/lausanne2008/resolution258/resolution258.graphml",
        "resolution150":"dsi2/example_data/lausanne2008/resolution150/resolution150.graphml",
        "resolution83":"dsi2/example_data/lausanne2008/resolution83/resolution83.graphml"}
    atlas_id = None
    for atlas in atlas_lut.keys():
        if atlas in aname:
            print "\t\t++ "
            atlas_id = atlas

    if atlas_id is None:
        return
    abs_graphml_path = os.path.join( dsi2_data_dir,atlas_lut[atlas_id])

    return abs_graphml_path


def graphml_from_label_source(label_source):
    if os.path.exists(label_source.graphml_path):
        print "\t\t++ Loading user supplied graphml path"
        return atlas_name
    
    return find_graphml_from_b0(label_source.b0_volume_path)


def get_builtin_atlas_parameters(label_source):
    if not label_source: raise ValueError("Must provide a b0 volume path")
    aname = os.path.split(label_source)[-1].lower()
    atlas_lut = {
        "scale33":{"scale":33,"param1":1},
        "scale60":{"scale":60,"param1":1},
        "scale125":{"scale":125,"param1":1},
        "scale250":{"scale":250,"param1":1},
        "scale500":{"scale":500,"param1":1},
        "resolution1015":{"scale":500,"param1":1},
        "resolution500":{"scale":250,"param1":1},
        "resolution258":{"scale":125,"param1":1},
        "resolution150":{"scale":60,"param1":1},
        "resolution83":{"scale":33,"param1":1}
    }
    atlas_id = None
    for atlas in atlas_lut.keys():
        if atlas in aname:
            atlas_id = atlas
    
    if atlas_id is None: 
        return {}
    return atlas_lut[atlas_id]
            
