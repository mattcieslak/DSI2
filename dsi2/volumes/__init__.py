import nibabel as nib
import os
from pkg_resources import Requirement, resource_filename

def get_MNI152():
    return nib.load(resource_filename(
                   Requirement.parse("dsi2"),
                   "example_data/MNI152_T1_2mm.nii.gz")
            )

def get_NTU90():
    return nib.load(resource_filename(
                   Requirement.parse("dsi2"),
                   "example_data/NTU90_QA.nii.gz")
            )

def find_graphml_from_b0(b0_path):
    if not b0_path: raise ValueError("Must provide a b0 volume path")
    aname = os.path.split(b0_path)[-1].lower()
    atlas_lut = {
        "scale33":"example_data/lausanne2008/resolution83/resolution83.graphml",
        "scale60":"example_data/lausanne2008/resolution150/resolution150.graphml",
        "scale125":"example_data/lausanne2008/resolution258/resolution258.graphml",
        "scale250":"example_data/lausanne2008/resolution500/resolution500.graphml",
        "scale500":"example_data/lausanne2008/resolution1015/resolution1015.graphml",
        "resolution1015":"example_data/lausanne2008/resolution1015/resolution1015.graphml",
        "resolution500":"example_data/lausanne2008/resolution500/resolution500.graphml",
        "resolution258":"example_data/lausanne2008/resolution258/resolution258.graphml",
        "resolution150":"example_data/lausanne2008/resolution150/resolution150.graphml",
        "resolution83":"example_data/lausanne2008/resolution83/resolution83.graphml"}
    atlas_id = None
    for atlas in atlas_lut.keys():
        if atlas in aname:
            print "\t\t++ "
            atlas_id = atlas
    
    if atlas_id is None: 
        return
    abs_graphml_path = resource_filename(
                Requirement.parse("dsi2"),
                atlas_lut[atlas_id])
                
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
            
