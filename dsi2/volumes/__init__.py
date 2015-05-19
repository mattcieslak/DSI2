import nibabel as nib
import os
import numpy as np
from dsi2.config import dsi2_data_path
from dsi2.streamlines.track_math import region_pair_dict_from_roi_list
import networkx as nx

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
                   os.path.join(dsi2_data_path,
                   "NTU90_QA.nii.gz")
            )

def find_graphml_from_b0(b0_path):
    if not b0_path: raise ValueError("Must provide a b0 volume path")
    aname = os.path.split(b0_path)[-1].lower()
    atlas_lut = {
        "scale33":"lausanne2008/resolution83/resolution83.graphml",
        "scale60":"lausanne2008/resolution150/resolution150.graphml",
        "scale125":"lausanne2008/resolution258/resolution258.graphml",
        "scale250":"lausanne2008/resolution500/resolution500.graphml",
        "scale500":"lausanne2008/resolution1015/resolution1015.graphml",
        "resolution1015":"lausanne2008/resolution1015/resolution1015.graphml",
        "resolution500":"lausanne2008/resolution500/resolution500.graphml",
        "resolution258":"lausanne2008/resolution258/resolution258.graphml",
        "resolution150":"lausanne2008/resolution150/resolution150.graphml",
        "resolution83":"lausanne2008/resolution83/resolution83.graphml"}
    atlas_id = None
    for atlas in atlas_lut.keys():
        if atlas in aname:
            atlas_id = atlas

    if atlas_id is None:
        return
    abs_graphml_path = os.path.join( dsi2_data_path,atlas_lut[atlas_id])

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
            

# Shapes of the volume data
QSDR_SHAPE = (79,95,69) # Shape of QSDR output
QSDR_AFFINE = np.array(
      [[ -2.,   0.,   0.,  78.],
       [  0.,  -2.,   0.,  76.],
       [  0.,   0.,   2., -50.],
       [  0.,   0.,   0.,   1.]])


def load_lausanne_graphml(graphml):
    graph = nx.read_graphml(graphml)
    graphml_data = {'region_labels':{},
                    'region_pairs_to_index':{},
                    'index_to_region_pairs':{},
                    'region_pair_strings_to_index':{},
                    'regions':[]}
    for roi_id,roi_data in graph.nodes(data=True):
        graphml_data['region_labels'][roi_id] = roi_data
    graphml_data['regions'] = np.array(sorted(map( int,graph.nodes() )))
    
    graphml_data['region_pairs_to_index'] = region_pair_dict_from_roi_list(
                                                       graphml_data['regions'])
    
    # Which regionpairs map to which unique id in this dataset?
    graphml_data['index_to_region_pairs'] = dict(
        [
        (idxnum,(graphml_data['region_labels'][str(id1)]['dn_name'],
                   graphml_data['region_labels'][str(id2)]['dn_name']) ) \
         for (id1,id2), idxnum in graphml_data['region_pairs_to_index'].iteritems()
        ]
    )
    graphml_data['region_pair_strings_to_index'] = dict([
        (value,key) for key,value in graphml_data['index_to_region_pairs'].iteritems()
    ])
    return graphml_data
