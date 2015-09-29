import nibabel as nib
import os
import gzip
import numpy as np
from dsi2.config import dsi2_data_path
from dsi2.streamlines.track_math import region_pair_dict_from_roi_list
import networkx as nx
from scipy.io.matlab import loadmat

def get_region_ints_from_graphml(graphml):
    """
    Returns an array of region ints from a graphml file.
    """
    graph = nx.read_graphml(graphml)
    return sorted(map(int, graph.nodes()))

def get_MNI152_path():
    return os.path.join(dsi2_data_path,
                   "MNI152_T1_2mm.nii.gz")

def get_MNI152():
    return nib.load(get_MNI152_path())

def get_NTU90():
    return nib.load(
                   os.path.join(dsi2_data_path,
                   "NTU90_QA.nii.gz")
            )

def find_graphml_from_filename(b0_path):
    if not b0_path: raise ValueError("Must provide a  volume path")
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
    
    return find_graphml_from_filename(label_source.b0_volume_path)


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
QSDR_VOXEL_SIZE=np.array([2.0, 2.0, 2.0])


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

def get_lausanne_spec(scale):
    
    lausanne_scale_lookup = {
                  33:resource_filename(Requirement.parse("dsi2"),
                    "dsi2/example_data/lausanne2008/resolution83/resolution83.graphml"),
                  60:resource_filename(Requirement.parse("dsi2"),
                    "dsi2/example_data/lausanne2008/resolution150/resolution150.graphml"),
                  125:resource_filename(Requirement.parse("dsi2"),
                    "dsi2/example_data/lausanne2008/resolution258/resolution258.graphml"),
                  250:resource_filename(Requirement.parse("dsi2"),
                    "dsi2/example_data/lausanne2008/resolution500/resolution500.graphml"),
                  500:resource_filename(Requirement.parse("dsi2"),
                    "dsi2/example_data/lausanne2008/resolution1015/resolution1015.graphml")
                  }

    return load_lausanne_graphml(lausanne_scale_lookup[scale])

def get_fib(fib):
    if isinstance(fib_file,basestring):
        if fib.endswith("gz"):
            fibf = gzip.open(fib_file,"rb")
        else:
            fibf = gzip.open(fib_file,"rb")
        m = loadmat(fibf)
        fibf.close()
        return m
    elif isinstance(fib, dict):
        return m
    else:
        raise ValueError("Unable to load " + fib)

def b0_to_qsdr_map(fib_file, b0_atlas, output_v):
    """
    Creates a qsdr atlas from a DSI Studio fib file and a b0 atlas.
    
    Parameters:
    =======
    fib_file:str or dict
      Either the path to a fib file on disk, or a loaded fib file
    b0_atlas:str or nib.Nifti1Image
      Path to an atlas nifti file OR a nib.Nifti1Image
    output_v:str or nib.Nifti1Image
      Path where output resides
    """
    # Load the mapping from the fib file
    m = get_fib(fib_file)
    volume_dimension = m['dimension'].squeeze().astype(int)
    if 'mx'  in m.keys():
        xvar, yvar,zvar = "mx", "my", "mx"
    elif "_x" in m.keys():
        xvar, yvar,zvar = "_x", "_y", "_x"
        
    mx = m[xvar].squeeze().astype(int)
    my = m[yvar].squeeze().astype(int)
    mz = m[zvar].squeeze().astype(int)

    # Labels in b0 space
    _old_atlas = nib.load(b0_atlas)
    old_atlas = _old_atlas.get_data()
    old_aff = _old_atlas.get_affine()
    # QSDR maps from LPS+ space.  Force the input volume to conform
    if old_aff[0,0] > 0:
        print "\t\t+++ Flipping X"
        old_atlas = old_atlas[::-1,:,:]
    if old_aff[1,1] > 0:
        print "\t\t+++ Flipping Y"
        old_atlas = old_atlas[:,::-1,:]
    if old_aff[2,2] < 0:
        print "\t\t+++ Flipping Z"
        old_atlas = old_atlas[:,:,::-1]
        
    # XXX: there is an error when importing some of the HCP datasets where the
    # map-from index is out of bounds from the b0 image. This will check for
    # any indices that would cause an index error and sets them to 0.
    bx, by, bz = old_atlas.shape
    idx_err_x = np.flatnonzero( mx >= bx)
    if len(idx_err_x):
        print "\t\t+++ WARNING: %d voxels are out of original data x range" % len(idx_err_x)
        mx[idx_err_x] = 0
    idx_err_y = np.flatnonzero( my >= by)
    if len(idx_err_y):
        print "\t\t+++ WARNING: %d voxels are out of original data y range" % len(idx_err_y)
        my[idx_err_y] = 0
    idx_err_z = np.flatnonzero( mz >= bz)
    if len(idx_err_z):
        print "\t\t+++ WARNING: %d voxels are out of original data z range" % len(idx_err_z)
        mz[idx_err_z] = 0
        
    
    # Fill up the output atlas with labels from b0, collected through the fib mappings
    new_atlas = old_atlas[mx,my,mz].reshape(volume_dimension,order="F")
    onim = nib.Nifti1Image(new_atlas,QSDR_AFFINE)
    onim.to_filename(output_v)
