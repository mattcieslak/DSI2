from dsi2.ui.local_data_importer import LocalDataImporter
from dsi2.streamlines.track_math import region_pair_dict_from_roi_list
from dsi2.volumes import graphml_from_label_source
from pkg_resources import Requirement, resource_filename
from scipy.io.matlab import savemat
import networkx as nx
import nibabel as nib
import numpy as np
import os

N_PROCS=8

lausanne_scale_lookup = {
                  33:resource_filename(Requirement.parse("dsi2"),
                    "example_data/lausanne2008/resolution83/resolution83.graphml"),
                  60:resource_filename(Requirement.parse("dsi2"),
                    "example_data/lausanne2008/resolution150/resolution150.graphml"),
                  125:resource_filename(Requirement.parse("dsi2"),
                    "example_data/lausanne2008/resolution258/resolution258.graphml"),
                  250:resource_filename(Requirement.parse("dsi2"),
                    "example_data/lausanne2008/resolution500/resolution500.graphml"),
                  500:resource_filename(Requirement.parse("dsi2"),
                    "example_data/lausanne2008/resolution1015/resolution1015.graphml")
                  }

print "caching connection id lookup tables..."
lookup_cache = {}
for scale in [33, 60, 125, 250, 500]:
    atlas_graphml = lausanne_scale_lookup[scale]
    print "\t\t+ loading regions from", atlas_graphml
    graph = nx.read_graphml(atlas_graphml)
    lookup_cache[scale] = {'region_labels':{},
                    'region_pairs_to_index':{},
                    'index_to_region_pairs':{},
                    'region_pair_strings_to_index':{},
                    'regions':[]}
    graphml_data = lookup_cache[scale]
    
    for roi_id,roi_data in graph.nodes(data=True):
        graphml_data['region_labels'][roi_id] = roi_data
    graphml_data['regions'] = np.array(sorted(map( int,graph.nodes() )))
    graphml_data['region_pairs_to_index'] = region_pair_dict_from_roi_list(
                                                       graphml_data['regions'])
    # Which regionpairs map to which unique id in this dataset?
    graphml_data['index_to_region_pairs'] = dict(
        [ (idxnum,(id1, id2)) for (id1,id2), idxnum in graphml_data['region_pairs_to_index'].iteritems()
        ]
    )
    graphml_data['region_pair_strings_to_index'] = dict([
        (value,key) for key,value in graphml_data['index_to_region_pairs'].iteritems()
    ])
print "done."


def save_connectivity_matrices(scan):
    out_data = {}
    opath = scan.pkl_trk_path + ".mat"
    for lnum, label_source in enumerate(scan.track_label_items):
        # Load the mask
        # File containing the corresponding label vector
        npy_path = label_source.numpy_path
        if not os.path.exists(npy_path):
            raise ValueError( 
                "\t\t++ [%s]"%sid, npy_path, "does not exist")
        conn_ids = np.load(npy_path)
        
        # Make a prefix
        prefix = "_".join(
            ["%s_%s" % (k,v) for k,v in label_source.parameters.iteritems()]) + "_"

        # Get the region labels from the parcellation
        graphml = graphml_from_label_source(label_source)
        if graphml is None:
            raise ValueError("\t\t++ No graphml exists")
            
        graphml_data = lookup_cache[label_source.parameters['scale']] 
        regions = graphml_data['regions']
        
        # Empty connectivity matrix
        out_data[prefix+'streamline_count'] = np.zeros((len(regions),len(regions)))
        connectivity = out_data[prefix+'streamline_count']
        scalars = {}
        for scalar in scan.track_scalar_items:
            # Load the actual array
            scalars[prefix + scalar.name] = np.load(scalar.numpy_path)
            # Empty matrix for 
            out_data[prefix + scalar.name] = np.zeros((len(regions),len(regions)))
            out_data[prefix + scalar.name + "_sd"] = np.zeros((len(regions),len(regions)))
            
        # extract the streamline lengths and put them into an array
        out_data[prefix+'length'] = np.zeros((len(regions),len(regions)))
        out_data[prefix+'length_sd'] = np.zeros((len(regions),len(regions)))
        streams, hdr = nib.trackvis.read(scan.trk_file)
        lengths = np.array([len(arr) for arr in streams])
        
        # extract the streamline scalar index
        for conn, (_i,_j) in graphml_data["index_to_region_pairs"].iteritems():
            i,j = _i-1, _j-1
            indexes = conn_ids == conn
            sl_count = np.sum(indexes)
            connectivity[i,j] = connectivity[j,i] = sl_count
            
            out_data[prefix+'length'][i,j] = out_data[prefix+'length'][j,i] = lengths[indexes].mean()
            out_data[prefix+'length_sd'][i,j] = out_data[prefix+'length_sd'][j,i] = lengths[indexes].std()
            
            # Fill in the average scalar value
            for scalar_name, scalar_data in scalars.iteritems():
                scalar_vals = scalars[scalar_name][indexes]
                scalar_mean = scalars_vals.mean()
                out_data[scalar_name][i,j] = out_data[scalar_name][j,i] = scalar_mean
                scalar_std = scalars_vals.std()
                out_data[scalar_name+"_sd"][i,j] = out_data[scalar_name + "_sd"][j,i] = scalar_std
    
    print "saving", opath
    savemat(opath, out_data)

            
if __name__ == "__main__":
    json_file = "/home/matt/testing_data/testing_input/data_wscalars.json"
    from dsi2.database.local_data import get_local_data
    local_data = get_local_data(json_file)
    import multiprocessing
    pool = multiprocessing.Pool(processes=N_PROCS)
    result=pool.map(save_connectivity_matrices,local_data)    
    pool.close()
    pool.join()
    print "finished!"
    