#!/usr/bin/env python
import time, sys, os
from dsi2.database.mongo_track_datasource import MongoTrackDataSource
from dsi2.streamlines.track_math import sphere_around_ijk
from dsi2.aggregation.region_labeled_clusters import RegionLabelAggregator
from dsi2.volumes.mask_dataset import MaskDataset
from traits.api import Int
import numpy as np
import pprocess
from scipy.stats import ttest_ind, mannwhitneyu
import pymongo
from bson.binary import Binary
connection = pymongo.MongoClient()
db = connection.dsi2

# Here are all the subjects that got successfully loaded into mongo
# If you want to use just a subset, slice the list
all_subjects = [ "1005y", "1410c", "2843B", "2843C", "2843A", 
                 "0002a", "4078B", "4078C", "4078A", "3735r", 
                 "1505l", "4673A", "3309A", "3309B", "3309C", 
                 "0305p", "2986m", "3075b", "3213t", "1656q", 
                 "1130w", "0009z", "1656A", "1656C", "1656B", 
                 "3906c", "0701j", "0377C", "1291r", "1935x", 
                 "3057t", "0004a", "2.1369B", "2.1369C", "3164q", 
                 "2.1369A", "4827r", "1169j", "0674k", "1694e", 
                 "4893d", "0682c", "1188q", "2121t", "0003a", 
                 "4458w", "0815z", "3503p", "4108i", "2.3527B", 
                 "0819v", "0105A", "0105C", "0105B", "3596a", 
                 "1319p", "3808w", "2895z", "0731f", "2.1394B", 
                 "2.1394C", "2.1394A", "0714p", "2888g", "3708s", 
                 "2.3527A", "2.3527C", "1347n", "1220A", "1220B", 
                 "1220C", "0282m", "0040u", "2066w", "1037B", 
                 "1037C", "1037A", "1943p", "3640i", "0377B", 
                 "3997p", "3987z", "0377A", "2268c", "2318e", 
                 "2664w", "0437n", "1415x", "3444h" ]

# Read the MNI brain mask and find where it is nonzero
wm_mask = MaskDataset(
    os.path.join(os.getenv("DSI2_DATA"),"MNI_BRAIN_MASK_FLOAT.nii.gz")
    )
# These will serve as the sphere centers
wm_ijk = wm_mask.in_mask_voxel_ijk

def make_aggregator(data_source,atlas_name="Lausanne2008",atlas_scale=60):
    # Create an empty aggregator and set the atlas to Lausanne2008, scale 60
    # TODO: Make this much easier!!!
    cl = RegionLabelAggregator()
    cl.set_track_source(data_source)
    cl.atlas_name = atlas_name
    cl.add_trait(atlas_name+"_scale",Int)
    cl.Lausanne2008_scale = atlas_scale
    cl.update_atlas()

    return cl

def evaluate_clustering_over_centers(centers, data_source, aggregator, radius):
    """
    Query each coordinate in ``centers`` as the center of a sphere with 
    radius ``radius``. Then send that subset of tracks to ``aggregator``
    for some aggregation operation.
    """
    results = []
    for cnum, center_index in enumerate(centers):
        if cnum%1000 == 0:
            print float(cnum)/len(centers)*100, "%"
        aggregator.set_track_sets(
            data_source.query_ijk(
                sphere_around_ijk(radius, center_index)
                )
        )
        aggregator.update_clusters()
        
        # once the aggregator has done some operation on the tracks
        # something gets pulled out and appended to results
        conn_ids, cvec_mat =  aggregator.connection_vector_matrix()
        # store the shape of the connection vector matrix
        results.append(cvec_mat.shape)
    return results



data_source = MongoTrackDataSource(
        scan_ids=all_subjects,
        db_name="dsi2",
        client=connection
        )


# Test a single result
radius = 2
agg = make_aggregator(data_source)
results = evaluate_clustering_over_centers([(33,54,45)],data_source,agg,2)

# In a typical ltpa, you will search every voxel in white matter (wm_ijk)
#N_EXAMPLES = wm_ijk.shape[0] # Use all the centers as query coordinates
#N_EXAMPLES = 900   # Only make this many queries
#for radius in [2,3]:
#    aggregator = make_aggregator(data_source)
#    results =  evaluate_clustering_over_centers(
#                wm_ijk[:N_EXAMPLES], data_source, aggregator, radius )
