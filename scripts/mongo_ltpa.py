#!/usr/bin/env python
import time, sys, os
from dsi2.database.mongo_track_datasource import MongoTrackDataSource
from dsi2.streamlines.track_math import sphere_around_ijk
from dsi2.volumes.mask_dataset import MaskDataset
import numpy as np
import pprocess
from scipy.stats import ttest_ind, mannwhitneyu
import pymongo
from bson.binary import Binary
connection = pymongo.MongoClient()
db = connection.dsi2

print "Loading datasets"

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


# How many times to run a streamline count shuffle within scan
NPERMUTATIONS = 100
data_source = MongoTrackDataSource(
        scan_ids=all_subjects,
        db_name="dsi2",
        client=connection
        )


# Read the MNI brain mask and find where it is nonzero
wm_mask = MaskDataset(
    os.path.join(os.getenv("DSI2_DATA"),"MNI_BRAIN_MASK_FLOAT.nii.gz")
    )
wm_ijk = wm_mask.in_mask_voxel_ijk



test_scan_ids = ["0377A","2843A"]

def query_sphere(sphere_coords):
    sl_ids = db.connections.aggregate(
        # Find the coordinates for each subject
        [
            {"$match":{
                "scan_id":{"$in":test_scan_ids},
                "ijk":{"$in":
                       ["%d_%d_%d" % tuple(map(int,coord)) for coord in test_coordinates]}
                }},
            {"$project":{"scan_id":1,"sl_id":1}},
            {"$unwind":"$sl_id"},
            {"$group":{"_id":"$scan_id", "sl_ids":{"$addToSet":"$sl_id"}}}
        ]
    )["result"]


    track_datasets = []
    for subj in sl_ids:
        sl_data = db.streamlines.find(
            {
             "scan_id":sl_ids[0]["_id"],
             "sl_id":{"$in":sl_ids[0]["sl_ids"]}
            }
        )
        streamlines = [pickle.loads(d['data']) for d in sl_data]
        scans = get_local_data(os.path.join(test_output_data,"example_data.json"))
        toy_dataset = TrackDataset(header={"n_scalars":0}, streams = (),properties=scans[0])
        toy_dataset.tracks = np.array(streamlines,dtype=object)
        track_datasets.append(toy_dataset)
    return track_datasets

###### Create a clusterer
def evaluate_clustering_over_centers(centers,ATLAS,RADIUS):
    cl = RegionLabelAggregator()
    cl.set_atlas(ATLAS)
    results = []
    for cnum, center_index in enumerate(centers):
        if cnum%1000 == 0:
            print float(cnum)/len(centers)*100, "%"
        cl.set_track_sets(
            data_source.query_ijk(
                sphere_around_ijk(RADIUS, center_index)
                )
        )
        cl.update_clusters()
        results.append( within_between_analysis_percentage(
            cl.connection_vector_matrix()       )
        )
    return results


## Run the
#N_EXAMPLES = wm_ijk.shape[0]
#N_EXAMPLES = 90000

# Make the sphere evaluation parallel
#use_parallel = True
#N_PROC = 4
#p_results = pprocess.Map(limit=N_PROC)
#compute = p_results.manage(
        #              pprocess.MakeParallel(evaluate_clustering_over_centers))
# Split the white matter voxels into N_PROC chunks
#index_blocks = np.array_split(wm_ijk[:N_EXAMPLES],N_PROC)


#for RADIUS in [2,3]:
#    for ATLAS in ["scale33.thick2", "scale60.thick2"]: #, "scale125.thick2"]:
#        #RADIUS=2
#        #ATLAS="scale33.thick2"
#        data_source.set_atlas(ATLAS)
#        OUTPUT_PREFIX = os.getenv("DSI2PATH") + "/TPA/BIB_permutations/%s.slr%i"%(ATLAS,RADIUS)
#        print "Running spheres, saving to",OUTPUT_PREFIX
#
#        # Execute the search over each index block in parallel
#        if use_parallel:
#            for block in index_blocks:
#                compute(block,ATLAS,RADIUS)
#            all_results = []
#            for r in p_results:
#                all_results += r
#            results = np.array(all_results)
#            del all_results
#        else:
#            results = np.array(
#                    evaluate_clustering_over_centers(
#                        wm_ijk[:N_EXAMPLES], ATLAS, RADIUS)
#                    )
#
#        # Save results
#        results[np.isnan(results)] = -2
#        #print "Processing of", N_EXAMPLES, "took", t1-t0
#        save_to_nifti(
#                "%s.count.within.mean.qsdr.nii.gz"%OUTPUT_PREFIX,
#                results[:,0], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.between.mean.qsdr.nii.gz"%OUTPUT_PREFIX,
#                results[:,1], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.within.perm_mean.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
#                results[:,2], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.between.perm_mean.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
#                results[:,3], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.within.perm_std.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
#                results[:,4], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.between.perm_std.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
#                results[:,5], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.within.perm_percentile.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
#                results[:,6], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.between.perm_percentile.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
#                results[:,7], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.within.std.qsdr.nii.gz"%OUTPUT_PREFIX,
#                results[:,8], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.between.std.qsdr.nii.gz"%OUTPUT_PREFIX,
#                results[:,9], indices=np.arange(N_EXAMPLES))
#        save_to_nifti(
#                "%s.count.n_region_pairs.qsdr.nii.gz"%OUTPUT_PREFIX,
#                results[:,10], indices=np.arange(N_EXAMPLES))
