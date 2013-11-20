#!/usr/bin/env python
import time, sys, os
sys.path.append("..")
from dsi2.clustering.region_clusters import RegionClusterizer
from dsi2.database.local_data import get_local_data
from dsi2.streamlines.track_math import sphere_around_ijk
from dsi2.volumes.mask_dataset import MaskDataset
import numpy as np
import pprocess
from scipy.stats import ttest_ind, mannwhitneyu

from dsi2.database.local_data import local_qsdrdb as local_trackdb
print "Loading datasets"

# How many times to run a streamline count shuffle within scan
NPERMUTATIONS = 100
data_source = get_local_data("example_data.json")

# Read the MNI brain mask and find where it is nonzero
wm_mask = MaskDataset(
    os.path.join(os.getenv("DSI2_DATA"),"MNI_BRAIN_MASK_FLOAT.nii.gz")
    )
wm_ijk = wm_mask.in_mask_voxel_ijk

def save_to_nifti(fname, results, indices=np.array([]) ):
    newimg = wm_mask.empty_copy()
    new_data = newimg.get_data()
    ix, jx, kx = wm_ijk.T
    if indices.size > 0:
        ix = ix[indices]
        jx = jx[indices]
        kx = kx[indices]
    new_data[ix,jx,kx] = results
    newimg.to_filename(fname)
    os.system("3drefit -newid -redo_bstat %s"%fname)


def within_between_analysis_percentage(arg):
    """Finds the mean within subject and between subject correlation means"""
    conn_ids,cvec = arg
    cvec = cvec.astype(np.float)
    correl = np.corrcoef(cvec)
    if correl.size == 0: return tuple([-2]*11)

    # How many connection pairs were present?
    n_connection_pairs = cvec.shape[1]
    # pull out correlation coefficients depending on whether they are from
    # the same or 2 different subjects
    within_values  = correl[within_indices]
    between_values = correl[between_indices]
    # summary values
    within_mean  = within_values.mean()
    within_perms = np.zeros(NPERMUTATIONS,)
    between_mean = between_values.mean()
    between_perms = np.zeros(NPERMUTATIONS,)

    for N in xrange(NPERMUTATIONS):
        # Shuffle the streamlines counts
        _ = map(np.random.shuffle, cvec)
        correl = np.corrcoef(cvec)
        # observing a value higher due to chance
        within_perms[N]   = correl[within_indices].mean()
        between_perms[N]  = correl[between_indices].mean()

    return  within_mean, between_mean, within_perms.mean(), between_perms.mean(), within_perms.std(), \
            between_perms.std(), \
            np.sum(within_perms < within_mean)/float(NPERMUTATIONS), \
            np.sum(between_perms < between_mean)/float(NPERMUTATIONS), \
            within_values.std(), between_values.std(), n_connection_pairs

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
N_EXAMPLES = wm_ijk.shape[0]
#N_EXAMPLES = 90000

# Make the sphere evaluation parallel
use_parallel = True
N_PROC = 4
p_results = pprocess.Map(limit=N_PROC)
compute = p_results.manage(
              pprocess.MakeParallel(evaluate_clustering_over_centers))
# Split the white matter voxels into N_PROC chunks
index_blocks = np.array_split(wm_ijk[:N_EXAMPLES],N_PROC)


for RADIUS in [2,3]:
    for ATLAS in ["scale33.thick2", "scale60.thick2"]: #, "scale125.thick2"]:
        #RADIUS=2
        #ATLAS="scale33.thick2"
        data_source.set_atlas(ATLAS)
        OUTPUT_PREFIX = os.getenv("DSI2PATH") + "/TPA/BIB_permutations/%s.slr%i"%(ATLAS,RADIUS)
        print "Running spheres, saving to",OUTPUT_PREFIX

        # Execute the search over each index block in parallel
        if use_parallel:
            for block in index_blocks:
                compute(block,ATLAS,RADIUS)
            all_results = []
            for r in p_results:
                all_results += r
            results = np.array(all_results)
            del all_results
        else:
            results = np.array(
                    evaluate_clustering_over_centers(
                        wm_ijk[:N_EXAMPLES], ATLAS, RADIUS)
                    )

        # Save results
        results[np.isnan(results)] = -2
        #print "Processing of", N_EXAMPLES, "took", t1-t0
        save_to_nifti(
                "%s.count.within.mean.qsdr.nii.gz"%OUTPUT_PREFIX,
                results[:,0], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.between.mean.qsdr.nii.gz"%OUTPUT_PREFIX,
                results[:,1], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.within.perm_mean.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
                results[:,2], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.between.perm_mean.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
                results[:,3], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.within.perm_std.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
                results[:,4], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.between.perm_std.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
                results[:,5], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.within.perm_percentile.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
                results[:,6], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.between.perm_percentile.qsdr.2pp.nii.gz"%OUTPUT_PREFIX,
                results[:,7], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.within.std.qsdr.nii.gz"%OUTPUT_PREFIX,
                results[:,8], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.between.std.qsdr.nii.gz"%OUTPUT_PREFIX,
                results[:,9], indices=np.arange(N_EXAMPLES))
        save_to_nifti(
                "%s.count.n_region_pairs.qsdr.nii.gz"%OUTPUT_PREFIX,
                results[:,10], indices=np.arange(N_EXAMPLES))
