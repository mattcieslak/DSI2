from tables import subject_space
from collections import defaultdict
import sys, os
TBI_DIR = os.getenv("TBI")
sys.path.append(TBI_DIR)
from dsi2.endpoint_difference_map import \
        MultiSubjectDataset, SubjectEndpointDataset
from dsi2.volumes.mask_dataset import MaskDataset
from dsi2.streamlines.track_dataset import TrackDataset
from dsi2.streamlines.track_math import tracks_to_regionpairs
from dsi2 import tbi, control, repro, repro2
group_mask = TBI_DIR + "/group_mask.nii"
import numpy as np

def subject_paths(subject_id,atlas,fatness):
    paths = {"subject":subject_id,"atlas":atlas,"fatness":fatness}
    paths["trk"] = TBI_DIR + "/raw/%s/%s_inwm_DSI_MNI.trk"%(subject_id,
                                                               subject_id)
    paths["endpoint_trk"] = TBI_DIR + "/raw/%s/%s.%s.ep.DSI_MNI.trk"%(
                                subject_id,atlas,fatness)
    paths["atlas_nii"]    = TBI_DIR + "/raw/%s/%s.%s.MNI.nii.gz"%(
                                subject_id,atlas,fatness)
    paths["endpoint_pkl"] = TBI_DIR + "/raw/%s/%s.%s.pkl"%(
                                subject_id,atlas,fatness)
    return paths

def query_voxel_subset_for_subject(subject_id, voxels):
    q = {'subject': subject_id, 'voxel': {'$in': voxels}}
    results = subject_space.find(q, {'endpoints': 1})

    density_map = defaultdict(int)
    for result in results:
        for endpoint in result['endpoints']:
            density_map[tuple(endpoint)] += 1

    return density_map

def query_voxel_subset(voxels):
    q = {'voxel': {'$in': voxels}}
    results = subject_space.find(q, {'subject': 1, 'endpoints': 1})

    def defaultgen():
        return defaultdict(int)

    subject_map = defaultdict(defaultgen)
    for result in results:
        subject_id = result['subject']
        for endpoint in result['endpoints']:
            subject_map[subject_id][tuple(endpoint)] += 1

    return subject_map

def sphere_space(radius):
    """
    NB: radius is in VOXELS, not mm
    """
    # Determine what the valid distances from a sphere center are
    # given the radius
    rg = np.arange(-radius,radius+1)
    indices = []
    for i in rg:
        for j in rg:
            for k in rg:
                if np.sqrt(i**2+j**2+k**2) <= radius and \
                  (i,j,k) not in indices:
                    indices.append((i,j,k))
    return np.array(indices)

def sphere_around_ijk( ijk, radius):
    _ijk = np.array(ijk)
    return [ coord.astype(int).tolist() for
            coord in sphere_space(radius) + _ijk ]

if __name__ == "__main__":
    import time
    t0 = time.time()
    for subj in repro2:
        paths = subject_paths(subj,"APARC","fat02")
        trk_ds = TrackDataset(paths["endpoint_pkl"])
        subject_id = '.'.join([subj,"APARC","fat02"])
        create_subject_space_from_trackset(trk_ds, subject_id, verbose=False)
    t1 = time.time()
    print 'load time:', t1-t0
    #import time
    t0 = time.time()
    query_voxel_subset(sphere_around_ijk([63, 48, 32], 4))
    t1 = time.time()
    print 'query time', t1-t0
