#!/usr/bin/env python
import numpy as np
from itertools import chain
from scipy.io.matlab import savemat as sm

def symmetricize(network):
    if not np.sum(np.tril(network,-1)):
        conn = network + network.T - np.diag(network.diagonal())
        return conn
    return network

triu_indices = lambda x, y=0: zip(*list(chain(*[[(i, j) for j in range(i + y, x)] for i in range(x - y)])))

def triu_indices_from(X):
    sz = X.shape[0]
    return zip(*triu_indices(sz))

def euclidean_len(track,total_distance=True):
    """compute the length along all points of a track
    Parameters:
    -----------
    track:np.ndarray n,3
      3d coordinates of points on the track
    total_distance:bool
      Return the total distance along the track, or the
      distance between each point?

    """
    if track.dtype == object:
        #case of the single track
        track = track.astype("<f8")
    #sq_points_diff = (x1-x2)^2+(y1-y2)^2+(z1-z2)^2
    points_dist = np.sqrt(np.sum(np.diff(track,axis=0)**2,axis=1))
    if not total_distance:
        return points_dist
    return np.sum(points_dist)

def remove_duplicates(a):
    is_uniq = np.array([True]*a.shape[0])
    is_uniq[1:] = np.abs(a[:-1] - a[1:]).sum(1) > 0
    return a[is_uniq]

"""
Functions that involve both a track dataset AND a mask dataset
"""

def region_pair_dict_from_roi_list(roi_list):
    roi_ids = np.array(roi_list)
    return dict(
      [((roi_ids[index1], roi_ids[index2]),pairnum ) for \
         pairnum, (index1,index2) in enumerate(
          np.array(np.triu_indices(len(roi_ids))).T) ]
       )

def connection_ids_from_tracks(msk_dset, trk_dset, region_ints=None, save_npy="", n_endpoints=3,scale_coords=np.array([1,1,1])):
    """ Take a mask dataset and some tracks. For each track in the tracks
    make sure that the last point(s) is(are) in gray_regions and that at least
    one of its points is in `white_regions`. Make a new trk dataset with these
    tracks if `savetrk`.
    Parameters:
    ===========
    msk_dset:dsi2.mask_dataset.MaskDataset
      labeled voxels to use as the "atlas"
    trk_dset:dsi2.track_dataset.TrackDataset
      TrackDataset you want to use for the fibers
    region_ints:np.ndarray
      an array of integers for each region of interest. If not provided,
      only the integer labels observed in msk_dset will be considered.
      This is not ideal is a high resolution atlas may lose a region
      when mapped to another resolution. Use this argument!
    save_npy:str
      Path to save the .npy of connection ids
    scale_coords:
      Amount to scale streamline coordinates in x,y,z. TODO: Make this
      an affine matrix instead of scalars

    n_endpoints:int
      The number of points at the ends of the fiberss to consider
      when trying to label the endpoints (ie if the last 2 points
      in a fiber are outside the mask, but the 3rd is intersecting,
      the fiber will get labeled if n_endpoints > 2)

    Returns:
    ========
    connection_ids:np.ndarray
    """


    endpoints = np.zeros((trk_dset.get_ntracks(),))

    # Configure data structures for labeling connections
    if region_ints is None:
        region_ints = msk_dset.roi_ids
    roi_pair_lookup = region_pair_dict_from_roi_list(region_ints)
    maskdata = msk_dset.data.astype(np.int32)

    for trknum, trk in enumerate(trk_dset):
        # Convert all points to their label vals
        pt = np.floor(trk/scale_coords).astype(np.int32)
        labels = maskdata[pt[:,0],pt[:,1],pt[:,2]]

        # Must terminate in gray regions
        lbl = np.flatnonzero(labels)

        # Are there any regions found in this track?
        if not len(lbl) > 2: continue
        startpoint, endpoint = labels[lbl[0]], labels[lbl[-1]]
        #if startpoint == endpoint: continue
        stoppers = sorted([startpoint,endpoint])

        # Fiber has met all the requirements
        endpoints[trknum] = \
            roi_pair_lookup[tuple(stoppers)]

    if save_npy: np.save(save_npy,endpoints)
    return endpoints

def sphere_around_ijk(radius,ijk):
    """
    NB: radius is in VOXELS, not mm
    """
    # Determine what the valid distances from a sphere center are
    # given the radius
    assert type(radius) == int

    i, j, k = np.ogrid[(-radius):(radius+1),
                       (-radius):(radius+1),
                       (-radius):(radius+1)]

    ok_coords = (np.vstack(
        np.nonzero((i**2 + j**2 + k**2) <= radius**2)
        ).T - radius + np.array(ijk)).astype(np.uint32)

    return map(tuple,ok_coords)

def tracks_to_endpoints(lines,pairs=True, LPS_sort=False,
                             fake_midpoints=False):
    if not lines.size:
        print "no endpoints for empty array"
        return np.array([])
    if not fake_midpoints:
        if pairs:
            return np.vstack([[[line[0],line[-1]]] for line in lines])
        return np.vstack([[line[0],line[-1]] for line in lines])
    if pairs:
        return np.vstack([[[line[0],0,line[-1]]] for line in lines])
    return np.vstack([[line[0],0,line[-1]] for line in lines])
