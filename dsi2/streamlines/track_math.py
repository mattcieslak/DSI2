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

def unique_int2D(a):
    """ Returns unique rows of a matrix
    Ref
    ---
    http://www.mail-archive.com/numpy-discussion@scipy.org/msg04265.html
    """

    b = np.unique(a.view([('',a.dtype)]*a.shape[1])).view(a.dtype).reshape(-1,a.shape[1])
    return b

def remove_duplicates(a):
    is_uniq = np.array([True]*a.shape[0])
    is_uniq[1:] = np.abs(a[:-1] - a[1:]).sum(1) > 0
    return a[is_uniq]

def interpolate_track(trk,interpolate="linear",mm=1.):
    #(x1-x2)^2+(y1-y2)^2+(z1-z2)^2
    points_dist = euclidean_len(trk,total_distance=False)
    regrid_trk = [] # collect points here
    for p1,p2,dist in zip(trk[:-1],trk[1:],points_dist):
        if all(dist < mm/4.):
            #This length is smaller than a voxel
            regrid_trk.append(p1)
            continue
        # how many points to interpolate along this line?
        npoints = dist/(mm/4.)
        tpoints = max(npoints.astype(int))
        newpoints = np.column_stack([
                      np.linspace(p1[0],p2[0],endpoint=False,num=tpoints),
                      np.linspace(p1[1],p2[1],endpoint=False,num=tpoints),
                      np.linspace(p1[2],p2[2],endpoint=False,num=tpoints)
                    ])
        regrid_trk.append(newpoints)
    return np.row_stack(regrid_trk)


"""
Functions that involve both a track dataset AND a mask dataset
"""

def tracks_as_nii(msk_dset,trk_dset,interpolate="linear"):
    """Writes tracks into a nifti file. Use to verify space
    Parameters
    ----------
    tracklist:TrackDataset

    """
    newimg = msk_dset.empty_copy()
    newdata = newimg.get_data()
    for ijk in trk_dset.get_ijk_tracks(interpolate=interpolate):
        newdata[ijk[:,0],ijk[:,1],ijk[:,2]] += 1
    newimg.to_filename(outpath)

def roi_filter_tracks(msk_dset,trk_dset,roi_id,save_trk=""):
    """
    """
    roi_ijk = msk_dset.get_roi_ijk(roi_id)
    print "%i voxels found in the mask"%len(roi_ijk)
    roi_fibers = trk_dset.get_tracks_by_ijks(roi_ijk)
    print "%i fibers found in ROI"%len(list(roi_fibers))

    if save_trk:
        trk_dset.set_tracks(roi_fibers)
        trk_dset.save(save_trk)

    return roi_fibers

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

def tracks_to_regionpairs(msk_dset, trk_dset,savemat=False,savetrk=False,
        gray_regions=(),white_mask=False,n_endpoints=3):
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
    white_mask:dsi2.mask_dataset.MaskDataset
      a nii volume with nonzero voxels in white matter
    gray_regions:list or np.ndarray
      Region ids in `msk_dset` that represent gray matter
    savetrk:str
      Path to save the .trk of fibers that survive ttc
    n_endpoints:int
      The number of points at the ends of the fiberss to consider
      when trying to label the endpoints (ie if the last 2 points
      in a fiber are outside the mask, but the 3rd is intersecting,
      the fiber will get labeled if n_endpoints > 2)

    Returns:
    ========
    newtrks:dsi2.track_dataset.TrackDataset
      .trk file representing all fibers included after filtering
      and has a special attribute, endpoints:np.ndarray which is
      the start and end regions of all tracks that survive. col 0
      is the lower-labeled region.

    NB: The indices of endpoints and newtrks.tracks will represent
        equivalent fibers.

    """
    endpoints = [] # Store (lower label, higher label)
    tracks = []    # Store fiber id
    reg_crossings = np.zeros(trk_dset.get_ntracks(),) # How many regions does each fiber cross

    for trknum, trk in enumerate(trk_dset.get_ijk_tracks(interpolate="linear")):
        # Convert all points to their label vals
        labels = np.array([msk_dset.data[tuple(pt)] for pt in trk])
        reg_crossings[trknum] = (np.unique(labels) > 0).sum()

        # must intersect white matter (if wm mask provided
        if white_mask:
            if white_mask.data[trk].sum() == 0: continue

        # Must terminate in gray regions
        startpoint  = 0
        endpoint    = 0
        for ep in range(min(len(trk),n_endpoints)):
            if startpoint == 0:
                startpoint = labels[ep]
            if endpoint == 0:
                endpoint = labels[-(ep+1)]

        stoppers = sorted([startpoint,endpoint])
        # regions must be unique
        if (stoppers[0] == stoppers[1]) or (0 in stoppers): continue
        stoppers.sort()

        # Fiber has met all the requirements
        endpoints.append(stoppers)
        tracks.append(trknum)

    newtrks = trk_dset.subset(set(tracks))
    if savetrk: newtrks.save(savetrk)
    # Add the special attribure
    newtrks.endpoints = np.array(endpoints)
    newtrks.crossings = reg_crossings
    newtrks.centers   = msk_dset.region_centers()
    newtrks.roi_ids   = msk_dset.roi_ids
    return newtrks

def connectivity(msk_dset, trk_dset,percent_conversion=False,
                 savemat=False,symmetric=True,roistats=False,
                 region_ints = None):
    """compute a connectivity matrix based on the regions defined
    in this mask
    """
    voxtracks = []
    region_fibers = {}

    # If an explicit list of region ints is provided, use them
    # otherwise use the labels found in the actual atlas
    if region_ints is None:
        region_ints = msk_dset.roi_ids
        
    for roi_id in region_ints:
        roi_ijks = msk_dset.roi_ijk.get(roi_id,[])
        roi_fibers = trk_dset.get_tracks_by_ijks(roi_ijks)
        region_fibers[roi_id] = roi_fibers
        voxtracks.append([len(roi_ijks),len(roi_fibers)])
    voxtracks = np.array(voxtracks)

    # Empty connectivity matrix
    conn = np.zeros((len(region_ints),len(region_ints)),dtype=int)
    # Empty fiber length matrix
    #flen = np.zeros((len(region_ints),len(region_ints)),dtype=int)
    voxel_counts = np.zeros((len(region_ints),len(region_ints)),dtype=int)
    # iterate over every combination of regions, store the number of
    # common tracks
    upper_tri = np.array(triu_indices_from(conn))
    for i,j in upper_tri:
        roi_id_i,roi_id_j = region_ints[i], region_ints[j]
        connections = region_fibers[roi_id_i] & \
                                     region_fibers[roi_id_j]

        if not len(connections):
            n_connections = 0
            #avglen = 0
        else:
            #avglen = trk_dset.track_lengths[
            #  np.array(list(connections)).astype(int)].mean()
            n_connections = len(connections)

        conn[i,j] = n_connections
        voxel_counts[i,j] = len(msk_dset.roi_ijk.get(roi_id_i,[])) + \
                            len(msk_dset.roi_ijk.get(roi_id_j,[]))
        #flen[i,j] = avglen

    if symmetric:
        conn = symmetricize(conn)
        #flen = symmetricize(flen)
        voxel_counts = symmetricize(voxel_counts)

    outdata = {"network":conn.astype("<f8"),
            #"fiber_length":flen.astype("<f8"),
               "voxel_count":voxel_counts.astype("<f8"),
               "regions":np.array(region_ints).astype("<f8"),
               "centers":msk_dset.region_centers().astype("<f8"),
               "voxtracks":voxtracks.astype("<f8")
               }

    if savemat:
        sm(savemat,outdata)

    return conn

def connectivity_filter(msk_dset, trk_dset,percent_conversion=False,
                 savemat=False,symmetric=True,roistats=False,
                 incl_trk=False,excl_trk=False):
    """compute a connectivity matrix based on the regions defined
    in this mask
    """
    voxtracks = []
    region_fibers = {}
    for roi_id in msk_dset.roi_ids:
        roi_ijks = msk_dset.roi_ijk[roi_id]
        roi_fibers = trk_dset.get_tracks_by_ijks(roi_ijks)
        region_fibers[roi_id] = roi_fibers
        voxtracks.append([len(roi_ijks),len(roi_fibers)])
    voxtracks = np.array(voxtracks)

    # Empty connectivity matrix
    conn = np.zeros((len(msk_dset.roi_ids),len(msk_dset.roi_ids)),dtype=int)
    # Empty fiber length matrix
    flen = np.zeros((len(msk_dset.roi_ids),len(msk_dset.roi_ids)),dtype=int)
    # iterate over every combination of regions, store the number of
    # common tracks
    upper_tri = np.array(triu_indices_from(conn))
    incl_tracks = set()

    for i,j in upper_tri:
        if i == j: continue
        roi_id_i,roi_id_j = msk_dset.roi_ids[i], msk_dset.roi_ids[j]
        try:
            connections = region_fibers[roi_id_i] & \
                                         region_fibers[roi_id_j]
        except KeyError:
            n_connections = 0
            avglen = 0
            connections = []
        else:
            avglen = trk_dset.track_lengths[np.array(list(connections)).astype(int)].mean()
            n_connections = len(connections)
            # update the connecting
            incl_tracks.update(set(connections))

        conn[i,j] = n_connections
        flen[i,j] = avglen

    if symmetric:
        conn = symmetricize(conn)
        flen = symmetricize(flen)

    conn[np.diag_indices_from(conn)] = np.sum(conn,axis=1)
    for nrow,row in enumerate(flen):
        if row.sum() == 0:
            flen[nrow,nrow] = 0
        else:
            flen[nrow,nrow] = np.mean(row[row>0])

    outdata = {"network":conn.astype("<f8"),
               "fiber_length":flen.astype("<f8"),
               "regions":np.array(msk_dset.roi_ids).astype("<f8"),
               "centers":msk_dset.region_centers().astype("<f8"),
               "voxtracks":voxtracks.astype("<f8")
               }

    if savemat:
        sm(savemat,outdata)

    if incl_trk:
        # save the fibers that are included in the connectivity
        _incl = trk_dset.subset(incl_tracks)
        _incl.save(incl_trk)

    if excl_trk:
        # save the fibers that are included in the connectivity
        _incl = trk_dset.subset(incl_tracks,inverse=True)
        _incl.save(excl_trk)

    return conn

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
    if not fake_midpoints:
        if pairs:
            return np.vstack([[[line[0],line[-1]]] for line in lines])
        return np.vstack([[line[0],line[-1]] for line in lines])
    if pairs:
        return np.vstack([[[line[0],0,line[-1]]] for line in lines])
    return np.vstack([[line[0],0,line[-1]] for line in lines])
