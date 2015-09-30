#!/usr/bin/env python
import numpy as np
import warnings
from itertools import chain
from scipy.io.matlab import savemat as sm
import nibabel as nib
from dipy.tracking.streamline import transform_streamlines
from collections import defaultdict

def symmetricize(network):
    if not np.sum(np.tril(network,-1)):
        conn = network + network.T - np.diag(network.diagonal())
        return conn
    return network

triu_indices = lambda x, y=0: zip(*list(chain(*[[(i, j) for j in range(i + y, x)] for i in range(x - y)])))

def triu_indices_from(X):
    sz = X.shape[0]
    return zip(*triu_indices(sz))

def trackvis_header_from_info(
                    streamline_orientation, tracking_volume_shape,
                    tracking_volume_voxel_size):
    
    # Empty header to be filled
    hdr = nib.trackvis.empty_header(version=2)
    
    # Set up the zooms for the tracking volume
    try:
        hdr["voxel_size"] = tracking_volume_voxel_size
    except ValueError:
        raise ValueError("tracking_volume_voxel_size must be an iterable"
                         " of length 3. Got " + str(tracking_volume_voxel_size))
    hdr["vox_to_ras"][(0,1,2),(0,1,2)] = tracking_volume_voxel_size
    hdr["vox_to_ras"][-1,-1] = 1
    
    # Set up the zooms for the tracking volume
    try:
        hdr["dim"] = tracking_volume_shape
    except ValueError:
        raise ValueError("tracking_volume_shape must be an iterable"
                         " of length 3. Got " + str(tracking_volume_shape))
    
    # Set up the voxel order of the tracking volume
    slo = streamline_orientation.upper()
    def ori_error():
        raise ValueError("No trackvis header supplied, so attempted to use other" + \
                         "streamline_orientation must be [RL][AP][IS]. Got " + \
                         slo )
    if not slo[0] in ("R","L"): ori_error()
    if slo[0] == "L":
        hdr['vox_to_ras'][0,0] = -hdr['vox_to_ras'][0,0]
    if not slo[1] in ("A","P"): ori_error()
    if slo[1] == "P":
        hdr['vox_to_ras'][1,1] = -hdr['vox_to_ras'][1,1]
    if not slo[2] in ("I","S"): ori_error()
    if slo[2] == "I":
        hdr['vox_to_ras'][2,2] = -hdr['vox_to_ras'][2,2]
    hdr["voxel_order"] = slo
    
    return hdr
    


def streamlines_to_ijk(streams, target_volume=None, trackvis_header=None,
                    streamline_space="voxmm",  return_coordinates="voxel_index",
                    streamline_orientation="LPS", tracking_volume_shape=(),
                    tracking_volume_voxel_size=(), tracking_volume_affine=()
                    ):
    """
    The one place to convert streamlines to voxels.  Properties of the streamlines can 
    either be supplied by a nibabel trackvis header file via the ``trackvis_header`` 
    argument OR through the ``streamline_orientation``, ``tracking_volume_shape``,
    and ``tracking_volume_voxel_size``
    
    If you want the coordinates to be returned, set ``return_coordinates`` to True.
    Otherwise sequentially unique voxels will be returned.
    
    
    Parameters:
    -----------------
    streams:list or object array of (N,3) np.ndarrays
      Streamlines to be converted to voxel indices
    target_volume:nibabel.nifti1Image
      output voxel_space
    trackvis_header:recarray
      used to define properties of the streamline coordinates
    return_coordinates: "voxel_index", "voxel_coordinates", "both"
    """
    
    # Create a trackvis header if none is given
    if trackvis_header is None:
        warnings.warn("Approximating a trackvis header from vol info")
        trackvis_header = trackvis_header_from_info(streamline_orientation,
                                                    tracking_volume_shape, tracking_volume_voxel_size)
        
    # If no volume is given, use the info from the trackvis header    
    if target_volume is None:
        if not len(tracking_volume_affine) or not len(tracking_volume_voxel_size) \
           or not len(tracking_volume_shape):
            raise ValueError("Insufficient information to determine tracking volume")
        ref_affine = tracking_volume_affine
        img_voxel_size = tracking_volume_voxel_size
        ref_vol_shape = tracking_volume_shape
    else:
        ref_affine = target_volume.get_affine()
        img_voxel_size = np.array(target_volume.get_header().get_zooms())
        ref_vol_shape = np.array(target_volume.get_shape())
        
    trk_affine = trackvis_header['vox_to_ras']
    # Perform basic checks to make sure the tracks
    # and image are compatible
    if not np.all( ref_vol_shape == trackvis_header['dim']):
        raise ValueError("Shape mismatch between streamlines and volume")
    
    if not np.all(trackvis_header['voxel_size']==img_voxel_size):
        raise ValueError("Size mismatch between streamlines and volume voxels")
    
    # Check for valid affines
    if np.any(np.diag(ref_affine)==0):
        raise ValueError("invalid affine in target_vol")
    if np.any(np.diag(trk_affine) == 0):
        raise ValueError("invalid trackvis header, update dsi studio")
        
    # Do the volume orientations match?
    orientation_match = (np.sign(np.diag(ref_affine)) == \
            np.sign(np.diag(trk_affine)))[:3]
    flipx, flipy, flipz = np.logical_not(orientation_match)
    any_flip = any((flipx, flipy,flipz))
    
    # Convert the streamlines to voxel coordinates.
    # First convert the voxmm coordinates to the correct orientation
    extents = img_voxel_size * (trackvis_header['dim'] -1)
    ijk = []
    for _stream in streams:
        if any_flip:
            stream = _stream.copy()
            if flipx:
                stream[:,0] = extents[0] - stream[:,0]
            if flipy:
                stream[:,1] = extents[1] - stream[:,1]
            if flipz:
                stream[:,2] = extents[2] - stream[:,2]
        else:
            stream = _stream
        
        coords = stream / trackvis_header['voxel_size'] + 0.5
        ijk.append(coords)
    if return_coordinates == "voxel_coordinates":
        return np.array(ijk)#,dtype=np.object)
    voxel_idx = []
    for stream in ijk:
        voxel_idx.append(
            remove_sequential_duplicates(stream.astype(np.int)))
    if return_coordinates == "voxel_index":
        return np.array(voxel_idx)#, dtype=np.object)
    #return np.array(ijk),dtype=np.object), np.array(voxel_idx)
    return np.array(ijk), np.array(voxel_idx)

def streamline_voxel_lookup(voxelized_streamlines):
        # index tracks by the voxels they pass through
        tracks_at_ijk = defaultdict(set)
        for trknum, ijk in enumerate(voxelized_streamlines):
            data = set([trknum])
            for _ijk in ijk:
                tracks_at_ijk[tuple(_ijk)].update(data)
        return tracks_at_ijk
    
def streamlines_to_itk_world_coordinates(streams, target_volume,trackvis_header=None,
                    streamline_space="voxmm",
                    streamline_orientation="LPS", tracking_volume_shape=(),
                    tracking_volume_voxel_size=()
                    ):
    ijk_streamlines = streamlines_to_ijk(streams, target_volume,
                    trackvis_header=trackvis_header,
                    streamline_space=streamline_space,  return_coordinates="voxel_coordinates"
                    )
    ref_affine = target_volume.get_affine()
    ref_shape = target_volume.get_shape()
    # ITK uses LPS
    # Convert voxel indices to LPS
    flipx,flipy,flipz = np.sign(ref_affine[(0,1,2),(0,1,2)]) != np.array([-1,-1,1])
    LPS_affine = np.eye(4)
    if flipx:
        LPS_affine[0,0] = -1
        LPS_affine[0,-1] = ref_shape[0]
    if flipy:
        LPS_affine[1,1] = -1
        LPS_affine[1,-1] = ref_shape[1]
    if flipz:
        LPS_affine[2,2] = -1
        LPS_affine[2,-1] = ref_shape[2]
    final_affine = LPS_affine * ref_affine
    
    return transform_streamlines(ijk_streamlines,final_affine)

def voxels_to_streamlines(voxel_coordinates, nib_img=None, volume_affine=None,
                          volume_shape=None,volume_voxel_size=None,  voxmm_orientation="LPS"):
    """
    Transforms voxel indexes to corresponding streamline coordinates in
    voxmm.
    """
    if nib_img is None:
        if (volume_shape is None or volume_voxel_size is None or volume_affine is None):
            raise ValueError("Must provide a nibel image or volume description")
        ref_affine = volume_affine
        ref_shape = volume_shape
        ref_zooms = volume_voxel_size
    else:
        if not (volume_shape is None and volume_voxel_size is None \
                and volume_affine is None):
            warnings.warn("ignoring volume shape/size arguments and using image provided")
        ref_affine = nib_img.get_affine()
        ref_shape = nib_img.get_shape()
        ref_zooms = np.array(nib_img.get_header().get_zooms())
    if np.any(ref_affine[(0,1,2),(0,1,2)] == 0): raise ValueError("Invalid affine")
    
    # Flip voxels to desired trk orientation
    ijk_affine = np.eye(4)
    xvoxel = ref_affine[0,0]
    xtrk = voxmm_orientation[0]
    flipi = xvoxel < 0 and xtrk == "R" or xvoxel > 0 and xtrk == "L"
    yvoxel = ref_affine[1,1]
    ytrk = voxmm_orientation[1]
    flipj = yvoxel < 0 and ytrk == "A" or yvoxel > 0 and ytrk == "P"
    zvoxel = ref_affine[2,2]
    ztrk = voxmm_orientation[2]
    flipk = zvoxel < 0 and ztrk == "S" or zvoxel > 0 and ztrk == "I"
    if flipi:
        ijk_affine[0,0] = -1
        ijk_affine[0,-1] = ref_shape[0] - 1
    if flipj:
        ijk_affine[1,1] = -1
        ijk_affine[1,-1] = ref_shape[1] -1
    if flipk:
        ijk_affine[2,2] = -1
        ijk_affine[2,-1] = ref_shape[2] -1
    if any([flipi, flipj,flipk]):
        corrected_ijk = transform_streamlines(voxel_coordinates, ijk_affine)
    else:
        corrected_ijk = voxel_coordinates
        
    # apply the voxel size scaling to the voxel coordinates
    return [ref_zooms *  cijk   for cijk in corrected_ijk]

def remove_duplicates(a):
    is_uniq = np.array([True]*a.shape[0])
    is_uniq[1:] = np.abs(a[:-1] - a[1:]).sum(1) > 0
    return a[is_uniq]

def remove_sequential_duplicates(a):
    uniq = np.array([True] * a.shape[0])
    uniq[1:] = -np.all(a[:-1] == a[1:],axis=1)
    return a[uniq]

"""
Functions that involve both a track dataset AND a mask dataset
"""

def region_pair_dict_from_roi_list(roi_list):
    #WARNING: MADE THIS pairnum+1 so that it doesn't start at 0
    # This breaks backwards compatibility with previous versions
    # to use the old numbering set 
    import os
    if os.getenv("DSI2_CONN_START_AT_0"):
        inc = 0
    else:
        inc = 1
    roi_ids = np.array(roi_list)
    return dict(
      [((roi_ids[index1], roi_ids[index2]), pairnum + inc ) for \
         pairnum, (index1,index2) in enumerate(
          np.array(np.triu_indices(len(roi_ids))).T) ]
       )

def connection_ids_from_voxel_coordinates( voxel_streamlines, 
                atlas_voxels=None,  roi_pair_lookup=None, atlas_label_int_array=None, save_npy="", 
                correct_labels=None):

    # Configure data structures for labeling connections
    labeled = np.zeros((len(voxel_streamlines),))
    maskdata = atlas_voxels.astype(np.int)
    
    # Configure ROI pair mapper
    if roi_pair_lookup is None:
        if atlas_label_int_array is None:
            import warnings
            warnings.warn("No roi pair mapping or region list provided")
            atlas_label_int_array = np.unique(atlas_voxels[atlas_voxels > 0])
        roi_pair_lookup = region_pair_dict_from_roi_list(atlas_label_int_array)
        
    # Label the streamlines
    endpoints = np.zeros(len(voxel_streamlines),dtype=np.int)
    for trknum, trk in enumerate(voxel_streamlines):
        trk = trk.astype(np.int)
        # Convert all points to their label vals
        labels = atlas_voxels[trk[:,0], trk[:,1], trk[:,2]]
        # Must terminate in gray regions
        lbl = labels[labels > 0]

        # Are there any regions found in this track?
        if not len(lbl) > 1: continue
        startpoint, endpoint = lbl[0], lbl[-1]
        stoppers = sorted([startpoint,endpoint])
        roi_id = roi_pair_lookup[tuple(stoppers)]
        if not (correct_labels is None):
            assert correct_labels[trknum] == roi_id
        endpoints[trknum] = roi_id
    
    if save_npy: np.save(save_npy,endpoints)
    return endpoints

def connection_ids_from_tracks(msk_dset, trk_dset, 
                region_ints=None, save_npy="", 
                n_endpoints=3, scale_coords=np.array([1,1,1]),
                correct_labels=None,check_affines = False):
    """ Take a mask dataset and some tracks. For each track in the tracks
    make sure that the last point(s) is(are) in gray_regions and that at least
    one of its points is in `white_regions`. Make a new trk dataset with these
    tracks if `savetrk`.
    Parameters:
    ===========
    msk_dset:dsi2.mask_dataset.MaskDataset or list
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

    # Check that the mapping from streamline coordinates to 
    # voxels is is possible through division
    
    for trknum, trk in enumerate(trk_dset):
        # Convert all points to their label vals
        # Add 0.5 to the voxel index to conform to DICOM standard
        pt = np.floor(trk/scale_coords + 0.5).astype(np.int32)
        labels = maskdata[pt[:,0],pt[:,1],pt[:,2]]

        # Must terminate in gray regions
        lbl = np.flatnonzero(labels)

        # Are there any regions found in this track?
        if not len(lbl) > 1: continue
        startpoint, endpoint = labels[lbl[0]], labels[lbl[-1]]
        #if startpoint == endpoint: continue
        stoppers = sorted([startpoint,endpoint])
        
        roi_id = roi_pair_lookup[tuple(stoppers)]
        
        if not (correct_labels is None):
            assert correct_labels[trknum] == roi_id
        # Fiber has met all the requirements
        endpoints[trknum] = roi_id

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
