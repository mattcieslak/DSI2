#!/usr/bin/env python
import os, re, sys
import numpy as np
import nibabel as nib
import subprocess
import warnings
from dsi2.config import dsi2_data_path
from skimage.measure import marching_cubes, mesh_surface_area

class MaskDataset(object):
    def __init__(self, volume, region_int_labels=[], region_names=[]):
        """ Holds a 3d Mask image and offers operations on its grid.
        Parameters
        ----------
        fpath:str
          Path to Nifti-1 image (or any other format supported by Nibabel)
        label_table:str (path)
          Path to a text file of the format

            int ROIname\\n
            int ROIname\\n
            ...

        And a lookup tables will be created for ROIname<->id<->voxel_ijk's
        """
        if isinstance(volume, basestring):
            self.dset = nib.load(volume)
        else:
            self.dset = volume
        self.voxel_size = self.dset.get_header().get_zooms()
        self.data = self.dset.get_data()
        self.roi_ids = []
        self.roi_names = []
        self.roi_ijk = {}
        
        # Configure the region ints and names
        if not len(region_int_labels):
            for _id in np.unique(self.data[self.data>0]).astype(int):
                self.roi_ids.append(_id)
            if not len(region_names):
                warnings.warn("No usable region names provided - making them up")
                self.roi_names.append("region%i"%_id)
            else:
                self.roi_names=region_names
        else:
            if not len(region_names) == len(region_int_labels):
                raise ValueError("region_names does not match region_int_labels")
            self.roi_ids = region_int_labels
            self.roi_names = region_names
        for _id in self.roi_ids:        
            self.roi_ijk[_id] = np.array(np.nonzero(self.data==_id)).T

    def empty_copy(self):
        """Returns an empty (all zero) NIfTI-1 file of the same shape and affine
        """
        hdr = self.dset.get_header()
        img = np.zeros(self.data.shape)
        return nib.Nifti1Image(img,self.dset.get_affine(),header=hdr)

    def get_roi_ijk(self,roi_id):
        if type(roi_id) == str:
            return self.roi_ijk[self.roi_id(roi_id)]
        return self.roi_ijk[roi_id]

    def roi_name(self, roi_id):
        """returns the string name of an integer roi id"""
        roi_id = self.roi_names[self.roi_ids.index(roi_id)]
        return roi_id

    def roi_id(self, roi_name):
        """returns the integer id of a string roi_name"""
        roi_id = self.roi_ids[self.roi_names.index(roi_name)]
        return roi_id

    def region_centers(self):
        """Returns an n_rois x 3 ndarray of ROi centers"""
        centers = []        
        for roi in self.roi_ids:
            voxels = self.get_roi_ijk(roi)
            if voxels.shape == (0,):
                centers.append(-np.ones(3))
            else:
                centers.append(voxels.mean(axis=0))
        return np.array(centers)
    
    def region_volume(self):
        """Returns an n_rois x 3 ndarray of ROi centers"""
        voxel_volume = np.prod(self.voxel_size)
        return np.array(
           [self.get_roi_ijk(roi).shape[0] * voxel_volume for roi in self.roi_ids])
    
    def compute_surface_area(self):
        surface_area = []
        for roi_id in self.roi_ids:
            msk = (self.data == roi_id).astype(np.int)
            if msk.sum() == 0:
                surface_area.append(0)
                continue
            verts, faces = marching_cubes(msk,spacing=self.voxel_size)
            surface_area.append(
                mesh_surface_area(verts,faces))
        return np.array(surface_area)
            
    def get_stats(self):
        surface_area = self.compute_surface_area()
        region_centers = self.region_centers()
        region_volume = region_volume()
        return {
            "surface_area_mm2":surface_area, 
            "region_centers_voxel_coords":region_centers, 
             "region_volume_mm3":region_volume
            }
            

def get_MNI_wm_mask():
    return MaskDataset(os.path.join(dsi2_data_path, "MNI_BRAIN_MASK_FLOAT.nii.gz"))
