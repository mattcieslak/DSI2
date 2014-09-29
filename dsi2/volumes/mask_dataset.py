#!/usr/bin/env python
import os, re, sys
import numpy as np
import nibabel as nib
import subprocess
from pkg_resources import Requirement, resource_filename

class MaskDataset(object):
    def __init__(self, fpath, label_table=None,secondary_msk=None):
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
        # Load actual data
        self.dset = nib.load(fpath)
        self.data = self.dset.get_data()
        self.in_mask_voxel_ijk = np.array(np.nonzero(self.data)).T
        self.roi_ids = []
        self.roi_names = []
        self.roi_ijk = {}

        if secondary_msk:
            try:
                nim = nib.load(secondary_mask)
            except:
                print "ERROR: unable to load",secondary_msk
                sys.exit(1)
            # apply mask to the atlas's data
            msk2 = nim.get_data()
            oldsum = np.sum(self.data)
            self.data = (msk2.get_data()>0)*self.data
            newsum = np.sum(self.data)
            if oldsum == newsum:
                print "WARNING: masking operation did not change the data"
            elif newsum == 0:
                print "FATAL: no voxels survive secondary masking"
                sys.exit(1)

        if label_table is not None:
            # Load the roi_id table
            fop = open(label_table,'r')
            for line in fop:
                try:
                    spl = line.split()
                    roi_id = int(spl[0])
                    roi_name = " ".join(spl[1:])
                except:
                    print "Unable to label regions, improper file format"
                    break
                self.roi_ids.append(roi_id)
                self.roi_names.append(roi_name)
                self.roi_ijk[roi_id] = map(tuple,np.array(np.nonzero(self.data==roi_id)).T)
        else:
            for _id in np.unique(self.data[self.data>0]).astype(int):
                self.roi_ids.append(_id)
                self.roi_names.append("region%i"%_id)
                self.roi_ijk[_id] = map(tuple,np.array(np.nonzero(self.data==_id)).T)

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
        return np.row_stack(
           [np.array(self.get_roi_ijk(roi)).mean(axis=0) for roi in self.roi_ids])

def get_MNI_wm_mask():
    return MaskDataset(resource_filename(
                   Requirement.parse("dsi2"),
                   "example_data/MNI_BRAIN_MASK_FLOAT.nii.gz")
            )
