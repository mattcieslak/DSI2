#!/usr/bin/env python
import os, re, sys
import numpy as np
import nibabel as nib
import subprocess
#from mvpa.datasets.nifti import NiftiDataset

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

    def dump_nifti(self,fpath,all_one=False):
        """Save the masks included in this object to a nifti file at fpath"""
        new_nim = self.empty_copy()
        data = new_nim.get_data()
        for val in self.roi_ids:
            if not all_one:
                data[self.data==val] = val
            else:
                data[self.data==val] = 1.0
        new_nim.to_filename(fpath)

    def pairwise_dist(self):
        """ Returns the pairwise distance between all regions defined"""
        from dsi2.streamlines.track_math import triu_indices_from
        n_rois = len(self.roi_names)
        dists = np.zeros((n_rois,n_rois))
        centers = [np.array(self.get_roi_ijk(roi)).mean(axis=0) for roi in self.roi_ids]
        for r1, r2 in triu_indices_from(dists):
            dists[r1,r2] = np.sqrt(np.sum(centers[r1]-centers[r2])**2)
        return dists

    def region_centers(self):
        """Returns an n_rois x 3 ndarray of ROi centers"""
        return np.row_stack(
           [np.array(self.get_roi_ijk(roi)).mean(axis=0) for roi in self.roi_ids])

    def integerize_voxels(self,savepath=""):
        """When mapping from a rotated sphere to the standard sphere, it is
        the case that a number of decimal values get put into these voxels.
        This function tries to round the decimal-containing unique labels.
        optionally, save the integerized dataset to savepath"""
        pass

class WhiteMatterMask(MaskDataset):
    def __init__(self, fpath):
        """ Holds a 3d Mask image from aparc.a2009s.aseg's white matter

        Parameters
        ----------
        fpath:str
          Path to Nifti-1 image (or any other format supported by Nibabel)

        """
        # Load actual data
        self.dset = nib.load(fpath)
        self.data = self.dset.get_data()
        self.in_mask_voxel_ijk = np.array(np.nonzero(self.data)).T
        # Get a neighborhood mapper
        #nds = NiftiDataset(fpath,mask=fpath,labels=[1])
        #self.get_neighbors = nds.mapper.getNeighbors

        # Load the roi_id table
        self.roi_ids = []
        self.roi_names = []
        self.roi_ijk = {}
        ulabels = np.unique(self.data).astype(int)
        for label in ulabels[ulabels>0]:
            prc_patchname = subprocess.Popen(
                              ['@FS_roi_label','-lab','%i'%label],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            patchname = prc_patchname.communicate()[0]
            if not patchname:
                print "patch label %i is not an official patch"%label
                continue
            mtc = re.match(
                    '.*(White-Matter|Caudate|Putamen|Brain-Stem|Hippocampus|CC|Chiasm|Pallidum|DC).*',patchname)
            if not mtc:
                #reason = mtc.groups()[0]
                #print "ignoring label %s because it's not White-Matter %s"%(patchname,reason)
                continue

            label,pname = patchname.split()
            label=int(label)

            self.roi_ids.append(label)
            self.roi_names.append(pname.strip())
            self.roi_ijk[label] = map(tuple,np.array(np.nonzero(self.data==label),dtype=int).T)

class DestrieuxMask(MaskDataset):
    def __init__(self, fpath,secondary_msk=None):
        """ Holds a 3d Mask image and offers operations on its grid.
        Specifically made to hold aparc.a2009s.aseg

        Parameters
        ----------
        fpath:str
          Path to Nifti-1 image (or any other format supported by Nibabel)

        """
        # Load actual data
        self.dset = nib.load(fpath)
        self.data = self.dset.get_data()
        self.in_mask_voxel_ijk = np.array(np.nonzero(self.data)).T
        # Get a neighborhood mapper
        from mvpa.datasets.nifti import NiftiDataset
        nds = NiftiDataset(fpath,mask=fpath,labels=[1])
        self.get_neighbors = nds.mapper.getNeighbors

        # Load the roi_id table
        self.roi_ids = []
        self.roi_names = []
        self.roi_ijk = {}
        ulabels = np.unique(self.data).astype(int)
        for label in ulabels[ulabels>0]:
            prc_patchname = subprocess.Popen(
                              ['@FS_roi_label','-lab','%i'%label],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            patchname = prc_patchname.communicate()[0]
            if not patchname:
                print "patch label %i is not an official patch"%label
                continue
            mtc = re.match(
                    '.*(CSF|Lat-Vent|hypointensities|[Uu]nknown|abnormality|[Vv]entricle|vessel|choroid|CC|non-WM|White-Matter).*',patchname)
            if mtc:
                reason = mtc.groups()[0]
                print "ignoring label %s because it matched %s"%(patchname,reason)
                continue

            label,pname = patchname.split()
            label=int(label)

            self.roi_ids.append(label)
            self.roi_names.append(pname.strip())
            self.roi_ijk[label] = map(tuple,np.array(np.nonzero(self.data==label),dtype=int).T)

    def circos_karyotype_string(self):
        """returns a string that satisfies the circos karyotype file format
        """

        chrs = {'rh':[],'lh':[],'sc':[]} # Append (regname,nvox)
        for reg in self.roi_names:
            regvox = len(self.get_roi_ijk(reg))
            if reg.startswith("ctx_rh"):
                chrs['rh'].append((reg,regvox))
            elif reg.startswith("ctx_lh"):
                chrs['lh'].append((reg,regvox))
            else:
                chrs['sc'].append((reg,regvox))

        # Do it this way because we need the total
        rh_voxels = np.array([0] + [b for a,b in chrs['rh']])
        rh_cumsum = np.cumsum(rh_voxels)
        lh_voxels = np.array([0] + [b for a,b in chrs['lh']])
        lh_cumsum = np.cumsum(lh_voxels)
        sc_voxels = np.array([0] + [b for a,b in chrs['sc']])
        sc_cumsum = np.cumsum(sc_voxels)

        kary = """
        chr - rh rh 0 %i red
        chr - lh lh 0 %i blue
        chr - sc sc 0 %i green
        """%(rh_voxels.sum(),lh_voxels.sum(),sc_voxels.sum())

        # Collect the bands
        bands = []
        for regnum,(reg,regvox) in enumerate(chrs['rh']):
            color   = 'gneg'
            bandstr = "band rh %s %s %i %i %s"%(
              reg, reg, rh_cumsum[regnum], rh_cumsum[regnum+1],
              color)
            bands.append(bandstr)
        for regnum,(reg,regvox) in enumerate(chrs['lh']):
            color   = 'gneg'
            bandstr = "band lh %s %s %i %i %s"%(
              reg, reg, lh_cumsum[regnum], lh_cumsum[regnum+1],
              color)
            bands.append(bandstr)
        for regnum,(reg,regvox) in enumerate(chrs['sc']):
            color   = 'gneg'
            bandstr = \
             "band sc %s %s %i %i %s"%(
               reg, reg,
               sc_cumsum[regnum], sc_cumsum[regnum+1],
               color)
            bands.append(bandstr)

        return "\n".join([kary] + bands)
