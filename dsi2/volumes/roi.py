from traits.api import \
     HasTraits, Str, Int, List, Button, File, Instance, Dict,Enum, on_trait_change
from traitsui.api import Group, View, Handler, Item, \
                                    OKButton, CancelButton, EnumEditor
from .mask_dataset import MaskDataset
from ..streamlines.track_math import region_pair_dict_from_roi_list
import networkx as nx
import numpy as np
import os
from scipy.ndimage.morphology import binary_dilation

class ROI(HasTraits):
    filepath = File
    include_labels = List(Int)
    nifti_mask = Instance(MaskDataset)
    dilation = Int(0)

    def _include_labels_default(self):
        return [1]

    def _filepath_changed(self):
        self.nifti_mask = MaskDataset(self.filepath)

    def query_indices(self):
        """
        produces a list of tuple coordinates inside the regions
        If a dilation is requested, the dilated coordinates are returned
        """
        if self.dilation == 0:
            return self.nifti_mask.get_roi_ijk(self.include_labels[0])
        # Dilation!
        out = np.zeros_like(self.nifti_mask.data)
        for label in self.include_labels:
            out += self.nifti_mask == label
        dilated = binary_dilation(out,iterations=self.dilation)
        return map(tuple,np.array(np.nonzero(dilated)).T)
    
    def get_savename(self):
        """returns a name that is uesful for saving the results of this ROI 
        query
        """
        labelstr = ",".join([ "%i" % reg for reg in self.include_labels])
        volstr = os.path.split(self.filepath)[-1].rstrip(".nii.gz")
        return volstr + "_region_" + labelstr 
        
    traits_view = View(
        Item("filepath"), Item("include_labels"),Item("dilation"),
        kind="modal", buttons=[OKButton,CancelButton]
        )
