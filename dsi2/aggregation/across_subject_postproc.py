#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt

from traits.api import HasTraits, Instance, Array, Enum, \
    Str, File, on_trait_change, Bool, Dict, Range, Color, List, Int, Property, Button, DelegatesTo #, on_trait_change, Str, Tuple
from traitsui.api import Group,Item, RangeEditor, EnumEditor, TableEditor, ObjectColumn, View
from traitsui.menu import OKButton, CancelButton, ApplyButton
from ..streamlines.track_math import region_pair_dict_from_roi_list
import networkx as nx
from ..database.traited_query import Scan

graphml_lookup = {
                  "scale33":"resolution83",     "scale60":"resolution150",
                  "scale125":"resolution258",   "scale250":"resolution500",
                  "scale500":"resolution1015"
                  }

label_table = TableEditor(
    columns = \
    [ ObjectColumn(name="scan_id",editable=False),
      ObjectColumn(name="scan_group",editable=False),
      ObjectColumn(name="label",editable=True),
    ],
    auto_size=True
    )

class AcrossSubjectPostproc(HasTraits):
    filter_operation = Enum("None","XOR","Minimum NonZero Count")
    filterer = Instance(HasTraits)
    # For defining groups of subjects to apply filters to
    split_factor = Enum("scan_group","subject_id","scan_gender")
    subjects = List(Instance(Scan))
    subject_labels = Array

    # lazily create these
    def _filter_obj_default(self):
        return NoneFilter()

    def _split_factor_changed(self):
        """ 
        assign a value to props.label for each of the 
        info objects based on that info's value for 
        ``self.split_factor``
        """
        labels = set()
        for info in self.subjects:
            labels.update([getattr(info, self.split_factor)])
        label_lut = dict([(name,code) for code,name in enumerate(labels)])
        print label_lut
        lbl = []
        for info in self.subjects:
            _lbl = label_lut[getattr(info,self.split_factor)]
            info.label = _lbl
            lbl.append(_lbl)
        self.subject_labels = np.array(lbl)

    def _filter_operation_changed(self):
        if self.filter_operation == "XOR":
            self.filterer = XORFilter()
            self.sync_trait("subject_labels",self.filterer,mutual=True)
        elif self.filter_operation == "None":
            self.filterer = NoneFilter()
        elif self.filter_operation == "Minimum NonZero Count":
            #TODO: Implement mnz filter
            self.filterer = MinimumNonZeroCountFilter()
    
    def filter_clusters(self, arg):
        return self.filterer.filter_clusters(arg)

    # widgets for editing algorithm parameters
    traits_view = View(Group(
                        Item("filter_operation"),
                        Item("filterer",style="custom"),
                        Item("split_factor"),
                        Item("subjects",editor=label_table),
                         show_border=True,
                        ),
                         resizable=True,
                         height=400,
                         kind='nonmodal',
                         buttons=[ApplyButton,OKButton,CancelButton]
                       )

class NoneFilter(HasTraits):
    def filter_clusters(self,arg):
        assert 0
    traits_view = Group()

class XORFilter(HasTraits):
    """
    Checks to see if an XOR pattern is present in any of the observed connections.
    Parameters:
    -----------
    arg:tuple
        termination pattern matrix, matrix column ids
    data_labels:np.ndarray
        -1 or 1 denoting the class membership for each row in the matrix
    class0_reqd:int
        How many samples labeled -1 must have non-zero streamline counts for
        a connection to be "True" for class 0?
    class1_reqd:int
        How many samples labeled 1 must have non-zero streamline counts for
        a connection to be "True" for class 1?
    """
    class0_reqd = Int
    class1_reqd = Int
    subject_labels = Array
    
    # UI
    traits_view = Group(
            Item("class0_reqd"),Item("class1_reqd"))

    def _subject_labels_changed(self):
        # Automatically set a strict XOR
        self.class0_reqd = (self.subject_labels <  1).sum()
        self.class1_reqd = (self.subject_labels == 1).sum()

    def filter_clusters(self, arg):
        conn_ids,cvec = arg
        if cvec.size == 0: return np.array([])

        # convert to boolean
        nz = cvec > 0
        # counts vs the threshold numbers for each class
        class0_nz = nz[self.subject_labels <  1,:].sum(0)
        class0_min = (self.subject_labels < 1).sum() - self.class0_reqd

        class1_nz = nz[self.subject_labels == 1,:].sum(0)
        class1_min = (self.subject_labels == 1).sum() - self.class1_reqd
        
        # Which columns fall in the middle-range where we can't be sure if it's absent ( < min )
        # or present ( > reqd )
        ambiguous = (class0_nz > class0_min ) & (class0_nz < self.class0_reqd) | (class1_nz > class1_min ) & (class1_nz < self.class1_reqd)
        xor = np.logical_xor( class0_nz >= self.class0_reqd, class1_nz >= self.class1_reqd ) 
        xor[ambiguous] = 0

        xor_labels = [ conn_ids[m] for m in np.flatnonzero( xor ) ]
        print "found", len(xor_labels), "XOR region pairs"
        return xor_labels



class MinimumNonZeroCountFilter(HasTraits):
    """
    Checks to see if an XOR pattern is present in any of the observed connections.
    Parameters:
    -----------
    arg:tuple
        termination pattern matrix, matrix column ids
    data_labels:np.ndarray
        -1 or 1 denoting the class membership for each row in the matrix
    class0_reqd:int
        How many samples labeled -1 must have non-zero streamline counts for
        a connection to be "True" for class 0?
    class1_reqd:int
        How many samples labeled 1 must have non-zero streamline counts for
        a connection to be "True" for class 1?
    """
    min_nonzero_required = Int
    plot_region_counts = Button(label="Plot Conuts")
    
    # UI
    traits_view = Group(
            Item("min_nonzero_required"),
            Item("plot_region_counts")
    )

    def filter_clusters(self, arg):
        conn_ids,cvec = arg
        if cvec.size == 0: return np.array([])

        # convert to boolean
        nz = cvec > 0
        ok_labels = [ conn_ids[m] for m in np.flatnonzero( nz.sum(0) >= self.min_nonzero_required) ]
        print "found", len(ok_labels), "non-zero count region pairs"
        return ok_labels
