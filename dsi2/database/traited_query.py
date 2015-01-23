#!/usr/bin/env python
import numpy as np
import sys, os
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File, Int
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn
from traitsui.extras.checkbox_column import CheckboxColumn
#
from ..ui.ui_extras import colormaps
import cPickle as pickle
import re


"""
How to query for track datasets

Keys:
-----
  ## Scan parameters
  subject_id:
    Unique identifier for a Scan. A Scan_id can be present in
    multiple scans.
  scan_id:
    Unique identifier for a single dataset. Should begin with a
    Scan_id followed by a suffic for which scan.
  scan_gender:
    "male" or "female"
  scan_age:
    Scan age in years.
  study:
    Name of the study/experiment. Each study should have a set of
    discrete Scan groups (for example the study "tbi" might have
    groups "patient" and "control")
  scan_group:
    Which group does this Scan belong to

  ## Tractography parameters
  software:
    DSI studio, DTK or dipy
  smoothing:
    DSI studio smoothing parameter
  cutoff_angle:
    Angle in degrees that determined
  qa_threshold:
    Quantitative anisotropy threshold
  gfa_threshold:
    Generalized FA threshold
  length_min:
    Tracks less than `length_min` have been excluded
  length_max:
    Tracks longer than `length_max` have been excluded


  ## Scanning parameters
  institution:
    Scanning center where the data was acquired (ie "UCSB", "CMU")
  scanner:
    "SIEMENS TIM TRIO"
  n_directions:
    "512",
  max_b_value:
    "5000"
  bvals:
    "",
  bvecs:
    ""
}
"""


class TrackScalarSource(HasTraits):
    """ Contains data specifying the nature of a streamline labeling
    scheme
    """
    # identifying traits
    name = Str("")
    description = Str("")
    parameters = Dict()

    # file paths to data
    numpy_path = File("")
    graphml_path = File("")
    volume_path = File("")
    b0_volume_path = File("")
    qsdr_volume_path = File("")

    scalars = Array

    def load_array(self):
        self.scalars = np.load(
            os.path.join(self.base_dir, self.numpy_path)).astype(np.uint64)
        return self.scalars
    
    def to_json(self):
        return {
            "name" : self.name,
            "description" : self.description,
            "parameters" : self.parameters,
            "numpy_path" : self.numpy_path,
            "graphml_path" : self.graphml_path,
            "b0_volume_path" : self.b0_volume_path,
            "qsdr_volume_path" : self.qsdr_volume_path
        }
    
# ------- Custom column colorizers based on whether the thing will exist 
# ------- after processing 

class b0VolumeColumn(ObjectColumn):
    def get_cell_color(self,object):
        if os.path.exists(object.b0_volume_path):
            return "white"
        return "red"
    
class NumpyPathColumn(ObjectColumn):
    def get_cell_color(self,object):
        if os.path.exists(object.numpy_path):
            return "white"
        return "lightblue"
    
class QSDRVolumeColumn(ObjectColumn):
    def get_cell_color(self,object):
        if os.path.exists(object.qsdr_volume_path):
            return "white"
        if object.parent is None:
            return "red"
        elif not os.path.exists(object.parent.fib_file):
            return "red"
        return "lightblue"
    
scalar_table = TableEditor(
    columns =
    [   ObjectColumn(name="name"),
        b0VolumeColumn(name="b0_volume_path"),
        QSDRVolumeColumn(name="qsdr_volume_path"),
        NumpyPathColumn(name="numpy_path"),
        ObjectColumn(name="description")
    ],
    deletable  = True,
    auto_size  = True,
    show_toolbar = True,
    row_factory = TrackScalarSource
    )

class Dataset(HasTraits):
    scan_id         = Str("")
    subject_id      = Str("")
    scan_gender     = List(["female","male"])
    scan_age        = Int(22)
    study           = Str("")
    scan_group      = Str("")

    software        = List(["DSI Studio","DTK"])
    smoothing       = Range(low=0., high=1., default=0.)
    cutoff_angle    = Range(low=0., high=180., default=55.)
    qa_threshold    = Range(low=0., high=1., default=0.)
    gfa_threshold   = Range(low=0., high=1., default=0.)
    length_min      = Range(low=0., high=100., default=10.)
    length_max      = Range(low=0., high=1000., default=400.)

    institution     = List(["UCSB", "CMU"])
    reconstruction  = Enum("dsi","gqi","qsdr")
    scanner         = List(["SIEMENS TIM TRIO"])
    n_directions    = Range(low=8, high=516, default=512)
    max_b_value     = List([5000, 1000])
    bvals           = List()
    bvecs           = List()


    def __init__(self,**traits):
        super(Dataset,self).__init__(**traits)


class Scan(Dataset):
    scan_gender     = Str("")
    software        = Str("")
    institution     = Str("")
    scanner         = Str("")
    max_b_value     = Int(5000)
    pkl_path        = File("") # Hashed tracks in MNI152
    pkl_trk_path    = File("") # corresponding trk file to check.
    atlases         = Dict({})
    label           = Int
    #data_dir        = File("") # Data used by dsi2 package
    #pkl_dir         = File("") # data root for pickle files
    trk_file        = File("") # path to the trk file
    fib_file        = File("") # path to the DSI Studio's .fib.gz
    trk_space       = Enum("qsdr", "mni") # Which space is it in?
    # ROI labels and scalar (GFA/QA) values
    track_labels    = List(Dict)
    track_scalars   = List(Dict)
    track_label_items    = List(Instance(TrackScalarSource))
    track_scalar_items   = List(Instance(TrackScalarSource))
    # Traits for interactive use
    color_map       = Enum(colormaps)
    dynamic_color_clusters  = Bool(True)
    static_color      = Color
    render_tracks     = Bool(False)
    representation    = Enum("Line", "Tube")
    unlabeled_track_style  = Enum(["Colored","Invisible","White"])
    # raw data from the user
    original_json     = Dict

    import_view = View(
        Group(
          Group(
            Group(
                Item("scan_id"),
                Item("subject_id"),
                Item("scan_gender"),
                Item("scan_age"),
                Item("study"),
                Item("scan_group"),
                Item("fib_file"),
                Item("pkl_path"),
                Item("pkl_trk_path"),
                orientation="vertical",
                show_border=True,
                label="Subject Information"
          ),
            Group(
                Item("software"),
                Item("reconstruction"),
                Item("smoothing"),
                Item("cutoff_angle"),
                Item("qa_threshold"),
                Item("gfa_threshold"),
                Item("length_min"),
                Item("length_max"),
                Item("trk_file"),
                Item("trk_space"),
                orientation="vertical",
                show_border=True,
                label="Reconstruction Information"
          ),
          orientation="horizontal",
          ),
          layout="tabbed"
        ),
        Group(
            Item("track_label_items",editor=scalar_table),
            show_labels=False,
            show_border=True,
            label = "Label values"
            ),
        Group(
            Item("track_scalar_items",editor=scalar_table),
            show_labels=False,
            show_border=True,
            label = "Scalar values"
            ),
        
    )

    def __init__(self,**traits):
        """
        Holds the information OF A SINGLE SCAN.
        """
        super(Scan,self).__init__(**traits)
        
        self.track_label_items = \
            [TrackScalarSource(base_dir=self.pkl_dir, parent=self, **item) for item in \
             self.track_labels ]
        self.track_scalar_items = \
            [TrackScalarSource(base_dir=self.pkl_dir, parent=self, **item) for item in \
             self.track_scalars ]
        self.atlases = dict(
            [ (d['name'],
               {  "graphml_path":d['graphml_path'],
                  "numpy_path":d['numpy_path'],
                } )\
               for d in self.track_labels ])

    @on_trait_change("track_scalar_items,track_label_items")
    def make_me_parent(self):
        #print "\t\t+++setting parent to ", self
        for tli in self.track_label_items:
            tli.parent = self
        for tsi in self.track_scalar_items:
            tsi.parent = self
            
    def get_track_dataset(self):
        pkl_file = self.pkl_path
        print "load:", pkl_file
        fop = open(pkl_file, "rb")
        _trkds = pickle.load(fop)
        _trkds.properties = self
        return _trkds
    
    def to_json(self):
        track_labels = [tl.to_json() for tl in self.track_label_items]
        track_scalars = [ts.to_json() for ts in self.track_scalar_items]
        return {
            "scan_id": self.scan_id, 
            "subject_id": self.subject_id,
            "scan_gender": self.scan_gender,
            "scan_age": self.scan_age,
            "study": self.study,
            "scan_group": self.scan_group,
            "smoothing": self.smoothing,
            "cutoff_angle": self.cutoff_angle,
            "qa_threshold": self.qa_threshold,
            "gfa_threshold": self.gfa_threshold,
            "length_min": self.length_min,
            "length_max": self.length_max,
            "institution": self.institution,
            "reconstruction": self.reconstruction,
            "scanner": self.scanner,
            "n_directions": self.n_directions,
            "max_b_value": self.max_b_value,
            "bvals": self.bvals,
            "bvecs": self.bvecs,
            "label": self.label,
            "trk_space": self.trk_space,
            "pkl_trk_path":self.pkl_trk_path,
            "pkl_path":self.pkl_path,
            "trk_file":self.trk_file,
            "fib_file":self.fib_file,
            "track_labels": track_labels,
            "track_scalars": track_scalars
        }
    
TrackScalarSource.add_class_trait("parent",Instance(Scan))

class MongoScan(Scan):
    mongo_result = Dict({})
    def __init__(self,**traits):
        super(Scan,self).__init__(**traits)
        if "header" in self.mongo_result:
            self.header = pickle.loads(self.mongo_result["header"])    
        else: 
            self.header = np.array([0])
        self.scan_id = self.mongo_result["scan_id"]
        self.subject_id = self.mongo_result["subject_id"]
        self.scan_gender = self.mongo_result["gender"]
        self.scan_age = self.mongo_result["age"]
        self.study = self.mongo_result["study"]
        self.scan_group = self.mongo_result["group"]
        self.smoothing = self.mongo_result["smoothing"]
        self.cutoff_angle = self.mongo_result["cutoff_angle"]
        self.qa_threshold = self.mongo_result["qa_threshold"]
        self.gfa_threshold = self.mongo_result["gfa_threshold"]
        self.length_min = self.mongo_result["length_min"]
        self.length_max = self.mongo_result["length_max"]
        self.institution = self.mongo_result["institution"]
        self.reconstruction = self.mongo_result["reconstruction"]
        self.scanner = self.mongo_result["scanner"]
        self.n_directions = self.mongo_result["n_directions"]
        self.max_b_value = self.mongo_result["max_b_value"]
        self.bvals = self.mongo_result["bvals"]
        self.bvecs = self.mongo_result["bvecs"]
        self.label = self.mongo_result["label"]
        self.trk_space = self.mongo_result["trk_space"]
    
        
        
def potential_multi_query(instr):
    if not instr:
        return None
    # String from gui that may request multiple things
    cleaned = re.split("[,\s]+",instr)
    if len(cleaned) == 1:
        return cleaned[0]
    return {"$in":cleaned}


class Query(Dataset):
    scan_age_min = Int()
    scan_age_max = Int()
    specify_scan_age = Bool() #checkbox	
    use_female = Bool(True)
    use_male = Bool(True)	
    def __init__(self,**traits):
        super(Query,self).__init__(**traits)
    traits_view = View(
        Group(
            VGroup(
                Item('scan_id'),
                Item('subject_id'),
                Item('scan_age_min'),
                Item('scan_age_max'),
                Item('specify_scan_age'),
                Item('use_female'),
                Item('use_male'),
                #Item('scan_gender'),#editor=CheckListEditor()), female/male use bool 
                Item('scan_age'),
                Item('study'),
                Item('scan_group'),
                #Item('attribute'),####################
                show_border=True,
                label="Study Information",
                ),
            VGroup(
                Item('software'),
                Item('smoothing'),
                Item('cutoff_angle'),
                Item('qa_threshold'),
                Item('gfa_threshold'),
                Item('length_min'),
                Item('length_max'),
                springy=True,
                show_border=True,
                label="Tractography Parameters",
                ),
            VGroup(
                Item('institution'),#editor=CheckListEditor()),
                Item('scanner'),
                Item('n_directions'),
                Item('max_b_value'),
                show_border=True,
                label="Acquisition Parameters",
                ),
            layout="tabbed"
            ),
        width=250)

    def error_checking(self):
        #check scan_age_min and scan_age_max is a number less than 0
        if scan_age_min <= 0:
            print "Min age needs to be greater than 0"
        elif scan_age_max <= 0:
            print "Max age needs to be greater than 0"

        #if no number is entered in scan_age_min and scan_age_max
        if scan_age_min == "":
            print "Please enter an age in min age"
        elif scan_age_max == "":
            print "Please enter an age in max age"

    def mongo_query(self):
        query = {}
        
        #scan age
        if self.specify_scan_age:
            query["scan_age_range"] = {"$gt":self.scan_age_min, "$lt":self.scan_age_max}
            
        #scan id
        scan_id_query = potential_multi_query(self.scan_id)
        if not scan_id_query is None:
            query["scan_id"] = scan_id_query
            
        #subject id
        subject_id_query = potential_multi_query(self.subject_id)
        if not subject_id_query is None:
            query["subject_id"] = subject_id_query
            
        #study
        study_query = potential_multi_query(self.study)
        if not study_query is None:
            query["study"] = study_query
            
        #scan group
        scan_group_query = potential_multi_query(self.scan_group)
        if not scan_group_query is None:
            query["scan_group"] = scan_group_query
            
        print query
        return query

    def __check_param(self,paramname,value):
        if getattr(self,paramname) \
           and getattr(self,paramname) == value:
            return True
        return False

    def local_matches(self,dataspec):
        """ Check that the details of dataspec match
        the important proprties of this query object.
        Works for local data only

        Note: KLUUUUUUUUDGE

        """
        #for match_param in ["study","subject_id","software"]:
        for match_param in ["study"]:
            if not self.__check_param(match_param, getattr(dataspec, match_param)):
                return False
        return True
