#!/usr/bin/env python
import numpy as np
import warnings
import sys, os
# Traits stuff
from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, \
     Color, List, Int, Property, Any, Function, DelegatesTo, Str, Enum, \
     on_trait_change, Button, Set, File, Int, CInt, cached_property
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn
from traitsui.extras.checkbox_column import CheckboxColumn
#
from ..ui.ui_extras import colormaps
from dsi2.volumes import get_fib, find_graphml_from_filename
from dsi2.volumes.mask_dataset import MaskDataset
from dsi2.streamlines.track_math import (streamlines_to_ijk, connection_ids_from_voxel_coordinates, 
                            remove_sequential_duplicates, trackvis_header_from_info, 
                            voxels_to_streamlines, voxels_to_streamline_generator,streamline_voxel_lookup)
import cPickle as pickle
import re
from ..volumes import get_builtin_atlas_parameters, get_region_ints_from_graphml
import nibabel as nib


class TrackLabelSource(HasTraits):
    """ Contains connection ids for streamlines
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
    template_volume_path = File("")
    qsdr_volume_path = File("")

    scalars = Array

    def load_array(self):
        self.scalars = np.load(self.numpy_path).astype(np.uint64)
        return self.scalars
    
    @on_trait_change("b0_volume_path")
    def update_params(self):
        if len(self.parameters) == 0:
            self.parameters = get_builtin_atlas_parameters(self.b0_volume_path)
        print self.parameters
        
    def get_tracking_image_filename(self):
        if self.parent.streamline_space == "native":
            atlas_key = "b0_volume_path"
        elif self.parent.streamline_space == "qsdr":
            atlas_key = "qsdr_volume_path"
        elif self.parent.streamline_space == "custom template":
            atlas_key = "template_volume_path"
        else:
            warnings.warn("Unrecognized streamline space, using 'template_volume_path' key")
            atlas_key = "template_volume_path"
        return getattr(self,atlas_key)
        
    def to_json(self):
        return {
            "name" : self.name,
            "description" : self.description,
            "parameters" : self.parameters,
            "numpy_path" : self.numpy_path,
            "graphml_path" : self.graphml_path,
            "b0_volume_path" : self.b0_volume_path,
            "qsdr_volume_path" : self.qsdr_volume_path,
            "template_volume_path" : self.template_volume_path
        }
    
class TrackScalarSource(HasTraits):
    """ Contains scalar data (GFA,QA,etc) for streamlines
    """
    # identifying traits
    name = Str("")
    description = Str("")
    parameters = Dict()

    # file paths to data
    numpy_path = File("")
    txt_path = File("")
    scalars = Array
    
    def __init__(self,**traits):
        super(TrackScalarSource,self).__init__(**traits)
        self.__scalars = None

    def load_array(self):
        self.scalars = np.load(self.numpy_path)
        return self.scalars
    
        
    def get_scalars(self):
        if not self.__scalars is None: return self.__scalars
        # If the array already exists in numpy format, use it
        if os.path.exists(self.numpy_path):
            return self.load_array()
        if not os.path.exists(self.txt_path):
            raise ValueError(self.txt_path + " must exist, but it doesn't")
        # Check to see if it's already summary data (one number per line)
        fop = open(self.txt_path, "r")
        for line in fop:
            break
        fop.close()
        if len(line.strip().split()) == 1:
            try:
                return np.loadtxt(self.txt_path)
            except Exception, e: pass
        fop = open(self.txt_path,"r")
        _scalars = np.array(
            [np.fromstring(line,sep=" ").mean() for line in fop] )
        fop.close()
        np.save(self.numpy_path, _scalars)
        self.__scalars = _scalars
        return self.__scalars
                
        
    def to_json(self):
        return {
            "name" : self.name,
            "description" : self.description,
            "parameters" : self.parameters,
            "numpy_path" : self.numpy_path,
            "txt_path" : self.txt_path,
        }
    
# ------- Custom column colorizers based on whether the thing will exist 
# ------- after processing 

class b0VolumeColumn(ObjectColumn):
    def get_cell_color(self,object):
        if os.path.exists(object.b0_volume_path):
            return "white"
        return "red"
    
class TemplateVolumeColumn(ObjectColumn):
    def get_cell_color(self,object):
        if os.path.exists(object.template_volume_path):
            return "white"
        return "red"
    
class txtColumn(ObjectColumn):
    def get_cell_color(self,object):
        if os.path.exists(object.txt_path):
            return "white"
        return "red"
    
class NumpyPathColumn(ObjectColumn):
    def get_cell_color(self,object):
        if os.path.exists(object.numpy_path):
            return "white"
        return "lightblue"
    
class QSDRVolumeColumn(ObjectColumn):
    def get_cell_color(self,object):
        if object.parent.streamline_space != 'qsdr':
            return "gray"
        if os.path.exists(object.qsdr_volume_path):
            return "white"
        if object.parent is None:
            return "red"
        elif not os.path.exists(object.parent.fib_file):
            return "red"
        return "lightblue"
    
label_table = TableEditor(
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
    row_factory = TrackLabelSource,
    columns_name="input_paths_columns"
    )

scalar_table = TableEditor(
    columns =
    [   ObjectColumn(name="name"),
        txtColumn(name="txt_path"),
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
    scan_age        = CInt(22)
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
    connectivity_matrix_path=File("")
    atlases         = Dict({})
    label           = Int
    trk_file        = File("") # path to the trk file
    fib_file        = File("") # path to the DSI Studio's .fib.gz
    
    # Lazy load the data for labelline
    #streamlines =Instance(Any()) 
    #voxelized_streamlines = Instance(Any())
    from_pkl= Bool(False)
    
    # Specify the volume associated with 
    trk_space       = Enum("qsdr", "mni","custom","native") # Which space is it in?
    streamline_space  = Enum("qsdr", "mni", "custom template", "native") # Which space is it in?
    streamline_space_name = Str("template name")
    template_voxel_size                = Array
    template_volume_shape         =  Array
    template_affine                       =  Array
    template_volume_path            =  File("")
    # ROI labels and scalar (GFA/QA) values
    track_labels    = List(Dict)
    track_scalars   = List(Dict)
    track_label_items    = List(Instance(TrackLabelSource))
    track_scalar_items   = List(Instance(TrackScalarSource))
    # Traits for interactive use
    color_map       = Enum(colormaps)
    dynamic_color_clusters  = Bool(True)
    static_color      = Color((255,0,0))
    render_tracks     = Bool(False)
    representation    = Enum("Line", "Tube")
    unlabeled_track_style  = Enum(["Colored","Invisible","White"])
    # raw data from the user
    original_json     = Dict
    input_paths_columns = Property(depends_on="streamline_space")
    
    def _get_input_paths_columns(self):
        if self.streamline_space == 'qsdr':
            return [   ObjectColumn(name="name"),
                b0VolumeColumn(name="b0_volume_path"),
                QSDRVolumeColumn(name="qsdr_volume_path"),
                NumpyPathColumn(name="numpy_path"),
                ObjectColumn(name="description")
                       ]
        if self.streamline_space in ("mni", 'custom template'):
            return [   ObjectColumn(name="name"),
                TemplateVolumeColumn(name="template_volume_path"),
                NumpyPathColumn(name="numpy_path"),
                ObjectColumn(name="description")
                       ]
        if self.streamline_space == 'native':
            return [   ObjectColumn(name="name"),
                b0VolumeColumn(name="b0_volume_path"),
                NumpyPathColumn(name="numpy_path"),
                ObjectColumn(name="description")
                       ]

    def _transform_qsdr_atlases(self, overwrite=False):
        """ Loops over the atlas labels, mapping them to qsdr space"""
        already_transformed = all(
            [ os.path.exists(ls.get_tracking_image_filename()) for ls in self.track_label_items])
        if already_transformed: return
        
        if not os.path.exists(self.fib_file):
            raise ValueError("fib file could not be found")
        loaded_fib_file = get_fib(self.fib_file)
        for lnum, label_source in enumerate(self.track_label_items):
            abs_qsdr_path = label_source.qsdr_volume_path
            abs_b0_path = label_source.b0_volume_path 
            # If neither volume exists, the data is incomplete
            if not os.path.exists(abs_b0_path):
                print "\t\t++ [%s] ERROR: must have a b0"%self.scan_id
                continue
            # Only overwrite existing data if requested
            ostring = ""
            if os.path.exists(abs_qsdr_path):
                if overwrite: ostring = "(OVERWRITING)"
                else: 
                    print "\t\t++ [%s] skipping mapping b0 label %s to qsdr space"%(
                                                                                        self.scan_id, abs_b0_path)
                    continue
            print "\t\t++ [%s] mapping %s b0 label %s to qsdr space"%(self.scan_id, ostring, abs_b0_path)
            b0_to_qsdr_map(loaded_fib_file, abs_b0_path,
                           abs_qsdr_path)

    def load_streamline_scalars(self):
        for label_source in self.track_scalar_items:
            # File containing the corresponding label vector
            npy_path = label_source.numpy_path if \
                os.path.isabs(label_source.numpy_path) else \
                os.path.join(self.pkl_dir,label_source.numpy_path)
            if os.path.exists(npy_path):
                print npy_path, "already exists"
                continue
            print "\t\t++ saving values to", npy_path
            fop = open(label_source.txt_path,"r")
            scalars = np.array(
                [np.fromstring(line,sep=" ").mean() for line in fop] )
            fop.close()
            np.save(npy_path,scalars)
            print "\t\t++ Done."
        pass

    def label_streamlines(self,overwrite=False):
        
        # Perform qsdr mapping if necessary
        if self.streamline_space == "qsdr":
            self._transform_qsdr_atlases(overwrite=overwrite)
            
        # Configure the voxel grid        
        self.load_template_vol() 
        
        # Load any scalar items associated with the streamlines
        self.load_streamline_scalars()        
            
        # Loop over the track labels, creating .npy files as needed
        n_labels = len(self.track_label_items)
        print "\t+ [%s] Intersecting"%self.scan_id, n_labels, "label datasets"
        for lnum, label_source in enumerate(self.track_label_items):
            # File containing the corresponding label array
            npy_path = label_source.numpy_path 
            print "\t\t++ [%s] Ensuring %s exists" % (self.scan_id, npy_path)
            if os.path.exists(npy_path):
                print "\t\t++ [%s]"%self.scan_id, npy_path, "already exists"
                if not overwrite:    continue
            # Load the volume
            abs_vol_path = label_source.get_tracking_image_filename()
            print "\t\t++ [%s] Loading volume %d/%d:\n\t\t\t %s" % (
                    self.scan_id, lnum + 1, n_labels, abs_vol_path )
            mds = MaskDataset(abs_vol_path)
        
            # Get the region labels from the parcellation
            graphml = find_graphml_from_filename(abs_vol_path)
            if graphml is None:
                print "\t\t++ [%s] No graphml exists: using unique region labels"%self.scan_id
                regions = mds.roi_ids
            else:
                print "\t\t++ [%s] Recognized atlas name, using Lausanne2008 atlas"%self.scan_id, graphml
                regions = get_region_ints_from_graphml(graphml)
    
            # Save it.
            tds = self.get_streamlines()
            conn_ids = connection_ids_from_voxel_coordinates(
                                                                  self.get_voxel_coordinate_streamlines(),
                                                                  mds.dset.get_data(),
                                                                  save_npy=npy_path,
                                                                  atlas_label_int_array=regions)
            print "\t\t++ [%s] Saved %s" % (self.scan_id, npy_path)
            print "\t\t\t*** [%s] %.2f percent streamlines not accounted for by regions"%( 
                                self.scan_id, 100. * np.sum(conn_ids==0)/len(conn_ids) )
            
    def get_voxel_hash(self):
        if not self._voxel_hash is None: return self._voxel_hash
        self._voxel_hash = streamline_voxel_lookup(self.get_voxelized_streamlines())
        return self._voxel_hash
    
    def save_streamline_lookup_in_template_space(self,overwrite=False):
        """
        Writes a final .pkl file to disk containing the streamlines and their mapping to
        template voxel voordinates.
        """
        if not len(self.pkl_path): raise AttributeError("No path sepcified for pkl output")
        needs_pkl = True
        if os.path.exists(self.pkl_path):
            print "pkl "+self.pkl_path + " exists"
            if not overwrite:
                needs_pkl = False
            else: print "overwriting..."
        if needs_pkl:
            ori_x = "R" if self.template_affine[0,0] > 0 else "L"
            ori_y = "A" if self.template_affine[1,1] > 0 else "P"
            ori_z = "S" if self.template_affine[2,2] > 0 else "I"
            output_header = trackvis_header_from_info(ori_x + ori_y + ori_z,
                                                      self.template_volume_shape,
                                                      self.template_voxel_size)
            from dsi2.streamlines.track_dataset import TrackDataset
            output_tds = TrackDataset(tracks=self.get_voxel_coordinate_streamlines(),
                                      header=output_header, coordinate_units="voxels",properties=self)
            output_tds.tracks_at_ijk = self.get_voxel_hash()
            # Write out the pkl file
            print "saving pkl file"
            fop = open(self.pkl_path,"wb")
            pickle.dump(output_tds,fop,pickle.HIGHEST_PROTOCOL)
            fop.close()
        
        # Save out a trk file that is readable by dsi studio
        print "Saving final streamlines in DSI Studio format"
        if len(self.pkl_trk_path):
            needs_trk = True
            if os.path.exists(self.pkl_path):
                print "pkl trk exists"
                if overwrite: print "overwriting..."
                else: needs_trk = False
            if needs_trk:
                dsi_studio_header = trackvis_header_from_info("LPS", self.template_volume_shape,
                                                              self.template_voxel_size)
                nib.trackvis.write(self.pkl_trk_path, voxels_to_streamline_generator(self.get_voxel_coordinate_streamlines(),
                                            volume_affine=self.template_affine, volume_shape=self.template_volume_shape,
                                            volume_voxel_size=self.template_voxel_size, voxmm_orientation="LPS",for_writing=True),
                                   dsi_studio_header)
            
    def clearmem(self):
        del self._streamlines
        self._streamlines = None
        del self._voxel_coordinate_streamlines
        self._voxel_coordinate_streamlines = None
        del self._voxelized_streamlines
        self._voxelized_streamlines = None
        del self._voxel_hash
        self._voxel_hash = None
        
            
        
    def __init__(self,**traits):
        """
        Holds the information OF A SINGLE SCAN.
        """
        super(Scan,self).__init__(**traits)
        self._streamlines = None
        self._voxelized_streamlines = None
        self._voxel_coordinate_streamlines = None
        self._voxel_hash = None
        
        self.track_label_items = \
            [TrackLabelSource(base_dir=self.pkl_dir, parent=self, **item) for item in \
             self.track_labels ]
        self.track_scalar_items = \
            [TrackScalarSource(base_dir=self.pkl_dir, parent=self, **item) for item in \
             self.track_scalars ]
        self.atlases = dict(
            [ (d['name'],
               {  "graphml_path":d.get('graphml_path',None),
                  "numpy_path":d['numpy_path'],
                } ) \
               for d in self.track_labels ])
        
    def load_template_vol(self):
        """Loads the template voxel grid and makes sure all atlases match"""
        # Configure the output volume space
        if self.streamline_space == "custom template":
            try:
                template_vol = nib.load(self.template_volume_path)
            except Exception,e:
                raise ValueError("Cannot open template image at " + \
                                 template_vol_path + ": " + e)
            # Load template information into the headers
            self.template_voxel_size = np.array(template_vol.get_header().get_zooms())
            self.template_volume_shape = template_vol.shape
            self.template_affine = template_vol.get_affine()
            
        elif self.streamline_space == "qsdr":
            from dsi2.volumes import QSDR_SHAPE, QSDR_AFFINE, QSDR_VOXEL_SIZE
            self.template_voxel_size = QSDR_VOXEL_SIZE
            self.template_volume_shape = np.array(QSDR_SHAPE)
            self.template_affine = QSDR_AFFINE
            
        elif self.streamline_space == "mni":
            from dsi2.volumes import get_MNI152
            template_vol = get_MNI152()
            self.template_voxel_size = np.array(template_vol.get_header().get_zooms())
            self.template_volume_shape = template_vol.shape
            self.template_affine = template_vol.get_affine()
            warnings.warn("If your tracking was done on a qsdr dataset you should"
                          "use 'qsdr' as the streamline space")
                                                  
        # If native space atlases, make sure that they all have the same grid
        if self.streamline_space == "native":
            if not len(self.track_label_items):
                raise ValueError("must supply b0-space label images")
            example = nib.load(self.track_label_items[0].b0_volume_path)
            self.template_affine = example.get_affine()
            self.template_volume_shape = example.shape
            self.template_voxel_size = example.get_header().get_zooms()
            
        # Check that all the atlases match the output template
        for label in self.track_label_items:
            img = nib.load(label.get_tracking_image_filename())
            affines_match = np.allclose(self.template_affine, img.get_affine())
            grids_match = np.all(self.template_volume_shape == img.shape)
            if not affines_match and grids_match:
                raise ValueError("All native space atlases must be on the same grid"
                        " but " + label.name + " " + label.description + "does not match" )
                

    @on_trait_change("track_scalar_items,track_label_items")
    def make_me_parent(self):
        #print "\t\t+++setting parent to ", self
        for tli in self.track_label_items:
            tli.parent = self
        for tsi in self.track_scalar_items:
            tsi.parent = self
            
    def __len__(self):
        return 1
    
    def get_voxelized_streamlines(self):
        if self._voxelized_streamlines is None:
            print "calculating voxelized streamlines"
            self._voxelized_streamlines = np.array([
                remove_sequential_duplicates(stream.astype(np.int)) for stream in \
                                                                self.get_voxel_coordinate_streamlines()])
        return self._voxelized_streamlines
    
    def get_voxel_coordinate_streamlines(self):
        if self._voxel_coordinate_streamlines is None:
            streamlines = self.get_streamlines()
            # If loaded from a pkl, we're already in voxel coordinates
            if self.from_pkl: return streamlines.tracks
            self._voxel_coordinate_streamlines = \
                streamlines_to_ijk(
                                           streamlines.tracks, trackvis_header=streamlines.header,
                                           tracking_volume_shape=self.template_volume_shape,
                                           tracking_volume_voxel_size=self.template_voxel_size,
                                           tracking_volume_affine=self.template_affine,
                                           return_coordinates="voxel_coordinates")
        return self._voxel_coordinate_streamlines
    
    
    def get_streamlines(self):
        if not self._streamlines is None: return self._streamlines
        print "getting streamlines"
        from dsi2.streamlines.track_dataset import TrackDataset
        if os.path.exists(self.pkl_path):
            pkl_file = self.pkl_path
            print "load:", pkl_file
            fop = open(pkl_file, "rb")
            _trkds = pickle.load(fop)
            _trkds.properties = self
            self.from_pkl = True
        elif os.path.exists(self.trk_file):
            _trkds = TrackDataset(self.trk_file, properties=self)
        self._streamlines=_trkds
        
        # Loop over the scalar items
        self.load_streamline_scalars()
        for scalar_item in self.track_scalar_items:
            setattr(_trkds,
                    scalar_item.name,
                    scalar_item.get_scalars())
            
        print "done." 
        return self._streamlines
    
                
    def get_track_dataset(self):
        return self.get_streamlines()
    
    def to_json(self):
        track_labels = [tl.to_json() for tl in self.track_label_items]
        track_scalars = [ts.to_json() for ts in self.track_scalar_items]
        return {
            "scan_id": self.scan_id, 
            "template_volume_path":self.template_volume_path,
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
            "streamline_space": self.streamline_space,
            "streamline_space_name": self.streamline_space_name,
            "pkl_trk_path":self.pkl_trk_path,
            "pkl_path":self.pkl_path,
            "trk_file":self.trk_file,
            "fib_file":self.fib_file,
            "track_labels": track_labels,
            "track_scalars": track_scalars,
            "software":self.software,
            "connectivity_matrix_path":self.connectivity_matrix_path
        }
    
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
                Item("connectivity_matrix_path"),
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
                Item("streamline_space"),
                Item("streamline_space_name",visible_when="streamline_space=='custom template'"),
                Item("template_volume_path",visible_when="streamline_space=='custom template'"),
                orientation="vertical",
                show_border=True,
                label="Reconstruction Information"
          ),
          orientation="horizontal",
          ),
          layout="tabbed"
        ),
        Group(
            Item("track_label_items",editor=label_table),
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
        self.scan_gender = self.mongo_result.get("gender","N")
        self.scan_age = self.mongo_result.get("age",0)
        self.study = self.mongo_result["study"]
        self.scan_group = self.mongo_result.get("group","")
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
        self.streamline_space = self.mongo_result.get("streamline_space","qsdr")
    
        
        
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
