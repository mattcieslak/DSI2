#!/usr/bin/env python
from nibabel import trackvis
import numpy as np
from copy import deepcopy
from ..streamlines import track_math
from ..volumes.mask_dataset import MaskDataset
from collections import defaultdict
import cPickle as pickle
#from dipy.tracking.utils import streamline_mapping
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn

from traits.api import HasTraits, Instance, Array, \
    CInt, Color, Bool, List, Int, Str, Instance, Any, Enum, \
    DelegatesTo, on_trait_change, Button
import cPickle as pickle
from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi import mlab
import gzip


from .track_math import tracks_to_endpoints
from mayavi.core.api import PipelineBase, Source

mni_hdr = trackvis.empty_header()
mni_hdr['dim'] = np.array([91,109,91],dtype="int16")
mni_hdr['voxel_order'] = 'LAS'
mni_hdr['voxel_size'] = np.array([ 2.,  2.,  2.], dtype='float32')
mni_hdr['image_orientation_patient'] = np.array(
                     [ 1.,  0.,  0.,  0.,  -1.,  0.], dtype='float32')
mni_hdr['vox_to_ras'] = \
           np.array([[ -2.,  0.,  0.,   90.],
                     [  0.,  2.,  0., -126.],
                     [  0.,  0.,  2.,  -72.],
                     [  0.,  0.,  0.,    1.]], dtype='float32')

def join_tracks(args):
    """
    combines a set of TrackDatasets into a single TrackDataset
    Parameters:
    -----------
    args:list of TrackDatasets
      a bunch of track datasets to be joined together

    Returns:
    --------
    newTrackDataset:
      A TrackDataset containing the tracks from all the args

    """
    assert len(args)
    trackful = [ds for ds in args if ds.tracks.size > 0]
    if len(trackful) < 1:
        return None
    if len(trackful) == 1:
        return trackful[0]
    all_tracks = []
    header = args[0].header
    for ds in args:
        all_tracks.extend([trk for trk in ds.tracks])
    
    return TrackDataset(header = header, 
                        tracks=all_tracks)

def random_color(*args):
    return (np.random.rand(),np.random.rand(),np.random.rand())

class Cluster(HasTraits):
    color       = Color
    mcolor      = Array(shape=(4,))
    visible     = Bool(True)
    render3d    = Bool(False)
    representation = Enum("Barbell","Splater")
    ntracks     = Int
    scan_id     = Str
    id_number   = Int
    start_coordinate = Array(shape=(3,))
    end_coordinate   = Array(shape=(3,))
    indices          = Array
    cluster_type     = "POINT"

    def _mcolor_changed(self):
        self.color = str(tuple(self.mcolor))
        
class Segment(HasTraits):
    color       = Color
    mcolor      = Array(shape=(4,))
    visible     = Bool(True)
    render3d    = Bool(False)
    representation = Enum("Barbell","Splater")
    ntracks     = Int
    ncoords     = Int
    scan_id     = Str
    segment_id   = Int
    start_coordinate = Array(shape=(3,))
    end_coordinate   = Array(shape=(3,))
    indices          = Array
    cluster_type     = "POINT"

    def _mcolor_changed(self):
        self.color = str(tuple(self.mcolor))

class RegionCluster(Cluster):
    start_coordinate = Str
    end_coordinate = Str
    cluster_type   = "REGION"
    connection_id  = CInt


class TrackDataset(HasTraits):
    # Holds original data: Never changes
    tracks = Array
    scalars = Array
    connections = Array
    labels = Array
    gfa = Array
    qa = Array
    offsets=Array
    properties = Any
    segments = List(Instance(Segment))
    #original_track_indices = Array(np.array([]))

    # Are the glyphs already rendered?
    tracks_drawn = Bool(False)
    clusters_drawn = Bool(False)
    empty = Bool(False)
    min_pts = Int(10)
    length_filter = Bool(False)

    # Holds the properties from dsi2.traited_query.Scan
    render_tracks =DelegatesTo('properties') # Needed so subsets inherit the visibility state
    scan_id = DelegatesTo('properties')
    dynamic_color_clusters=DelegatesTo('properties')
    static_color = DelegatesTo('properties')
    color_map  = DelegatesTo('properties')
    unlabeled_track_style = DelegatesTo('properties')

    def _static_color_changed(self):
        if not self.dynamic_color_clusters:
            self.color_scheme_changed()

    @on_trait_change("properties.color_map,properties.dynamic_color_clusters")
    def color_scheme_changed(self):
        print "\t++ TrackDataset color scheme changed"
        #if self.render_tracks:
        if self.tracks_drawn:
            self.paint_clusters()

    representation = DelegatesTo("properties")
    def _representation_changed(self):
        print "\t++ TrackDataset.representation is now", self.representation
        if self.render_tracks:
            if self.tracks_drawn:
                self.remove_glyphs()
            self.draw_tracks()

    # How should the clusters be displayed?
    clusters = List(Instance(Cluster))
    cluster_representation = Enum("Barbell","Splatter")
    storage_coords = Str("voxmm")

    # mayavi objects
    pts = Instance(PipelineBase)
    trk_lines = Instance(PipelineBase)
    src = Instance(PipelineBase)
    b_clear_glyphs = Button(label="clear glyphs")
    def _b_clear_glyphs_fired(self):
        self.remove_glyphs()

    data_source_view = View(
        Group(
            Item("dynamic_color_clusters"),
            Item("static_color"),
            Item("color_map"),
            Item("render_tracks"),
            Item("representation"),
            )
        )
    graphics_view = View(
        # Graphics objects
        Group(
            Item("b_clear_glyphs"),
            Item("pts"),
            Item("trk_lines"),
            Item("src"),
            Item("tracks_drawn"),
            ),
    )

    def __init__(self, fname="", tracks=None, header=None,
                 connections=np.array([]),
                 original_track_indices=np.array([]),
                 properties=None,**traits):
        """Convenience class for loading, manipulating and storing
        trackvis data.
        Parameters
        ----------
        fname:str
          path to the trackvis .trk file
        streams:list
          If there is already a list of tracks
        header:dict
          Trackvis header file
        connections:np.ndarray[dtype=int,ndim=1]
          array with connection Id's for each track
        storage_coords: {"ijk", "voxmm", "MNI", "qsdr"}
          How should this object store the tracks internally? Assumes
          that the original coords are in voxmm for trackvis
        properties: traited_query.Scan object
          information about this dataset
        """

        #if not None in (tracks,header):
        if not (header is None):
            # Tracks and header are provided
            self.header = header
            self.set_tracks(tracks)
        else:
            if fname.endswith("trk.gz"):
                fl = gzip.open(fname,"r")
            elif fname.endswith("trk"):
                fl = open(fname,"r")
                streams, self.header = trackvis.read(fl)
                # Convert voxmm to ijk
                self.set_tracks(np.array([stream[0] for stream in streams],
                                     dtype=np.object))
                fl.close()
                # Check for scalars, support them someday
                if self.header['n_scalars'] > 0:
                    print "WARNING: Ignoring track scalars in %s"%fname
            elif fname.endswith("txt"):
                fop = open(fname,"r")
                self.set_tracks( np.array(
                    [np.array(map(float, line.strip().split())).reshape(-1,3) for \
                     line in fop ], dtype=np.object ))
            elif fname.endswith("mat"):
                pass

        if properties is None:
            from dsi2.database.traited_query import Scan
            print "Warning: using default properties"
            self.properties = Scan()
        else:
            self.properties = properties

        self.connections = connections
        self.clusters = []
        self.original_track_indices = original_track_indices

    def __iter__(self):
        for trk in self.tracks:
            yield trk
            
    def set_connections(self,connections):
        """ directly set connections with a numpy array"""
        # TODO: remove the hasattr check
        if not hasattr(self,"original_track_indices") or self.original_track_indices.size == 0:
            self.connections = connections
            print "\t\t\t*** set full connections"
        else:
            self.connections = connections[self.original_track_indices]
            print "\t\t\t*** set a subset of connections"

    def load_connections(self,atlas_name):
        """Only reqd for non-dynamic region aggregator"""
        pth = self.properties.atlases[atlas_name]["numpy_path"]
        connections = np.load(pth)
        self.set_connections(connections)

    def set_tracks(self,tracklist):
        """Set new tracklist.
        Parameters
        ==========
        tracklist: list | set | np.ndarray(dtype=object)
          if list or set of int, select these indices
          np.ndarray: if objects, set these as tracks, else use them to slice
                      the current tracks
        """
        # convert set so it's indexable
        if type(tracklist) == set:
            print "WARNING: ordering of tracks lost!!"
            tracklist = list(tracklist)

        if type(tracklist) == list:
            if not len(tracklist): return
            if type(tracklist[0]) == int:
                # a list of indices to select from a previously existing tracklist
                if not hasattr(self,"tracks"):
                    raise IndexError("No tracks exist to select from by index")
                #print "pulling tracks from list of indices"
                new_tracks = self.tracks[np.array(tracklist)]
            elif type(tracklist[0]) == np.ndarray:
                # it is a list of arrays
                #print "Setting new tracks from list of tracks"
                new_tracks = np.array(tracklist,dtype=np.object)
        elif type(tracklist) == np.ndarray:
            if not tracklist.size: return
            if tracklist.dtype in (object,np.float64):
                #print "using a numpy object array"
                new_tracks = tracklist
            else:
                # It is an array of indices
                #print "using indices from ndarray to select tracks"
                new_tracks = self.tracks[tracklist]
        elif not (tracklist is None):
            raise ValueError("unable to select tracks from a %s"%type(tracklist))
        else:
            new_tracks = np.array([])
        self.tracks = new_tracks
        
        if self.tracks_drawn:
            self.remove_glyphs()
            if self.render_tracks:
                self.draw_tracks()
                

    def get_ntracks(self):
        return self.tracks.shape[0]

    def _compute_track_lengths(self):
        """Compute the length of the tracks in mm units
        """
        ntracks = len(self.tracks)
        track_lens = np.zeros(ntracks,)
        for ntrk,trk in enumerate(self):
            track_lens[ntrk] = track_math.euclidean_len(trk)
        return track_lens

    def length_filter(self,minlength=0,new=False):
        """ Remove all tracks with length (in mm) less than `minlength`
        Operates in-place unless new=True, in which case a new TrackDataset
        is returned and the current is left alone.
        """
        long_enough = self.track_lengths>=minlength
        if new:
            return self.subset(long_enough)
        survivors = self.tracks[self.track_lengths>=minlength]
        self.set_tracks(survivors)


    def subset(self,aset,inverse=False,every=0):
        """ Create a new TrackDataset from a list or set of track ids.
        Parameters:
        -----------
        aset: array-like or set
          Contains the indices of selected streamlines
        inverse: Bool
          Return a subset containing streamlines in `aset` or not in `aset`
        every: int
          If 0, returns all streamlines in aset. Otherwise every nth streamline
          is returned, beginning with the nth streamline. every=1 is equivalent
          to every=0. Use `every` to downsample your tracks to prevent overplotting.

        Returns:
        --------
         ds: TrackDataset
        """
        if every < 0:
            raise ValueError("every must be >= 0.")

        if type(aset) == set:
            idx = np.array(sorted(list(aset)),dtype=int)
        else:
            idx = aset
        if inverse:
            _trks = np.arange(self.get_ntracks(),dtype=int)
            idx = np.setdiff1d(_trks,idx)

        # Enable arbitrary downsampling through `every`
        if every < 1:
            every = 1

        survivors = self.tracks[idx] if hasattr(self,"tracks") and self.tracks.size > 0 else None
        if every > 1 and survivors.size > 0:
            survivors = survivors[every-1::every]
            
        # Similarly, subset the additional info
        connections = self.connections[idx] if self.connections.size > 0 \
                else np.array([])
        gfa = self.gfa[idx] if self.gfa.size > 0 \
                else np.array([])
        qa = self.qa[idx] if self.qa.size > 0 \
                else np.array([])
        
        if every > 1:
            if connections.size > 0:
                connections = connections[every-1::every]
            if gfa.size > 0:
                gfa = gfa[every-1::every] 
            if qa.size > 0:
                qa = qa[every-1::every] 
            
            
        # If we're subsetting a subset, get the original indices
        # TODO: my old pickles don't have this builtin. Remove hasattr in a future version
        if not hasattr(self,"original_track_indices") or self.original_track_indices.size==0:
            orig_idx = idx[every-1::every]
        elif self.original_track_indices.size > 0:
            orig_idx = self.original_track_indices[idx][every-1::every]

        return TrackDataset(tracks=survivors, header=self.header,
                               connections=connections, properties=self.properties,
                               original_track_indices=orig_idx,
                               qa=qa, gfa=gfa
                            )

    def voxmm_to_ijk(self, trk, to_order= "", floor=True):
        """Converts from trackvis voxmm to ijk.
        Parameters:
        ===========
        trk:np.ndarray (N,3) or int
          if ndarray, it will be treated as a list of xyz coordinates. If an
          int, it will return the ijk coordinates of that track id
        to_order: "[LR]" + "[AP]" + "[IS]"
          index value increases in the direction of the value.
          This string specifies how you want ijk to work in the OUTPUT.
          Data is flipped from the order specified in self.header['voxel_order']
        floor:bool
          Should the data be truncated to an integer? This is required if the purpose
          is to create a hash table. Otherwise, you can get a pretty track for display
          if this is set to False.
        Returns:
        ========
        (N,3) ndarray of ijk indices

        NOTE:
        =====
         REQUIRES that this trackdataset came from a trk file, as the header is
         essential to transforming tracks to ijk.

        """
        # as per voxmm standard, simply divide x,y,z by voxel size.
        ijk = trk.astype(np.float64) / self.header['voxel_size']

        # the goal is to hash the tracks, take them to ints using floor
        if floor:
            ijk = np.floor(ijk).astype(np.int32)

        # Do any of the axes need to be flipped?
        if len(to_order)==3:
            voxmm_order = str(self.header['voxel_order'])
            if not voxmm_order[0] == to_order[0]:
                ijk[:,0] = self.header['dim'][0] - ijk[:,0]
            if not voxmm_order[1] == to_order[1]:
                ijk[:,1] = self.header['dim'][1] - ijk[:,1]
            if not voxmm_order[2] == to_order[2]:
                ijk[:,2] = self.header['dim'][2] - ijk[:,2]
        if floor:
            unq = track_math.remove_duplicates(ijk).astype(np.int32)
            return unq
        return ijk

    def get_tracks_by_connection_id(self,connection_id,return_fail_fibers=False):
        """Get a set of all fiber_ids that are labeled
        Parameters
        ==========
        connection_id:int or np.ndarray
          the value(s) in self.connections to collect streamlines from

        Returns
        =======
        roi_fibers: np.ndarray
           indices of streamlines that intersect the requested region pairs
        """
        if not hasattr(self,"connections"):
            print "no connection id's available"
            return set([])
        if type(connection_id) == int:
            connection_id = np.array([connection_id])
        elif type(connection_id) != np.ndarray:
            connection_id = np.array(connection_id)

        return np.flatnonzero(np.in1d(self.connections,connection_id))

    def get_tracks_by_ijks(self,ijks,return_fail_fibers=False):
        """Get a set of all fiber_ids that pass through roi
        Parameters
        ==========
        ijks: list of tuples
          a list of ijk coordinates.

        Returns
        =======
        roi_fibers: set
           set of fibers that intersect the ijk's
        fail_fibers: set
           set of fibers that fail to intersect ijk's. Only
           returned if return_fail_fibers == True
        """
        if not hasattr(self,"tracks_at_ijk"): self.hash_voxels_to_tracks()
        roi_fibers = set()
        for _ijk in ijks:
            roi_fibers.update(self.tracks_at_ijk[_ijk])
        #
        if not return_fail_fibers:
            return roi_fibers
        else:
            all_fibers = np.arange(len(self.tracks))
            fail_fibers = np.setdiff1d(all_fibers,
                                       np.array(list(roi_fibers))
                                       )
            return roi_fibers, set(fail_fibers.tolist())

    def hash_voxels_to_tracks(self, **kwargs):
        """
        If no arguments are given, voxmm coordinates are simply
        divided by the voxel size in the header. Otherwise

        """
        #self.tracks_at_ijk = streamline_mapping(self.tracks,
        ijk_tracks = np.array([
            self.voxmm_to_ijk(trk,**kwargs) for trk in self],
                              dtype=object)

        # index tracks by the voxels they pass through
        tracks_at_ijk = defaultdict(set)
        for trknum, ijk in enumerate(ijk_tracks):
            data = set([trknum])
            for _ijk in ijk:
                tracks_at_ijk[tuple(_ijk)].update(data)
        self.tracks_at_ijk = tracks_at_ijk

    def dump_qsdr2MNI_track_lookup(self, output, savetrk=False):
        """Converts from trackvis voxmm to ijk.
        Parameters:
        ===========
        output:str
          .pkl file where the voxel to streamline id mapping is saved
        savetrk:str
          path to where the trackvis file will be saved.

        NOTE:
        =====
         REQUIRES that this trackdataset came from a trk file, as the header is
         essential to transforming tracks to ijk.

         From experience, tracks brought into alignment with the MNI152 template
         via DTK's track_transform appear as LAS -> LPS in trackvis's dataset info panel.
         I have been able to get qsdr2MNI to look correct in trackvis by converting
         it to LAS ordering. LAS also relates voxmm coordinates to MNI152 ijk by
         a scalar factor.

        """
        # index tracks by the voxels they pass through
        mni_voxel_size = np.array([2.]*3)
        tracks_at_ijk = defaultdict(set)
        output_voxmm = []
        for trknum, trk in enumerate(self.tracks):
            data = set([trknum])
            # convert voxmm to LAS
            ijk = self.voxmm_to_ijk(trk, to_order="LAS")
            pretty_ijk = self.voxmm_to_ijk(trk, to_order="LAS", floor=False)
            # QSDR from DSI Studio has a different bounding box.
            ijk = ijk + np.array([6,7,11])
            pretty_ijk = pretty_ijk + np.array([6,7,11])

            output_voxmm.append(pretty_ijk*mni_voxel_size)
            # Floor at the end?
            unq = track_math.remove_duplicates(ijk).astype(np.int32)
            for _ijk in unq:
                tracks_at_ijk[tuple(_ijk)].update(data)
        self.tracks_at_ijk = tracks_at_ijk
        print "original", self.tracks.shape, "tracks"
        self.tracks = np.array(output_voxmm)
        print "replaced by", self.tracks.shape, "tracks"

        # Save the hash dump
        fop = open(output,"wb")
        pickle.dump(self,fop,pickle.HIGHEST_PROTOCOL)
        fop.close()

        # Write out a new trackvis file
        if savetrk:
            # Actually write a trk file so we can check against the
            #   MNI brain in trackvis
            trackvis.write(
                savetrk,
                ((stream*mni_hdr['voxel_size'],None,None) for stream in output_voxmm),
                np.array(mni_hdr)
                )

    def save(self,fname,use_mni_header=False):
        """Save the object as a .trk file"""
        header = mni_hdr if use_mni_header else self.header
        trackvis.write(
            fname,
            ((stream,None,None) for stream in self),
            np.array(header)
        )
        
    def dump_voxel_track_lookup(self, output,**kwargs):
        """dumps this TrackDataset with all its lookup tables
        into a binary pickle file.
        """
        self.hash_voxels_to_tracks(**kwargs)
        fop = open(output,"wb")
        pickle.dump(self,fop,pickle.HIGHEST_PROTOCOL)
        fop.close()


    #==============================================================
    # Functions from the former TraitedTrackDataset
    #==============================================================
    def _name_default(self):
        return self.properties.scan_id

    def get_mayavi_kw(self):
        # Make arguments specific to the color settings
        if not self.dynamic_color_clusters:
            return {"color":(
                                    self.static_color.red()/255.,
                                    self.static_color.green()/255.,
                                    self.static_color.blue()/255.)
            }
        if self.color_map == "random":
            return {"colormap":random_color}
        return {"colormap":self.color_map}

    def draw_tracks(self):
        # If the tracks are already drawn and we don't need a re-draw
        if not self.render_tracks:
            print "\t\t++ TrackDataset.render_tracks is False, exiting"
            return
        if self.tracks_drawn:
            print "\t\t tracks drawn, FORCING re-drawing to tracks"
            self.remove_glyphs()

        if self.unlabeled_track_style == "Invisible":
            if not self.labels.size:
                print "\t\t special coloring requested but no labels present"
                return
            tracks = [ trk for trk,label in zip(self.tracks,self.labels) if label > 0 ]
        else:
            tracks = self.tracks
        self.offsets = np.cumsum(np.array([0] + [ trk.shape[0] for trk in tracks ]))
        connect = []
        allpts = []
        npts = 0
        
        for line in tracks:
            allpts.append(line)
            # Number of points in the line
            _npts = line.shape[0]
            conns = np.vstack(
                               [np.arange(npts, npts+_npts - 1.5),
                                np.arange(npts+1, npts+_npts -.5)]).T
            connect.append(conns)
            npts += _npts
        # If there are no tracks, set drawn to True and exit
        if len(allpts) == 0:
            print "\t\t++ No Tracks to draw"
            self.empty = True
            self.tracks_drawn = True
            return

        # TODO: Make scaling automatic
        self.x,self.y,self.z = np.vstack(allpts).T/2
        #self.x,self.y,self.z = np.vstack(allpts).T
        self.links = np.vstack(connect)

        # How will these shapes be constructed in mayavi?
        mayavi_args = self.get_mayavi_kw()

        # Initialize scalars to ones
        self.scalars = np.zeros(len(self.x))
        self.pts = mlab.pipeline.scalar_scatter(
            self.x,self.y,self.z,
            s=self.scalars,
            #figure=scene.mayavi_scene,
            name="streamlines",
            **mayavi_args)
        self.pts.mlab_source.dataset.lines = self.links
        self.trk_lines = mlab.pipeline.stripper(self.pts)#,
                                                #figure=scene.mayavi_scene)

        try:
            if self.representation == "Tube":
                self.src = mlab.pipeline.surface(
                                mlab.pipeline.tube(self.trk_lines, tube_radius=0.2),
                                #figure=scene.mayavi_scene, name="streamlines",
                                **mayavi_args)
            elif self.representation == "Line":
                self.src = mlab.pipeline.surface(self.trk_lines,line_width=2,
                                             #figure=scene.mayavi_scene, name="streamlines",
                                             **mayavi_args)
        except Exception, e:
            print "\t\t+++ Mysterious MayaVi Error", e

        self.tracks_drawn = True
        print "\t\t++ Set tracks_drawn to", self.tracks_drawn

        # IF there are clusters, try painting them
        print "\t\t\t*** Painting clusters on a fresh set of glyphs"
        if len(self.clusters): self.paint_clusters()

    def remove_glyphs(self):
        #print "\t\t++ Removing glyphs"
        if not self.tracks_drawn:
            #print "\t\t++ No glyphs to remove, Done."
            return
        for viz in ['src', 'trk_lines', 'pts']:
            try:
                getattr(self,viz).remove()
            except Exception, e:
                print "\t+++ Failed to TrackDataset.%s.remove()" % viz
                print "\t\t",e
        self.tracks_drawn = False
        #print "\t\t++ glyphs removed."

    def set_track_visibility(self, visibility):
        """ Toggles the visibility of streamline glyphs.
        * IF visibility is requested and the glyphs are not drawn,
          they will be drawn
        * If they are drawn, then visibility is toggled
        """
        self.render_tracks = visibility
        # User does NOT want to see tracks
        if not visibility:
            # If they're not drawn, do nothing
            if not self.tracks_drawn: return
        # User DOES want to see tracks
        else:
            # Draw them if they don't exist
            if not self.tracks_drawn:
                print "tracks not drawn, drawing them"
                self.draw_tracks()
        # Otherwise set the visibility of the glyphs to False
        for viz in [self.src, self.trk_lines, self.pts]:
            if viz:
                viz.visible = visibility

    def set_cluster_visibility(self, visibility):
        # Remove current cluster barbells
        for item in ["cluster_glyphs", "cluster_tubes", "cluster_src", "splatters"]:
            if hasattr(self,item):
                getattr(self,item).visible = visibility

    def draw_clusters(self):
        # if there are no clusters, get out of here
        if len(self.clusters) == 0 or self.clusters[0].cluster_type == "REGION": return
        if self.clusters_drawn: return
        if not self.render_tracks: return
        if self.cluster_representation == "Barbell":
            self.draw_cluster_barbells()
        elif self.cluster_representation == "Splatter":
            self.draw_cluster_splatter()
        self.clusters_drawn = True

    def draw_cluster_splatter(self):
        tep = tracks_to_endpoints(self.tracks)/2
        self.splatters = []
        for clust in self.clusters:
            if not self.dynamic_color_clusters:
                _color = self.static_color.red/255., \
                         self.static_color.green/255., \
                         self.static_color.blue/255.
            else:
                _color = clust.color[0]/255., \
                         clust.color[1]/255., \
                         clust.color[2]/255.,

            x,y,z = tep[clust.indices].reshape(-1,3).T
            self.splatters.append(
                mlab.pipeline.volume(
                    mlab.pipeline.gaussian_splatter(
                        mlab.pipeline.scalar_scatter(
                            x,y,z
                        ),
                    ),color=_color
                )
            )

    def draw_cluster_barbells(self,scene):
        scalars = []
        allpts = []
        for clust in self.clusters:
            allpts += [clust.start_coordinate, clust.end_coordinate]
            scalars += [clust.id_number]*2

        # Connections between points
        idx = np.arange(len(allpts)/2, dtype=int)*2
        links = np.vstack([idx, idx+1]).T
        c_x, c_y, c_z = np.vstack(allpts).T

        mayavi_args = self.get_mayavi_kw()
        self.cluster_glyphs = \
          mlab.points3d(
            c_x, c_y, c_z, np.array(scalars),
            mode = 'cube',
            scale_factor = 2.5, scale_mode='none',
            figure=scene.mayavi_scene,
            **mayavi_args)

        self.cluster_glyphs.mlab_source.dataset.lines = links
        lines = mlab.pipeline.stripper(
            self.cluster_glyphs, figure=scene.mayavi_scene)
        self.cluster_tubes = mlab.pipeline.tube(lines,tube_radius=0.5)

        self.cluster_src = mlab.pipeline.surface(self.cluster_tubes,line_width=1,
                                         figure=scene.mayavi_scene,
                                         **mayavi_args)


    def set_clusters(self, clusters, update_glyphs=True):
        """
        Applies a set of cluster assignments to the tracks, then
        loops over the clusters and applies the
        """
        self.clusters = clusters
        if not update_glyphs: return
        if self.empty: return

        # If this object has a static color, our job is easy
        if not self.dynamic_color_clusters:
            for clust in self.clusters:
                clust.color = self.static_color
            return

        if self.tracks_drawn:
            self.paint_clusters()

    def paint_clusters(self):
        print "\t\t\t+++ painting", len(self.clusters), "clusters to previously drawn glyphs"
        self.scalars = np.zeros(len(self.x))
        # If there are no labels set, then we can't paint the clusters
        if not self.labels.size: return
        labels = np.unique(self.labels)
        
        for clust in self.clusters:
            for idx in clust.indices:
                try:
                    self.scalars[self.offsets[idx]:self.offsets[idx+1]] = \
                        clust.id_number
                except IndexError:
                    print idx, "out of bounds"
        self.src.mlab_source.scalars = self.scalars
        cmap = self.src.module_manager.scalar_lut_manager.lut.table.to_array()
        idx_fetcher = self.src.module_manager.scalar_lut_manager.lut.get_index
        for clust in self.clusters:
            clust.mcolor = cmap[idx_fetcher(clust.id_number)]
            
            
    def set_segments(self, clusters, update_glyphs=True):
        """
        Applies a set of cluster assignments to the tracks, then
        loops over the clusters and applies the
        """
        self.segments = clusters
        if not update_glyphs: return
        if self.empty: return

        # If this object has a static color, our job is easy
        if not self.dynamic_color_clusters:
            for seg in self.segments:
                seg.color = self.static_color
            return

        if self.render_tracks:
            if not self.tracks_drawn:
                self.draw_tracks()
            self.paint_segments()
            
    def paint_segments(self,show_noise=True):
        print "\t\t\t+++ painting", len(self.segments), "segments to previously drawn glyphs"
        scalars = np.hstack(self.segments[0].segments)
        if not scalars.shape[0] == self.x.shape[0]: raise ValueError("Shape mismatch")
        self.scalars = scalars
        
        #for nseg, seg in enumerate(self.segments):
            #begin,end = self.offsets[nseg], self.offsets[nseg+1]        
            #usable = seg.segments[nseg][:(end-begin)]
            #self.scalars[begin:end] = usable
        self.src.mlab_source.scalars = self.scalars
        cmap = self.src.module_manager.scalar_lut_manager.lut.table.to_array()
        if not show_noise:
            cmap[0,-1] = 0
        idx_fetcher = self.src.module_manager.scalar_lut_manager.lut.get_index
        for segment in self.segments:
            segment.mcolor = cmap[idx_fetcher(segment.segment_id)]

class dtkTrackDataset(TrackDataset):
    pass
