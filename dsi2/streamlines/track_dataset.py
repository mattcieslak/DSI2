#!/usr/bin/env python
import warnings
from nibabel import trackvis
import numpy as np
from copy import deepcopy
from ..streamlines import track_math,qsdr_2mm_trk_header
from ..volumes.mask_dataset import MaskDataset
from collections import defaultdict
import cPickle as pickle
#from dipy.tracking.utils import streamline_mapping
from traitsui.api import View, Item, VGroup, HGroup, Group, \
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action, \
     CheckListEditor, ObjectColumn

from traits.api import HasTraits, Instance, Array, \
    CInt, Color, Bool, List, Int, Str, Instance, Any, Enum, \
    DelegatesTo, on_trait_change, Button,Str, Enum
import cPickle as pickle
from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi import mlab
import gzip

from .track_math import tracks_to_endpoints
from mayavi.core.api import PipelineBase, Source

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
    
    # Holds metadata
    coordinate_units = Enum("voxmm", "voxels", "world")
    orientation = Str("LPS")
    voxel_size = Array
    volume_shape = Array

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
            if fname.endswith("trk.gz") or fname.endswith("trk"):
                if fname.endswith("trk.gz"):
                    fl = gzip.open(fname,"r")
                elif fname.endswith("trk"):
                    fl = open(fname,"r")
                streams, self.header = trackvis.read(fl)
                # Convert voxmm to ijk
                self.set_tracks(np.asanyarray([stream[0] for stream in streams]))
                fl.close()
                # Check for scalars, support them someday
                if self.header['n_scalars'] > 0:
                    print "WARNING: Ignoring track scalars in %s"%fname
            elif fname.endswith("txt"):
                fop = open(fname,"r")
                self.set_tracks( np.asanyarray(
                    [np.array(map(float, line.strip().split())).reshape(-1,3) for \
                     line in fop ] ))
            elif fname.endswith("mat"):
                pass

        if properties is None:
            from dsi2.database.traited_query import Scan
            print "Warning: using default properties"
            self.properties = Scan()
        else:
            self.properties = properties
            
        # Configure the coordinate system
        if not hasattr(self,"header"):
            if self.voxel_size.size == 0 or self.volume_shape.size == 0:
                raise ValueError("Unable to determine streamline orientation")
            self.header = trackvis.empty_header()
            self.header['voxel_size'] = self.voxel_size
            self.header['dim'] = self.voxel_size
        else:
            # Change the header if voxel size or volume shape are requested
            if self.voxel_size.size > 0:
                warnings.warn("Overwriting header info with keyword arg voxel_size")
                self.header['voxel_size'] = self.voxel_size
            if self.volume_shape.size > 0:
                warnings.warn("Overwriting header info with keyword arg voxel_size")
                self.header['dim'] = self.voxel_size
        # Finally, sync the header and attrs
        try:
            self.voxel_size = self.header['voxel_size']
            self.volume_shape = self.header['dim']
        except Exception, e:
            warnings.warn("unable to load header")
            self.voxel_size = np.array([1,1,1])
            self.volume_shape = np.array([100,100,100])

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
                new_tracks = np.asanyarray(tracklist)
        elif type(tracklist) == np.ndarray:
            if not tracklist.size: return
            if tracklist.dtype in (object,np.float64,np.float16,np.float32):
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
        if not hasattr(self,"tracks_at_ijk"): raise ValueError("no voxel mapping available")
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


    def save(self,fname,use_mni_header=False,use_qsdr_header=False):
        """Save the object as a .trk file"""
        if use_mni_header:
            header = mni_hdr
        elif use_qsdr_header:
            header=qsdr_2mm_trk_header
        else:
            header= self.header
            
        trackvis.write(
            fname,
            ((stream,None,None) for stream in self),
            np.array(header)
        )
        


    #==================================
    # Functions from the former TraitedTrackDataset
    #==================================
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
        self.x,self.y,self.z = np.vstack(allpts).T
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
