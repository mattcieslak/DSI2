#!/usr/bin/env python
import numpy as np
# Traits stuff
from traits.api import HasTraits, Instance, Array, \
    CInt, Color, Bool, List, Int, Str, Instance, Any, Enum, DelegatesTo

from tvtk.pyface.scene import Scene
from tvtk.api import tvtk

from mayavi.core.ui.api import SceneEditor
from mayavi.core.api import Source
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi import mlab

from .track_math import tracks_to_endpoints

### Change this once the pics are made
LENGTH_FILTER=True

def random_color(*args):
    return (np.random.rand(),np.random.rand(),np.random.rand())

class Cluster(HasTraits):
    color       = Color
    mcolor      = Array(shape=(4,))
    visible     = Bool(True)
    render3d    = Bool(False)
    representation = Enum("Barbell","Splater")
    ntracks     = Int
    name        = Str
    id_number   = Int
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

class TraitedTrackDataset(HasTraits):
    # Holds original data: Never changes
    tracks = Array
    scalars = Array
    connections = Array
    # Are the glyphs already rendered?
    tracks_drawn = Bool(False)
    min_pts = Int(10)
    length_filter = Bool(False)
    # Holds the properties from dsi2.ui.browser_builder
    properties = Any
    name = Str("Tracks")
    # UI objects
    dynamic_color_clusters=DelegatesTo('properties')
    static_color = DelegatesTo('properties')
    color_map  = DelegatesTo('properties')
    #representation = Enum("Line","Tube")
    representation = Enum("Tube","Line")
    # How should the clusters be displayed?
    clusters = List(Instance(Cluster))
    cluster_representation = Enum("Barbell","Splatter")
    #cluster_representation = Enum("Splatter","Barbell")

    def __init__(self, trk_dset, **traits):
        """
        Holds a Traited version of ``dsi2.streamlines.track_dataset.TrackDataset``.
        Once initialized with a TrackDataset, a MayaVi representation is
        created automatically. When the cluster assignments are changed, the
        coloring of the MayaVi objects is updated

        """
        super(TraitedTrackDataset,self).__init__(**traits)
        # Pull properties from the parent TrackDataset
        self.properties = trk_dset.properties
        self.connections = trk_dset.connections
        self.clusters = trk_dset.clusters
        self.splatters = []
        
        if self.length_filter:
            self.tracks = np.array([trk for trk in trk_dset.tracks if trk.shape[0] > self.min_pts])
        else:
            self.tracks = trk_dset.tracks
        # Used to populate the colors
        self.offsets = np.cumsum(np.array([0] + [ trk.shape[0] for trk in self.tracks ]))

    def _name_default(self):
        return self.properties.scan_id

    def get_mayavi_kw(self):
        # Make arguments specific to the color settings
        if not self.dynamic_color_clusters:
            return {"color":(
                                    self.static_color.red/255.,
                                    self.static_color.green/255.,
                                    self.static_color.blue/255.)
            }
        if self.color_map == "random":
            return {"colormap":random_color}
        return {"colormap":self.color_map}

    def draw_tracks(self):
        connect = []
        allpts = []
        npts = 0
        for line in self.tracks:
            allpts.append(line)
            # Number of points in the line
            _npts = line.shape[0]
            conns = np.vstack(
                               [np.arange(npts, npts+_npts - 1.5),
                                np.arange(npts+1, npts+_npts -.5)]).T
            connect.append(conns)
            npts += _npts

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

        if self.representation == "Tube":
            self.src = mlab.pipeline.surface(
                            mlab.pipeline.tube(self.trk_lines, tube_radius=0.2),
                            #figure=scene.mayavi_scene, name="streamlines",
                            **mayavi_args)
        elif self.representation == "Line":
            self.src = mlab.pipeline.surface(self.trk_lines,line_width=2,
                                         #figure=scene.mayavi_scene, name="streamlines",
                                         **mayavi_args)
        #self.src.visible = self
        self.tracks_drawn = True

    def set_track_visibility(self, visibility):
        # Remove ster barbells
        for item in [self.src, self.trk_lines, self.pts]:
            try:
                item.visible = visibility
            except Exception,e :
                print "Failed to invisible", item, e


    def set_cluster_visibility(self, visibility):
        # Remove current cluster barbells
        if self.cluster_representation == "Barbell":
            for item in [self.cluster_glyphs, self.cluster_tubes, self.cluster_src]:
                try:
                    item.visible = visibility
                except Exception, e:
                    print item, "Failed to invisible", e
        elif self.cluster_representation == "Splatter":
            for item in self.splatters:
                item.volume.visibility = visibility

    def draw_clusters(self):
        # if there are no clusters, get out of here
        if len(self.clusters) == 0 or self.clusters[0].cluster_type == "REGION": return
        if self.cluster_representation == "Barbell":
            self.draw_cluster_barbells()
        elif self.cluster_representation == "Splatter":
            self.draw_cluster_splatter()

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
        if not self.tracks_drawn: self.draw_tracks()
        print "found", len(clusters), "clusters"
        self.scalars = np.zeros(len(self.x))
        # update the internal array of scalars
        for clust in self.clusters:
            for idx in clust.indices:
                try:
                    self.scalars[self.offsets[idx]:self.offsets[idx+1]] = \
                        clust.id_number
                except IndexError:
                    print idx, "out of bounds"
        self.src.mlab_source.scalars = self.scalars

        # Set the cluster colors
        if not self.dynamic_color_clusters:
            for clust in self.clusters:
                clust.color = self.static_color
                clust.name = self.name
        else:
            cmap = self.src.module_manager.scalar_lut_manager.lut.table.to_array()
            idx_fetcher = self.src.module_manager.scalar_lut_manager.lut.get_index
            for clust in self.clusters:
                clust.mcolor = cmap[idx_fetcher(clust.id_number)]
                clust.name = self.name

