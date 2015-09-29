#!/usr/bin/env python
import numpy as np
from ...streamlines.track_dataset import Segment
from .segmentation_ui import SegmentationEditor
from ...streamlines.track_math import streamlines_to_ijk

from traits.api import HasTraits, Instance, Array, \
    Bool, Dict, Range, Color, List, Int, Property, File, Button
from traitsui.api import Group,Item, RangeEditor


import tempfile
import subprocess
import os

class ReebGraphSegmentation(SegmentationEditor):
    
    parameters = ["epsilon", "delta", "voxel_size"]
    
    reeb_exe = File(os.path.join(os.getenv("HOME"),"ReebGen","reebgen"))
    min_tracks = Range( low=0, high=100, value=5, auto_set=False, name="min_tracks",
                          desc="A cluster label must be assigned to at least this many tracks",
                          label="Minimum tracks per Group"
                          )
    epsilon = Range( low=0.01, high=200.0,value=2.25, auto_set=False, name="epsilon",
                          desc="Distance threshold between cluster endpoint means",
                          label="Group Epsilon",
                          parameter=True
                          )
    delta = Range( low=0.0, high=1.0,value=0.33, auto_set=False, name="delta",
                          desc="how much mass you can lose and still stay the same bundle",
                          label="Group Delta",
                          parameter=True
                          )
    voxel_size = Range(low=0., high=5., value = 2., auto_set=False, name="voxel_size",
                       desc="How big should voxels be? If 0, then original coordinate streams are used",
                       parameter = True)
    volume_dimensions = Array(np.array([182, 218, 182]), shape=(3,), name="volume_dimensions",
                       desc="Shape of the voxel grid in mm",
                       parameter = True)
    show_noise = Bool(True, name="show_noise", 
                      desc="Render segments that have been classified as noise",
                      label="Render Noise Segs"
                      )
    shuffle_cmap = Button(label="Shuffle_cmap")
    
    # widgets for editing algorithm parameters
    algorithm_widgets = Group(
                         Item(name="reeb_exe"),
                         Item(name="voxel_size"),
                         Item(name="delta"),
                         Item(name="epsilon",
                              editor=RangeEditor(mode="slider", high = 200,low = 0,format = "%.2f")),
                         Item(name="min_tracks",
                              editor=RangeEditor(mode="slider", high = 100,low = 0,format = "%i")),
                         Item("show_noise"),Item("shuffle_cmap")
                              )

    
    # Segmentation parameters that can change but don't require reebgen to run again
    def _min_tracks_changed(self):
        print "Should update the colors based on ntracks"
        
    def _show_noise_changed(self):
        print "Rendering noise segments changed to", self.show_noise 
        if self.show_noise: 
            alpha = 255
        else:
            alpha = 0
        for tds in self.track_sets:
            if tds.tracks_drawn:
                cmap = tds.src.module_manager.scalar_lut_manager.lut.table.to_array()
                cmap[0,-1] = alpha
                tds.src.module_manager.scalar_lut_manager.lut.table = cmap
        
    def _shuffle_cmap_fired(self):
        print "Shuffling cmap"
        for tds in self.track_sets:
            if tds.tracks_drawn:
                cmap = tds.src.module_manager.scalar_lut_manager.lut.table.to_array()
                np.random.shuffle( cmap[1:] )
                tds.src.module_manager.scalar_lut_manager.lut.table = cmap
                
            
    
    def segment(self, ttracks):
        """
        """
        # Put streamlines into a format that reebgen can use
        print "\t\t\t+++Downsampling streamlines"
        tracks = ttracks.streamlines_to_ijk(
                            tracking_volume_voxel_size=np.array([self.voxel_size]*3),
                            tracking_volume_shape=self.volume_dimensions)
        ttracks.set_tracks(tracks)
        
        # Write the temporary track file
        otmp = tempfile.NamedTemporaryFile(delete=False)
        tmpname = otmp.name
        for trk in tracks:
            otmp.write(" ".join("%.4f"%f for _trk in trk for f in _trk ) + "\n")
        otmp.close()
        
        vmap = tempfile.NamedTemporaryFile(delete=False)
        vmap_name = vmap.name
        vmap.close()
        
        bmap = tempfile.NamedTemporaryFile(delete=False)
        bmap_name = bmap.name
        bmap.close()
        
        # Write the temporary dim file
        dimtmp = tempfile.NamedTemporaryFile(delete=False)
        geom_str = []
        geom_str.append(" ".join(["%.1f"%f for f in MNI_mm]))
        geom_str.append(" ".join(["%d"%f for f in MNI_mm/self.voxel_size]))
        dimtmp.write("\n".join(geom_str))
        dimtmp.close()
        print "geometry:", "\n".join(geom_str)
        
        # call the reebgen executable
        cmd_list =[self.reeb_exe, dimtmp.name, tmpname,"%.3f"%self.epsilon,
                          "%.2f"%self.delta, "%d"%self.min_tracks, 
                          "V", vmap_name, "B", bmap_name
                          ]
 
        print "COMMAND:", " ".join(cmd_list)
        proc = subprocess.Popen(cmd_list,
                          stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        sout, serr = proc.communicate()
        print sout
        
        # Process labels from stdout
        fop = open(bmap_name,"r")
        collected = []
        for line in fop.readlines():
            xx =np.array( map(int, line.strip().split()))
            xx[xx==-1] = -100
            collected.append(xx)
        labels = np.array(collected,dtype=np.object)
        fop.close()
        
        #TODO: Read the bmap file
        
        # Cleanup tmp files
        #os.system("rm " + otmp.name)
        #os.system("rm " + dimtmp.name)
        
        return labels
        
