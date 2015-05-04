#!/usr/bin/env python
import sys,os
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'

def browser_builder():
    from dsi2.ui.browser_builder import BrowserBuilder
    if len(sys.argv) > 1:
        if not os.path.exists(sys.argv[1]):
            print sys.argv[1], "FILE DOES NOT EXIST"
            sys.exit(1)
        bb = BrowserBuilder(local_json=sys.argv[1])
    else:
        bb=BrowserBuilder()
        
    bb.configure_traits()
    
def import_data():
    from dsi2.ui.local_data_importer import LocalDataImporter
    if len(sys.argv) > 1:
        if not os.path.exists(sys.argv[1]):
            print sys.argv[1], "FILE DOES NOT EXIST"
            sys.exit(1)
        ldi = LocalDataImporter(json_file=sys.argv[1])
    else:
        ldi = LocalDataImporter()
    
    ldi.configure_traits()
    
    
    
    
def view_tracks():
    from dsi2.streamlines.track_dataset import TrackDataset
    track_sets = []
    if len(sys.argv) > 1:
        for pth in sys.argv[1:]:
            try:
                track_sets.append(
                    TrackDataset(pth)
                    )
            except Exception,e:
                print "unable to load", pth
                print e
    from dsi2.ui.sphere_browser import SphereBrowser
    sb = SphereBrowser()
    sb.aggregator.set_track_sets(track_sets)
    sb.aggregator.render_tracks = True
    sb.configure_traits()
    
    
