#!/usr/bin/env python
import sys,os

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
