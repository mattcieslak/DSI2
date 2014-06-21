import sys, os
from .traited_query import Scan
import json

import dsi2.config

dsi2_data = dsi2.config.dsi2_data_path
local_tdb_var = dsi2.config.local_trackdb_path

home_pkl  = os.path.join(os.getenv("HOME"),"local_trackdb")
if local_tdb_var:
    pkl_dir = local_tdb_var
    print "Using $LOCAL_TRACKDB environment variable",
elif os.path.exists(home_pkl):
    pkl_dir = home_pkl
    print "Using local_trackdb in home directory for data"
if dsi2_data:
    dsi2_data = dsi2_data
    print "Using $DSI2_DATA environment variable"
else:
    raise OSError("DSI2_DATA needs to be set")

# Load the json file describing available data
#data_json = os.path.join(pkl_dir,"data.json")
#fop = open(data_json,"r")
#jdata = json.load(fop)
#fop.close()


def get_local_data(json_file, pkl_dir=pkl_dir):
    fop = open(json_file,"r")
    jdata = json.load(fop)
    fop.close()
    datasets = [Scan(pkl_dir=pkl_dir, data_dir=dsi2_data, original_json=d, **d) for d in jdata ]
    print "  " + "=" * 50
    print "  DSI2"
    print "  ----"
    print "  Loaded %i Scans" % len(datasets)
    print "    from %s" % json_file
    print "  " + "=" * 50
    return datasets

# storage of all loadable local_data
#local_data = get_local_data(data_json)

# legacy
#local_trackdb = [ d for d in local_data if d.reconstruction == "gqi" ]
#local_qsdrdb  = [ d for d in local_data if d.reconstruction == "qsdr" ]
