import sys, os
from .traited_query import Scan
import json
from pkg_resources import Requirement, resource_filename

import dsi2.config

dsi2_data = resource_filename(
                   Requirement.parse("dsi2"),
                   "example_data")

local_tdb_var = dsi2.config.local_trackdb_path

home_pkl  = os.path.join(os.getenv("HOME"),"local_trackdb")
if local_tdb_var:
    pkl_dir = local_tdb_var
    print "Using $LOCAL_TRACKDB environment variable",
elif os.path.exists(home_pkl):
    pkl_dir = home_pkl
    print "Using local_trackdb in home directory for data"


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