import sys,os
import numpy as np
sys.path.append("..")
from dsi2.database.local_data import get_local_data
test_output_data = os.getenv("TEST_OUTPUT_DATA")
test_input_data = "/home/cieslak/testing_data/testing_input"
test_output_data = "/home/cieslak/testing_data/testing_output"
from scipy.stats import itemfreq

import pymongo
connection = pymongo.MongoClient()
db = connection.dsi2

local_scans = get_local_data(test_output_data + "/example_data.json")


for sc in local_scans:
    # load the TrackDataset's one at a time
    trackds = sc.get_track_dataset()
    atlas_labels = trackds.properties.track_label_items[0].load_array(test_output_data)
    trackds.set_connections(atlas_labels)
    # insert a document for each coordinate
    total_n = len(trackds.tracks_at_ijk.keys())
    n=0
    print "total", total_n
    for coord,indices in trackds.tracks_at_ijk.iteritems():
        if n % 100 == 0:
            print n
        connections = trackds.connections[np.array(list(indices))]
        freqs = itemfreq(connections)
        db.Lausanne2008scale33.update(
            {"ijk":"%d_%d_%d" % tuple(map(int,coord))},
            {u"$set":{sc.scan_id:
                      dict([("c%d"%k, int(v)) for k,v in freqs])
                      }
            },
            True
          )
        n += 1