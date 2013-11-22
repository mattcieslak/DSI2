import sys,os
import numpy as np
sys.path.append("..")
from dsi2.database.local_data import get_local_data
test_output_data = os.getenv("TEST_OUTPUT_DATA")
test_input_data = "/home/cieslak/testing_data/testing_input"
test_output_data = "/home/cieslak/testing_data/testing_output"
from scipy.stats import itemfreq
import cPickle as pickle

import pymongo
from bson.binary import Binary
connection = pymongo.MongoClient()
db = connection.dsi2
# Ensure there is an index for fast querying
db.Lausanne2008scale33.ensure_index([("ijk",pymongo.ASCENDING), ("scan_id",pymongo.ASCENDING)])
db.streamlines.ensure_index([("scan_id",pymongo.ASCENDING),("sl_id",pymongo.ASCENDING)])
db.connections.ensure_index([("scan_id",pymongo.ASCENDING),("ijk",pymongo.ASCENDING)])

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
    inserts = []
    for coord,indices in trackds.tracks_at_ijk.iteritems():
        if n % 100 == 0:
            print n
        connections = trackds.connections[np.array(list(indices))]
        freqs = itemfreq(connections)
        inserts.append(
            {
             "ijk":"%d_%d_%d" % tuple(map(int,coord)),
             "scan_id":sc.scan_id,
             "con":[{"con":"c%d"%k, "count":int(v)} for k,v in freqs]
            }
          )
        n += 1
    print "actually inserting"
    db.try2.insert(inserts)
    print "done."

    continue
    inserts = []
    for ntrk, trk in enumerate(trackds.tracks):
        if n % 1000 == 0:
            db.streamlines.insert(inserts)
            inserts = []
            print n
        inserts.append(
            {
              "scan_id":sc.scan_id,
              "sl_id": ntrk,
              "data":Binary(pickle.dumps(trk,protocol=2))
             }
        )
        n += 1
    db.streamlines.insert(inserts)

    n=0
    inserts = []
    print "connectiondb", total_n
    for coord,indices in trackds.tracks_at_ijk.iteritems():
        if n % 100 == 0:
            print n
        connections = trackds.connections[np.array(list(indices))]
        freqs = itemfreq(connections)
        inserts.append(
           {
            "ijk":"%d_%d_%d" % tuple(map(int,coord)),
            "scan_id":sc.scan_id,
            "sl_id":list(map(int,indices))
           }
          )
        n += 1
    print "actually inserting"
    db.connections.insert(inserts)
    print "done"
