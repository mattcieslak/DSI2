import sys,os
import numpy as np
sys.path.append("..")
from dsi2.database.local_data import get_local_data
test_input_data = "/home/dsi2/DSI2/testing_data/testing_input"
test_output_data = "/home/dsi2/DSI2/testing_data/testing_output"
from scipy.stats import itemfreq
import cPickle as pickle

import pymongo
from bson.binary import Binary
connection = pymongo.MongoClient()
db = connection.dsi2

# Ensure there is an index for fast querying
# TODO: How should the compound indexes be ordered? What is most efficient?

db.Lausanne2008scale33.ensure_index([("ijk",pymongo.ASCENDING), ("scan_id",pymongo.ASCENDING)])
db.streamlines.ensure_index([("scan_id",pymongo.ASCENDING),("sl_id",pymongo.ASCENDING)])
db.coordinates.ensure_index([("scan_id",pymongo.ASCENDING),("ijk",pymongo.ASCENDING)])

db.connections.ensure_index([("scan_id",pymongo.ASCENDING),("ijk",pymongo.ASCENDING)])
db.connections2.ensure_index([("con_id",pymongo.ASCENDING),("scan_id",pymongo.ASCENDING)])

db.scans.ensure_index([("scan_id",pymongo.ASCENDING),("subject_id",pymongo.ASCENDING)])

local_scans = get_local_data(test_output_data + "/example_data.json")

for sc in local_scans:
    # load the TrackDataset's one at a time
    trackds = sc.get_track_dataset()
    atlas_labels = trackds.properties.track_label_items[0].load_array(test_output_data)
    trackds.set_connections(atlas_labels)

    inserts = []
    con_ids = set(atlas_labels)
    print "Building alternate connections collection..."
    for con_id in con_ids:
        sl_ids = list(map(int,np.where(atlas_labels == con_id)[0]))
        inserts = []
        inserts.append(
                {
                    "con_id":"%d" % con_id,
                    "scan_id":sc.scan_id,
                    "sl_ids":sl_ids
                }
                )

    db.connections2.insert(inserts)

    print "done."


    print "Building connections collection..."
    # insert a document for each coordinate
    total_n = len(trackds.tracks_at_ijk.keys())
    n=0
    print "total", total_n
    inserts = []
    for coord,indices in trackds.tracks_at_ijk.iteritems():
        connections = trackds.connections[np.array(list(indices))]
        freqs = itemfreq(connections)
        inserts.append(
            {
             "ijk":"(%d, %d, %d)" % tuple(map(int,coord)),
             "scan_id":sc.scan_id,
             "con":[{"con":"c%d"%k, "count":int(v)} for k,v in freqs]
            }
          )
        n += 1
    db.connections.insert(inserts)
    print "done."


    print "Building streamline collection..."
    inserts = []
    for ntrk, trk in enumerate(trackds.tracks):
        if n % 1000 == 0:
            db.streamlines.insert(inserts)
            inserts = []
        inserts.append(
            {
              "scan_id":sc.scan_id,
              "sl_id": ntrk,
              "data":Binary(pickle.dumps(trk,protocol=2))
             }
        )
        n += 1
    db.streamlines.insert(inserts)
    print "done."

    inserts = []
    print "Building coordinate collection..."
    for coord,indices in trackds.tracks_at_ijk.iteritems():
        inserts.append(
           {
            "ijk":"(%d, %d, %d)" % tuple(map(int,coord)),
            "scan_id":sc.scan_id,
            "sl_id":list(map(int,indices))
           }
          )
    db.coordinates.insert(inserts)
    print "done."

    # TODO: add more scan metadata
    inserts = []
    print "Building scan collection..."
    inserts.append(
            {
                "scan_id":sc.scan_id,
                "subject_id":sc.subject_id,
                "sls":len(trackds.tracks),
                "header":Binary(pickle.dumps(trackds.header,protocol=2)),
            }
            )
    db.scans.insert(inserts)
    print "done."

