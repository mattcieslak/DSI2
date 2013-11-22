#!/usr/bin/env python
import sys
sys.path.append("..")
import os
test_input_data = os.getenv("TEST_INPUT_DATA")
test_output_data = os.getenv("TEST_OUTPUT_DATA")
test_input_data = "/home/cieslak/testing_data/testing_input"
test_output_data = "/home/cieslak/testing_data/testing_output"
import nibabel as nib
import numpy as np
import cPickle as pickle
from dsi2.database.local_data import get_local_data
from dsi2.streamlines.track_dataset import TrackDataset
from dsi2.aggregation.clustering_algorithms import FastKMeansAggregator, QuickBundlesAggregator
from dsi2.streamlines.track_math import sphere_around_ijk



import pymongo
from bson.binary import Binary
connection = pymongo.MongoClient()
db = connection.dsi2

test_coordinates = sphere_around_ijk(3,(33,54,45))
test_scan_ids = ["0377A","2843A"]

sl_ids = db.connections.aggregate(
    # Find the coordinates for each subject
    [
        {"$match":{
            "scan_id":{"$in":test_scan_ids},
            "ijk":{"$in":
                   ["%d_%d_%d" % tuple(map(int,coord)) for coord in test_coordinates]}
            }},
        {"$project":{"scan_id":1,"sl_id":1}},
        {"$unwind":"$sl_id"},
        {"$group":{"_id":"$scan_id", "sl_ids":{"$addToSet":"$sl_id"}}}
    ]
)["result"]


sl_data = db.streamlines.find(
    {
     "scan_id":sl_ids[0]["_id"],
     "sl_id":{"$in":sl_ids[0]["sl_ids"]}
    }
)
streamlines = [pickle.loads(d['data']) for d in sl_data]
scans = get_local_data(os.path.join(test_output_data,"example_data.json"))
toy_dataset = TrackDataset(header={"n_scalars":0}, streams = (),properties=scans[0])
toy_dataset.tracks = np.array(streamlines,dtype=object)
toy_dataset.render_tracks = True
toy_dataset.draw_tracks()

from bson.code import Code

mapper = Code(
    """
    function() {
      for ( var key in this.con) {
         emit(key, this.con[key]);
         }
    }
    """
)

reducer = Code(
    """
    function(keyConId,valuesCounts) {
        return Array.sum(valuesCounts);
    }
    """
)



def query_lausanne(coordinates):
    results = db.Lausanne2008scale33.map_reduce(
        mapper, reducer, "my_results", query = \
            {"scan_id":{"$in":test_scan_ids},
            "ijk":{"$in":
                   ["%d_%d_%d" % tuple(map(int,coord)) for coord in test_coordinates]
                   }
        }, )

def aggegate_lausanne(coordinates):
    results = db.try2.aggregate([
        {"$match":{
            "scan_id":{"$in":test_scan_ids},
            "ijk":{"$in":
                   ["%d_%d_%d" % tuple(map(int,coord)) for coord in test_coordinates]
                   }
            }},
        {"$unwind":"$con"},
        {"$group":{"_id":{"scan_id":"$scan_id",
                          "con_id":"$con.con"},
                   "counts":{"$sum":"$con.count"}}
         }
    ])

def aggegate_lausanne(coordinates):
    results = db.try2.aggregate([
        {"$match":{
            "scan_id":{"$in":test_scan_ids},
            "ijk":{"$in":
                   ["%d_%d_%d" % tuple(map(int,coord)) for coord in test_coordinates]
                   }
            }},
        {"$unwind":"$con"},
        {"$group":{"_id":{"scan_id":"$scan_id",
                          "con_id":"$con.con"},
                   "counts":{"$sum":"$con.count"}}
         }

    ])
