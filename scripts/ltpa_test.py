#!/usr/bin/env python
import time, sys, os
from dsi2.database.mongo_track_datasource import MongoTrackDataSource
from dsi2.ltpa import mni_white_matter, run_ltpa
import numpy as np

NCOORDS = 200
N_PROCS = 1
SOURCE_TYPE = "mongodb"
RADIUS = 2

# scans to use for testing
test_subjects = [ "1005y", "1410c", "2843B", "2843C", "2843A" ]

# Create a data_source
if SOURCE_TYPE == "mongodb":
    data_source = MongoTrackDataSource(
        scan_ids = test_subjects,
        mongo_host = "127.0.0.1",
        mongo_port = 27017,
        db_name="dsi2"
    )
elif SOURCE_TYPE == "local":
    # Make this later        
    assert 0

# specify the coordinates to search
np.random.shuffle(mni_white_matter)
coords = mni_white_matter[:NCOORDS]

# A function for processing the results of the aggregator
def get_cvec_shape(aggregator):
    conn_ids, cvec_mat = aggregator.connection_vector_matrix()
    return cvec_mat.shape

# Info necessary to create the aggregator
agg_args = {
            "algorithm":"region labels",
            "atlas_name":"Lausanne2008",
            "atlas_scale":60,
            "data_source":data_source
            }

results = run_ltpa(get_cvec_shape, data_source=data_source,
                aggregator_args=agg_args, radius = RADIUS,
                n_procs=N_PROCS, search_centers=coords)
