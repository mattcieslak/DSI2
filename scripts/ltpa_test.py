#!/usr/bin/env python
import time, sys, os
from dsi2.database.mongo_track_datasource import MongoTrackDataSource
from dsi2.ltpa import mni_white_matter, run_ltpa
import numpy as np
from time import time

NCOORDS = "all"
N_PROCS = 25
SOURCE_TYPE = "mongodb"
RADIUS = 1

# scans to use for testing
all_subjects = [ "1005y", "1410c", "2843B", "2843C", "2843A", 
                 "0002a", "4078B", "4078C", "4078A", "3735r", 
                 "1505l", "4673A", "3309A", "3309B", "3309C", 
                 "0305p", "2986m", "3075b", "3213t", "1656q", 
                 "1130w", "0009z", "1656A", "1656C", "1656B", 
                 "3906c", "0701j", "0377C", "1291r", "1935x", 
                 "3057t", "0004a", "2.1369B", "2.1369C", "3164q", 
                 "2.1369A", "4827r", "1169j", "0674k", # "1694e", 
                 "4893d", "0682c", "1188q", "2121t", "0003a", 
                 "4458w", "0815z", "3503p", "4108i", "2.3527B", 
                 "0819v", "0105A", "0105C", "0105B", "3596a", 
                 "1319p", "3808w", "2895z", "0731f", "2.1394B", 
                 "2.1394C", "2.1394A", "0714p", "2888g", "3708s", 
                 "2.3527A", "2.3527C", "1347n", "1220A", "1220B", 
                 "1220C", "0282m", "0040u", "2066w", "1037B", 
                 "1037C", "1037A", "1943p", "3640i", "0377B", 
                 "3997p", "3987z", "0377A", "2268c", "2318e", 
                 "2664w", "0437n", "1415x", "3444h" ]
test_subjects = all_subjects

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
if type(NCOORDS) == int:
    coords = mni_white_matter[:NCOORDS]
else:
    coords = mni_white_matter

# A function for processing the results of the aggregator
def get_cvec_shape(aggregator):
    conn_ids, cvec_mat = aggregator.connection_vector_matrix()
    return cvec_mat.shape

def get_n_streamlines(aggregator):
    conn_ids, cvec_mat = aggregator.connection_vector_matrix()
    sl_counts = cvec_mat.sum(1)
    return sl_counts.mean(), sl_counts.std()

# Info necessary to create the aggregator
agg_args = {
            "algorithm":"region labels",
            "atlas_name":"Lausanne2008",
            "atlas_scale":60,
            "data_source":data_source
            }

t0 = time()
results = run_ltpa(get_n_streamlines, data_source=data_source,
                aggregator_args=agg_args, radius = RADIUS,
                n_procs=N_PROCS, search_centers=coords)                
t1 = time()
runtime = t1-t0
if type(NCOORDS) == int:
    whole_brain_estimate = float(mni_white_matter.shape[0]) / NCOORDS * runtime
else:
    whole_brain_estimate = runtime

print "LTPA took %.2f seconds. It would take approx %.2f to run whole-brain" %(
         runtime, whole_brain_estimate )