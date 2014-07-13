#!/usr/bin/env python
import time, sys, os
from dsi2.database.mongo_track_datasource import MongoTrackDataSource
from dsi2.database.track_datasource import TrackDataSource
from dsi2.ltpa import mni_white_matter, run_ltpa
import numpy as np
from time import time

# How many coordinates should you search over?
# If this is an integer N, then N random coordinates in white matter
# are chosen and searched.  If a string, then all white matter
# coordinates are searched
NCOORDS = 900

# If you use more than 1 process, specify it here. Before running with
# more N_PROCS > 1, be sure to start ipcluster beforehand with a
# command like
# $ ipcluster start --n=N_PROCS
N_PROCS = 1

# Specify if you want to use mongodb or pickle files on disk 
SOURCE_TYPE = "mongodb" # "mongodb" or "local"
# if "local", specify the path to the json file
JSON_FILE = "/path/to/json/file.json"

# How big should the search sphere be (in voxels)?
RADIUS = 1

# all the scans we have available
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

# If you want to only do a subset of these, select them here.
# Otherwise, set test_subject=all_subjects
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
    from dsi2.database.local_data import get_local_data
    ok_datasets = [tds.get_track_dataset() for tds in \
            get_local_data(JSON_FILE) if tds.properties.scan_id in test_subjects]
    data_source = TrackDataSource(track_datasets=ok_datasets)

# specify the coordinates to search
if type(NCOORDS) == int:
    np.random.shuffle(mni_white_matter)
    coords = mni_white_matter[:NCOORDS]
else:
    coords = mni_white_matter

"""
Create a dictionary containing all the information necessary to construct
an aggregator for your analysis function. It works this way instead of 
by directly creating an Aggregator object because run_ltpa constructs a new
Aggregator inside each independent process.

The dictionary must contain at least the key "algorithm", which can be one of 
{ "region labels", "k-means", "quickbundles"}. The rest of the keys are sent
as keyword arguments to dsi2.aggregation.make_aggregator. 

When run_ltpa is looping over coordinates, result of a apatial query is sent
to an instance of the aggregator.  The aggregator's ``aggregate()`` method 
is called for each TrackDataset returned from the query, then the aggregator
is sent to whichever function you provided to run_ltpa.

NOTE: If you select the "region labels" aggregator, then you won't have access
to the streamline objects. To access streamlines, choose "k-means" or 
"quickbundles".
"""
agg_args = {
            "algorithm":"region labels",
            "atlas_name":"Lausanne2008",
            "atlas_scale":60,
            "data_source":data_source
            }

def get_n_streamlines(aggregator):
    """
    This function should be replaced with a function that accepts a single argument,
    does something, then returns the results you care about.
    
    This particular function calculates the mean number of streamlines observed
    in each subject and returns this value and its standard deviation across all
    subjects.  We used this to calculate how many streamlines pass through each voxel
    then compared this number to how many real axons are known to pass through a voxel
    (based on electron microscopy).
    
    NOTE: you can access streamlines directly by the aggregator's ``track_sets``
    attribute, which is a list of TrackDataset objects.  Each will have a ``.tracks``
    attribute containing the numpy array of streamline coordinates.  Again, in this
    case ``.tracks`` will be empty because we are using a region label aggregator.
    """
    conn_ids, cvec_mat = aggregator.connection_vector_matrix()
    # The "region labels" aggregator has a ``connection_vector_matrix()``
    # function, which returns a list of all connections observed going through
    # the query coordinates (``conn_ids``) and a matrix where each row is a 
    # subject and column is ea connection. 
    sl_counts = cvec_mat.sum(1)
    # Sums across all connections for each subject
    return sl_counts.mean(), sl_counts.std()


t0 = time()
# This is the key function for interacting with DSI2: run_ltpa
# You give it a function, data source (either mongodb or local files) 
# a dictionary of aggregator constructor parameters, then sprcify how 
# to conduct the sphere search (radius, how many processes to use, 
# which coordinates to search)
#
# All the results from your function will be stored in ``results``
# corresponding to the search centers in ``coords``
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