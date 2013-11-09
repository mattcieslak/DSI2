#!/usr/bin/env python
import time, sys, os
sys.path.append("..")
from dsi2.aggregation.region_clusters import RegionAggregator
from dsi2.aggregation.clustering_algorithms import QuickBundlesAggregator,FastKMeansAggregator
from dsi2.database.track_datasource import TrackDataSource
from dsi2.streamlines.track_math import sphere_around_ijk
from dsi2.volumes.mask_dataset import MaskDataset

from dsi2.ui.sphere_browser import SphereBrowser

# Test an atlas or aggregation algorithm?
#test = "regions"
test = "clustering"

# If aggregation, which algorithm?
#aggregator = "dipy"
aggregator = "kmeans"

# Want to load multiple subjects?
#TEST_MULTISUBJECTS = True
TEST_MULTISUBJECTS = False

from dsi2.database.local_data import get_local_data
from dsi2.database.local_data import pkl_dir
os.environ["LOCAL_TRACKDB"] = "/home/cieslak/example_trackdb"
local_trackdb = get_local_data("/home/cieslak/example_trackdb/example_data.json")

# how many subjects?
if TEST_MULTISUBJECTS:
    track_source = TrackDataSource(track_datasets = [ ltb.get_track_dataset() for ltb in local_trackdb ])
else:
    track_source = TrackDataSource(track_datasets = [local_trackdb[0].get_track_dataset()])

if test == "regions":
    cl = RegionAggregator()
elif test == "clustering":
    if aggregator == "dipy":
        cl = QuickBundlesAggregator()
    elif aggregator=="kmeans":
        cl = FastKMeansAggregator()

# construct the actual browser
sb = SphereBrowser(track_source = track_source, aggregator=cl)
sb.configure_traits()
