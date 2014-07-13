from .clustering_algorithms import FastKMeansAggregator, QuickBundlesAggregator
from .region_labeled_clusters import RegionLabelAggregator
from traits.api import Int


def make_aggregator(algorithm = "", **kwargs):
    """
    A convenience function that returns an aggregator object
    The code for dealing with region labels is currently confusing
    so this will function provides an interface that will remain stable after
    things get cleaned up.
    
    
    """
    if algorithm == "region labels":
        if not "data_source" in kwargs:
            raise ValueError("region labels aggregator must have a " \
                             "data source to see which atlases are " \
                             "available")
        data_source = kwargs["data_source"]
        cl = RegionLabelAggregator()
        cl.set_track_source(data_source)
        assert "atlas_name" in kwargs, "Must provide an 'atlas_name'"
        atlas_name = kwargs['atlas_name']
        cl.atlas_name = atlas_name
        if "atlas_scale" in kwargs:
            cl.add_trait(atlas_name+"_scale",Int)
            setattr(cl,atlas_name + "_scale", kwargs['atlas_scale'])
        cl.update_atlas()
    elif algorithm == "k-means":
        cl = FastKMeansAggregator()
        if "k" in kwargs:
            cl.k = kwargs["k"]
    elif algorithm == "quickbundles":
        cl = QuickBundlesAggregator()
        if "dthr" in kwargs:
            cl.dthr = kwargs["dthr"]
    else:
        raise ValueError('algorithm must be one of "region labels",' \
                         '"quickbundles" or "k-means"')

    if "min_tracks" in kwargs:
        cl.min_tracks = kwargs["min_tracks"]
    return cl
