import os, logging
import numpy as np
import multiprocessing
from functools import partial

from dsi2.streamlines.track_math import sphere_around_ijk
from dsi2.volumes.mask_dataset import MaskDataset
from dsi2.database.mongo_track_datasource import MongoTrackDataSource
from dsi2.aggregation import make_aggregator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load the standard MNI white matter mask and get its coordinates
wm_mask = MaskDataset(
    os.path.join(os.getenv("DSI2_DATA"),"MNI152_T1_2mm_brain_mask.nii.gz")
    )
mni_white_matter = wm_mask.in_mask_voxel_ijk

    

def run_ltpa( function, data_source, aggregator_args, 
              radius=0, n_procs=1, search_centers=mni_white_matter ):
    """
    Performs a LTPA over a set of voxels
    
    Parameters:
    -----------
    function:function
      A function that takes an aggregator as its only argument and
      returns all the information you're interested in getting
      from each search sphere
    
    data_source:dsi2.database.data_source.DataSource
      A DataSource instance from which to query spatial coordinates
      
    aggregator:dict
      Arguments to be passed to ``dsi2.aggregation.make_aggregator``
      
    radius:int
      Radius (in voxels) that should be included around each coordinate
      in ``search_centers``. Default: 1
      
    n_procs:int
      split ``search_centers`` into ``n_procs`` pieces and process each 
      piece on a different processor. Default is 1, must be less than or
      equal to the number of processors on the local machine.
      
    search_centers:np.ndarray
      n x 3 matrix with the coordinates to serve as centers for the 
      LTPA search.
        
    Returns:
    --------
    results:list
      
    """
    def process_centers(centers, func = None, data_source = None, 
                        aggregator_args = None, radius = None):
        if None in (func, data_source, aggregator_args, radius):
            raise ValueError("Must specify all arguments")
        # Create a fresh aggregator for this process
        aggregator = make_aggregator(**aggregator_args)
        # Give this data source a fresh connection to mongodb if that's 
        # where the data's coming from.
        if type(data_source) == MongoTrackDataSource:
            data_source.new_mongo_connection()
        
        fetch_streamlines = aggregator_args["algorithm"] != "region labels"
        results = []
        for cnum, center_index in enumerate(centers):
            # Which coordinates to search?        
            if radius > 0:
                coords = sphere_around_ijk(radius,center_index)
            else:
                coords = center_index
            # Send the result to the aggregator and process the data
            track_sets = data_source.query_ijk(coords,
                                    fetch_streamlines=fetch_streamlines)
            aggregator.set_track_sets(track_sets)
            aggregator.update_clusters()
            # Send the freshly updated aggregator to the evaluator function
            try:
                result = func(aggregator)
            except Exception, e:
                result = e
            results.append(result)
        return results
    
    # Process normally on this cpu
    if n_procs == 1:
        results = process_centers(search_centers, function,  
                                  data_source, aggregator_args, radius)
        return results

    # Using multiple processors
    available_processors = multiprocessing.cpu_count()
    
    # Check the cpu arguments against the system
    if n_procs > available_processors:
        raise ValueError("Only have %d cpu's available, %d requested" %(
            available_processors, n_procs))
    
    # Split the centers for the different processes
    center_chunks = np.array_split(centers, n_procs)
    logger.info("splitting %d coordinates into %d chunks",len(coords), n_procs)
    
    # Create a pool of workers
    try:
        pool = multiprocessing.Pool(n_procs)
    except Exception, e:
        logger.critical("Unable to create a processing pool: %s, e")

    # run the processes in the pool
    partial_process_centers = partial(process_centers, func=function,
            data_source=data_source, aggregator_args=aggregator_args, 
            radius = radius)
    
    results = pool.map(partial_process_centers, center_chunks)
    
    pool.close()
    
    return results
    
    
    

