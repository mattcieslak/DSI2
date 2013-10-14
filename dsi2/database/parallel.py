import pymongo
from query import sphere_around_ijk
from query import MaskDataset, group_mask
import multiprocessing as mp
import numpy as np
import traceback
from collections import defaultdict

def query_voxel_subset(connection, voxels):
    q = {'voxel': {'$in': voxels}}
    #print q
    db = connection.dsi2
    subject_space = db.subject_space
    results = subject_space.find(q, {'subject': 1, 'endpoints': 1})

    def defaultgen():
        return defaultdict(int)

    subject_map = defaultdict(defaultgen)
    for result in results:
        subject_id = result['subject']
        for endpoint in result['endpoints']:
            subject_map[subject_id][tuple(endpoint)] += 1

    return subject_map

def compute_index_block(block, block_num):
    try:
        connection = pymongo.MongoClient('localhost', 27017)
        db = connection.dsi2
        subject_space = db.subject_space
        block_len = len(block)
        #print "blocklen",block_len
        
        results = []
        for i,coord in enumerate(block):
            sphere = sphere_around_ijk(coord, 4)
            subj_map = query_voxel_subset(connection, sphere)
            results.append(subj_map)
            if i % 1000 == 0:
                print '%d points of block %d done' % (i, block_num)

    except:
        traceback.print_exc()
        raise
    return results


class VoxelPipeline(object):
    def __init__(self, coords):
        self.coords = coords
        self.num_procs = mp.cpu_count() - 1
        self.pool = mp.Pool(processes=self.num_procs)
        self.results = []

    def collect_result(self, r):
        self.results.append(r)

    def run(self):
        #for coord in self.coords:
        index_blocks = np.array_split(np.arange(len(self.coords)), 
                    self.num_procs)
        for i,block in enumerate(index_blocks):
            self.pool.apply_async(compute_index_block, 
			        args=[self.coords[block[0]:block[-1]], i],
                    callback=self.collect_result)

        self.pool.close()
        self.pool.join()

if __name__ == '__main__':
    gmask = MaskDataset(group_mask)
    indices = gmask.in_mask_voxel_ijk
    print 'Processing %d indices' % len(indices)
    vx = VoxelPipeline(indices)
    import time
    t0 = time.time()
    vx.run()
    t1 = time.time()
    print 'time:', t1-t0

