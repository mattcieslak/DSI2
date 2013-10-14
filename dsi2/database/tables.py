import pymongo

connection = pymongo.MongoClient('localhost', 27017)

db = connection.dsi2

subject_space = db.subject_space

def make_subject_document(subject_id, voxel, endpoints):
    eps = map(lambda e: map(int, list(e)), endpoints)
    return {
            'subject': subject_id,
            'voxel': map(int, list(voxel)),
            'endpoints': eps
            }

def make_endpoint_document(subject_id, track_id, endpoints):
    return {
            'subject': subject_id,
            'track_id': int(track_id),
            'endpoints': map(int, endpoints)
            }

def create_subject_space_from_trackset(trackset, subject_id, verbose=True):
    for k in trackset.tracks_at_ijk:
        endpoints = trackset.get_endpoints_by_ijks([k])
        doc = make_subject_document(subject_id, k, endpoints)
        #print doc['endpoints']
        if verbose:
            print 'INSERTING VOXEL %s...' % str(k)
        subject_space.insert(doc)
        if verbose:
            print '...DONE'

