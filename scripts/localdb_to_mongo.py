import sys,os
import numpy as np
from dsi2.database.local_data import get_local_data
import cPickle as pickle
import pymongo
from bson.binary import Binary
import pdb
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


# Ensure there is an index for fast querying
# TODO: How should the compound indexes be ordered? What is most efficient?

def init_db(db):
    """
    adds collections and makes sure that indexes work for them
    """
    db.streamlines.ensure_index([("scan_id",pymongo.ASCENDING),
        ("sl_id",pymongo.ASCENDING)])
    db.coordinates.ensure_index([("scan_id",pymongo.ASCENDING),
        ("ijk",pymongo.ASCENDING)])
    db.connections.ensure_index([("con_id",pymongo.ASCENDING),
        ("scan_id",pymongo.ASCENDING),("atlas_id",pymongo.ASCENDING)])
    db.streamline_labels.ensure_index([("scan_id",pymongo.ASCENDING),
        ("atlas_id",pymongo.ASCENDING)])
    db.atlases.ensure_index([("name",pymongo.ASCENDING)])
    db.scans.ensure_index([("scan_id",pymongo.ASCENDING),("subject_id",pymongo.ASCENDING)])

def check_scan_for_files(sc):
    """Checks to make sure that all the necessary files for this scan are on disk.
    If they are, it returns True, otherwise False
    """
    pkl_file = os.path.join(sc.pkl_dir,sc.pkl_path)
    if not os.path.exists(pkl_file):
        print "Unable to locate pickle file %s" % pkl_file
        logging.error("Unable to locate pickle file %s", pkl_file)
        return False

    # Check that all the npy files exist
    for label in sc.track_label_items:
        npy_path = os.path.join(label.base_dir, label.numpy_path)
        if not os.path.exists(npy_path):
            print "unable to load %s" % npy_path
            logging.error("unable to load %s" % npy_path)
            return False
    return True




def upload_atlases(trackds,sc):
    """
    Reads the atlas info from a Scan and loads the npy files from disk. Then

    1) uploads the atlas info into db.atlases
    2) uploads the label array for each atlas/scan into db.streamline_labels 
    3) uploads the streamline ids for each connection in each atlas/scan into connections

    """
    try:
        atlases = []
        logging.info("processing %d atlases for %s", len(trackds.properties.track_label_items), sc.scan_id)
        for label in trackds.properties.track_label_items:
            #pdb.set_trace()
            atlas_labels = label.load_array()

            # Does this atlas already exist? If not, add it to the collection.
            atlas = None
            result = db.atlases.find( { "name": label.name, "parameters": label.parameters } )
            if result.count() != 0:
                atlas = result[0]["_id"]
            else:
                atlas = db.atlases.insert( { "name": label.name, "parameters": label.parameters } )

            atlases.append(atlas)
            db.streamline_labels.insert([
                    {
                        "scan_id": sc.scan_id,
                        "atlas_id": atlas,
                        "con_ids": list(map(int,atlas_labels))
                    }
            ])

            # -------------------------------------------
            # Insert data into the connections collection
            # -------------------------------------------
            inserts = []
            con_ids = set(atlas_labels)
            print "Building connections collection for %s %d..." % (label.name, label.parameters["scale"])
            for con_id in con_ids:
                sl_ids = list(map(int,np.where(atlas_labels == con_id)[0]))
                inserts.append(
                        {
                            "con_id":"%d" % con_id,
                            "scan_id":sc.scan_id,
                            "atlas_id":atlas,
                            "sl_ids":sl_ids
                        }
                )
            db.connections.insert(inserts)
            print "done."
    except Exception, e:
        print "Failed to upload atlases", e
        return False
    return True


def upload_streamlines(trackds, sc):
    """
    Inserts the binary streamline data into db.streamlines
    """
    try:
        logging.info("Building streamline collection")
        inserts = []
        for ntrk, trk in enumerate(trackds.tracks):
            # continue appending to inserts until it gets too big
            if len(inserts) >= 1000:
                # then insert it and clear inserts
                db.streamlines.insert(inserts)
                inserts = []
            inserts.append(
                {
                  "scan_id":sc.scan_id,
                  "sl_id": ntrk,
                  "data":Binary(pickle.dumps(trk,protocol=2))
                 }
            )
        # Finally, insert the leftovers
        db.streamlines.insert(inserts)
    except Exception, e:
        print "Failed to upload streamline data", e
        return False
    return True


def upload_coordinate_info(trackds, sc):
    try:
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
    except Exception, e:
        print "Failed to upload coordinate info", e
        return False

    return True

def upload_scan_info(sc):
    try:
        db.scans.insert([sc.original_json])
    except Exception, e:
        print "Failed to upload scan info", e
        return False
    return True


def upload_local_scan(sc):
    logging.info("uploading %s", sc.scan_id)
    try:
        trackds = sc.get_track_dataset()
    except:
        print "failed to read pkl file"
        return False, "pkl file corrupt"

    if not upload_atlases(trackds, sc):
        print "failed to upload atlases"
        return False, "upload_atlases"

    if not upload_streamlines(trackds, sc):
        print "failed to upload streamlines"
        return False, "upload_streamlines"

    if not upload_coordinate_info(trackds, sc):
        print "failed to upload spatial mapping"
        return False, "upload_coordinate_info"

    if not upload_scan_info(sc):
        print "failed to upload spatial mapping"
        return False, "upload scan info"

    return True, "hooray!"

# Make sure that mongod is running.
# numactl --interleave=all mongod --dbpath /extra/cieslak/dsi2_db --bind_ip 127.0.0.1
connection = pymongo.MongoClient()
db = connection.dsi2
# Add collections and indexes to the mongodb instance
init_db(db)

# Path to the json file containing all the qsdr
qsdr_json = "/storage2/cieslak/DSI2/all_qsdr.json"
local_scans = get_local_data(qsdr_json)

fails = []
for sc in local_scans:
    print sc.scan_id,
    if not check_scan_for_files(sc):
        fails.append((sc,"missing files"))
        continue

    upload_succeeded, because = upload_local_scan(sc)
    if not upload_succeeded:
        fails.append((sc,because))

