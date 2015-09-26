#!/usr/bin/env python
import pymongo
from bson.binary import Binary

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

def upload_atlases(db, trackds, sc):
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

def upload_streamlines(db, trackds, sc):
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

def upload_coordinate_info(db, trackds, sc):
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

def upload_scan_info(db, trackds, sc):
    try:
        #db.scans.insert([sc.original_json])
        atlases = []
        for label in sc.track_label_items:
            # Does this atlas already exist? If not, add it to the collection.
            atlas = None
            result = db.atlases.find( { "name": label.name, "parameters": label.parameters } )
            if result.count() != 0:
                atlas = result[0]["_id"]
            else:
                atlas = db.atlases.insert( { "name": label.name, "parameters": label.parameters } )
    
            atlases.append(atlas)
        db.scans.insert([
                {
                    "scan_id":sc.scan_id,
                    "subject_id":sc.subject_id,
                    "gender":sc.scan_gender,
                    "age":sc.scan_age,
                    "study":sc.study,
                    "group":sc.scan_group,
                    "smoothing":sc.smoothing,
                    "cutoff_angle":sc.cutoff_angle,
                    "qa_threshold":sc.qa_threshold,
                    "gfa_threshold":sc.gfa_threshold,
                    "length_min":sc.length_min,
                    "length_max":sc.length_max,
                    "institution":sc.institution,
                    "reconstruction":sc.reconstruction,
                    "scanner":sc.scanner,
                    "n_directions":sc.n_directions,
                    "max_b_value":sc.max_b_value,
                    "bvals":sc.bvals,
                    "bvecs":sc.bvecs,
                    "label":sc.label,
                    "trk_space":sc.trk_space,
                    "atlases":list(set(atlases)),
                    "sls": len(trackds.tracks),
                    "header":Binary(pickle.dumps(trackds.header,protocol=2)),
                    "original_json":sc.original_json
                }
        ])
    except Exception, e:
        print "Failed to upload scan info", e
        return False
    return True


def upload_local_scan(db, sc):
    logging.info("uploading %s", sc.scan_id)
    try:
        trackds = sc.get_track_dataset()
    except:
        print "failed to read pkl file"
        return False, "pkl file corrupt"

    if not upload_atlases(db, trackds, sc):
        print "failed to upload atlases"
        return False, "upload_atlases"

    if not upload_streamlines(db, trackds, sc):
        print "failed to upload streamlines"
        return False, "upload_streamlines"

    if not upload_coordinate_info(db, trackds, sc):
        print "failed to upload spatial mapping"
        return False, "upload_coordinate_info"

    if not upload_scan_info(db, trackds, sc):
        print "failed to upload spatial mapping"
        return False, "upload scan info"

    return True, "hooray!"
class MongoCreator(HasTraits):
    database_dir = File()
    log_path = File()
    b_start = Button("Start mongod")
    restrict_ips = Bool(True)
    numactl_interleave = Bool(False)
    port = Str("27017")

    def get_command(self):
        cmd = []
        if self.numactl_interleave:
            cmd += ["numactl", "--interleave=all" ]

        cmd += ["mongod", "--fork", "--dbpath", self.database_dir,
                "--logpath", self.log_path, "--port", self.port ]

        if self.restrict_ips:
            cmd += ["--bind_ip", "127.0.0.1"]
        return cmd

    def get_connection(self):
        conn = pymongo.MongoClient(
            port=int(self.port),host="localhost")
        return conn
    def _b_start_fired(self):
        print "Starting mongod"
        cmd = self.get_command()
        print cmd
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir)
        proc = subprocess.Popen(cmd,
                                stdout = subprocess.PIPE,shell=False)
        result = proc.communicate()
        print result
        return result

    def shutdown(self):
        conn = self.get_connection()
        dba = conn.admin
        try:
            dba.command({"shutdown":1})
        except Exception, e:
            print e

    traits_view = View(
        VGroup(
            Item("database_dir"),
            Item("log_path"),
            Item("restrict_ips"),
            Item("port"),
            Item("numactl_interleave"),
            Group(
                Item("b_start"), show_labels=False)
            ), title="MongoDB Connection"
    )
