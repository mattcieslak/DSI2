MongoDB Backend
================

MongoDB can be downloaded from mongodb.org. DSI2 uses PyMongo to connect
to mongodb.


Access
---------

You can access the mongodb using the mongoshell::

  mongo dsi2


The output should look like::

  MongoDB shell version: 2.X.X
  connecting to: dsi2
  >


MongoDB Organization
---------------------
``dsi2`` relies on a database called "dsi2" in your mongodb instance.  There are a number
of collections inside dsi2 that are used.

To get the list of collections in the database::

  > db.getCollectionNames()
  [
	"atlases",
	"connections",
	"coordinates",
	"my_collection_keys",
	"scans",
	"streamline_labels",
	"streamlines",
	"system.indexes",
	"tmp_sl"
  ]
  > 


"dsi2.scans"
  This collection holds non-identifiable information about the individuals in the database.
  Information such as which experiment they were a part of, age, gender, etc. is included
  in this collection.  Generally scans is the first database queried, then scan_id's from
  this query are used in spatial queries during analysis.
  
"dsi2.coordinates"
  This collection contains the mapping from spatial index to streamline id. By querying
  a set of spatial indices, mongodb will return the set of all streamline ids that 
  intersect those spatial coordinates. These streamline ids can be used to query the
  other collections such as atlases and streamlines
  
"dsi2.streamlines"
  This collection contains serialized binary data that can be loaded as a numpy array.
  You can search a streamline id from a specific subject and get its spatial trajectory.
  
"dsi2.atlases"
  This contains the unique atlases that have been used to label streamlines.
  
"dsi2.streamline_labels"
  This collection contains the lists of connection ids for each
  streamline in a dataset.  This is useful because the connection ids
  are ints and small, so upon loading a dataset, one can simply query
  that atlas's connection ids for a scan and hold them in memory
  instead of performing repeated joins in the mongo client.
  
"dsi2.connections":
  To query a specific connection (a pair of regions from a specific atlas) you 
  query this database with a subject id and connection id from the atlas. The 
  streamline ids connecting that region pair are returned.
  
Local Access
---------------

In case you want to connect to a remote mongodb server behind a NAT
that only offers ssh access you can use ssh tunnelling:

From your local machine::

  ssh -f uname@remote_mongo_machine -L 10000:localhost:27017 -N

Where 27017 is the default mongodb port assumed on the remote server,
and 10000 is the local port from which all connections will be
forwarded to. The above command will run in background and return
immediately.

Then, from your local machine, assuming you have mongoshell installed::

  mongo --port 10000 dsi2

Should give you::

  MongoDB shell version: 2.X.X
  connecting to: 127.0.0.1:10000/dsi2
  > 
