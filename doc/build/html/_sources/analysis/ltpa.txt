Accessing and analyzing your data
===================================

Once your data has been organized either into a MongoDB or local file format, you can
access streamline and label data to perform all sorts of analyses.

Searching your local datasource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the necessary streamline labels and pickle files are in their specified 
locations, we can begin to search the database. ::

  from dsi2.database.track_datasource import TrackDataSource

  # Load the json file to a list of Scan objects
  scans = get_local_data("path/to/file.json")

``scans`` will be a list of objects that can be used to select only the data 
you want for this analysis. It does **not** load the pkl into memory until 
you call its :py:meth:`~dsi2.database.traited_query.Scan.get_track_dataset()` function.
Let's select only scans from the "example" study and load them into memory.::

  example_scans = [ scan for scan in scans if scan.study_id == "example_study" ]
  # Use them to build a queryable data source
  data_source = TrackDataSource(track_datasets = [scan.get_track_dataset() for scan in scans])

``data_source`` provides an interface to a searchable MNI space. Raw streamlines 
aren't particularly useful, so we will create an object that uses this interface
to search through and aggregate streamlines.

Searching your MongoDB Datasource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you're using the unit testing data, you can access the ``dsi2_test`` database  
after making sure ``mongod`` is properly configured and running ::

  from dsi2.database.mongo_track_datasource import MongoTrackDataSource
  data_source = MongoTrackDataSource(
        scan_ids = ["0377A", "2843A" ],
        mongo_host = "127.0.0.1",
        mongo_port = 27017,
        db_name="dsi2"
    )

The MongoDB version has a number of advantages. Primarily, it makes concurrency easier
and scales much better than the pickle approach.  The 
:py:class:`~dsi2.database.mongo_track_datasource.MongoTrackDataSource` class has all the 
same methods as the pickle version.



Aggregating streamlines based on termination regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To apply perform the analysis from [1]_ we need to create an *aggregator*. An 
aggregator 
  
  * Subclasses :py:class:`~dsi2.aggregation.cluster_ui.ClusterEditor`
  * Overrides the ``aggregate`` method 
  * operates on a :py:class:`~dsi2.database.track_datasource.TrackDataSource` 

For a region-based LTPA, we can create a :py:class:`~dsi2.aggregation.region_labeled_clusters.RegionLabelAggregator`
that will provide some methods for easily analyzing these streamlines resulting from the
search. Suppose we'd like to search a set of coordinates around :math:`(33,54,45)`.::
  
  from dsi2.ltpa import mni_white_matter, run_ltpa
  from dsi2.aggregation import make_aggregator
  from dsi2.streamlines.track_math import sphere_around_ijk

  region_agg = make_aggregator( algorithm="region labels",
                                atlas_name="Lausanne2008",
                                atlas_scale=60, data_source=data_source)
  # create a set of search coordinates
  sphere_radius = 2                  # voxels
  center_coordinate = (33,54,45)     # in MNI152 i,j,k coordinates
  search_coords = sphere_around_ijk(sphere_radius, center_coordinate)

  # Query the data source with the coordinates, get new TrackDatasets
  query_tracks = data_source.query_ijk(search_coords, fetch_streamlines=False)

  # Feed the query results to the aggregator
  region_agg.set_track_sets(query_tracks)
  region_agg.update_clusters()

  # which regions pairs were found and how many streamlines to each?
  conn_ids, connection_vectors = region_agg.connection_vector_matrix()

  # if running an interactive ipython session, plot it
  region_agg.plot_connection_vector_lines(connection_vectors,conn_ids)

``conn_ids`` is a list of connections (regionA, regionB) found that pass through
the search coordinates. If there are :math:`n` individuals in the :py:class:`~dsi2.database.track_datasource.TrackDataSource`
and :math:`m` elements in ``conn_ids``, then ``connection_vectors`` will be an
:math:`n \times m` matrix where row :math:`i` column :math:`j` contains the streamline
count connection region pair math:`j` in subject :math:`i`\'s data.


Running a whole-brain LTPA
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code above is tedious and would take a long time to loop over the whole brain.
It is much more convenient to use the :py:meth:`~dsi2.ltpa.run_ltpa` function. Here 
is an example script that performs a simple whole-brain ltpa.  It requires a description
of an aggregator. 

Create a dictionary containing all the information necessary to construct
an aggregator for your analysis function. It works this way instead of 
by directly creating an Aggregator object because run_ltpa constructs a new
Aggregator inside each independent process it launches.

The dictionary must contain at least the key "algorithm", which can be one of 
{ "region labels", "k-means", "quickbundles"}. The rest of the keys are sent
as keyword arguments to :py:meth:`~dsi2.aggregation.make_aggregator`. 

When run_ltpa is looping over coordinates, result of a spatial query is sent
to an instance of the aggregator.  The aggregator's ``aggregate()`` method 
is called for each TrackDataset returned from the query, then the aggregator
is sent to whichever function you provided to run_ltpa.

NOTE: If you select the "region labels" aggregator, then you won't have access
to the streamline objects. To access streamlines, choose "k-means" or 
"quickbundles". ::
  
  agg_args = {
            "algorithm":"region labels",
            "atlas_name":"Lausanne2008",
            "atlas_scale":60,
            "data_source":data_source
            }

You will also need to define a function that will extract the information you care about
from the aggregator.::

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

The variable ``mni_white_matter`` contains the coordinates inside of FSL's MNI 2mm
white matter mask.  We can split up these coordinates across a number of processes.
Here we use two processors. ::

  results = run_ltpa(get_n_streamlines, data_source=data_source,
                     aggregator_args=agg_args, radius=2,
                     n_procs=2, search_centers=mni_white_matter)

For each coordinate in ``mni_white_matter`` a tuple is stored in ``results`` that contains
the mean streamline count and its standard deviation.  You can make the analysis function
return as much data as you'd like. It will always contain results in the same order as the
coordinates specified in ``search_centers``.


References
~~~~~~~~~~~

.. [1] Cieslak, M., & Grafton, S.T. Local termination pattern analysis:
    a tool for comparing white matter morphology. Brain Imaging Behav, DOI 10.1007/s11682-013-9254-z (2013).
