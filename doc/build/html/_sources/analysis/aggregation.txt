Creating a streamline aggregator
===================================

There are many ways to group streamlines.  Streamlines can be labeled based on 
which brain regions they connect, such as a 
:pyclass:`~dsi2.aggregation.region_labeled_clusters.RegionLabelAggregator`.  Or they
can be grouped based on the similarity of their morphology, such as with a QuickBundlesAggregator.

One of the advantages of using DSI2 is that you can test out a custom clustering algorithm
on any part of the brain while interactively changing the parameters of your algorithm.  Here 
we'll build a new Aggregator from scratch.

  * Subclasses :py:class:`~dsi2.aggregation.cluster_ui.ClusterEditor`
  * Overrides the ``aggregate`` method 
  * operates on a :py:class:`~dsi2.database.track_datasource.TrackDataSource` 

Defining the Aggregator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defining algorithm parameters as traits
""""""""""""""""""""""""""""""""""""""""
The UI and event listening functionality for Aggregators is inherited from the 
:pyclass:`~dsi2.aggregation.cluster_ui.ClusterEditor`.  Whichever parameters will
be needed by your aggregation algorithm should be defined as Traits in the class. ::

  from dsi2.aggregation.cluster_ui import ClusterEditor
  from sklearn.cluster import MiniBatchKMeans
  from dipy.tracking.metrics import downsample
  from traits.api import Float, Int, Enum
  from traitsui.api import Item, View, Group
  import numpy as np

  class KMeansAggregator(ClusterEditor):
      k = Int(3,
              label="Epsilon",
              desc="How many groups should the streamlines be grouped into? ",
              parameter=True)


The parameter of the k-means algorithm is attached to the class as Traits with 
some special metadata. It is crucial that ``parameter=True`` is passed to each of 
these Trait definitions so that ``ClusterEditor`` knows to update its streamline
labeling if the value gets changed in the UI. 

Specifying the GUI editors for your aggregator
"""""""""""""""""""""""""""""""""""""""""""""""
The ``ClusterEditor`` superclass looks for a variable in its subclasses called
``algorithm_widgets``.  Editor widgets from TraitsUI are defined in ``algorithm_widgets``. ::
  
  class KMeansAggregator(ClusterEditor):
      ...
      algorithm_widgets = Group(
                            Item("k")
                            )

More advanced editors can be specified, but for now we'll just use the default
editors provided by TraitsUI.


Overriding the ``.aggregate()`` method
"""""""""""""""""""""""""""""""""""""""
To fit the streamline data into groups, we must override the ``.aggregate()`` 
method.  This method should expect a single ``TrackDataset`` as its argument.
We will access its ``.tracks`` property and apply some transformation that
turns an arbitrarily shaped streamline into a feature vector that k-means can
use.  Here we will copy DSI Studio's clustering and extract the following features:
the first, middle and last coordinate and the length of each streamline form a 10-feature
vector. ::

  def aggregate(self, track_dataset):
      """
      An example implementation of the k-means algorithm implemented in 
      DSI Studio.  This function is automatically applied to all 
      TrackDatasets returned from a query.

      Parameters:
      -----------
      track_dataset:dsi2.streamlines.track_dataset.TrackDataset
      """
      # extract the streamline data
      tracks = track_dataset.tracks
      
      # Make a matrix of downsampled streamlines
      points = np.array([ downsample(trk, n_pol=3).flatten() \
                                  for trk in tracks])

      # Calculate the length of each streamline
      lengths = np.array([len(trk) for trk in tracks]).reshape(-1,1)
      
      # Concatenate the points and the track lengths
      features = np.hstack((points, lengths))
      
      # Initialize the k-means algorithm
      kmeans = MiniBatchKMeans(n_clusters=self.k, compute_labels=True)
      kmeans.fit(features)

      # Return the labels
      return kmeans.labels_


That's all there is to it!

Exploring your aggregator in realtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the aggregator class defined, we can begin applying it to streamlines.
Let's create a data source, an aggregator, and set up a sphere browser to
use them ::

  from dsi2.database.mongo_track_datasource import MongoTrackDataSource
  from dsi2.ui.sphere_browser import SphereBrowser

  # Only select a single scan from the test data
  test_subject = [ "0377A" ]

  data_source = MongoTrackDataSource(
    scan_ids = test_subject,
    mongo_host = "127.0.0.1",
    mongo_port = 27017,
    db_name="dsi2_test"
  )

  kmeans_agg = KMeansAggregator()

  browser = SphereBrowser(track_source=data_source, aggregator=kmeans_agg)
  browser.edit_traits()
