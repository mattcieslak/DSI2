Installing DSI2
================

Dependencies
~~~~~~~~~~~~
Here we describe how to install DSI2 on your system.  Before DSI2 will work, a number of python
packages must be installed.  The Enthought's Canopy_ is very useful if you work in academia.
Otherwise you will need to install the following packages:

- NumPy_ / Scipy_
- Traits_
- TraitsUI_
- Chaco_
- MayaVi_
- Matplotlib_

Even if using Canopy, you will need to install the following packages:

- PyMongo_
- Nibabel_
- DiPy_
- Matplotlib_
- Scikit-Image_
- Scikit-Learn_

You will also need to install MongoDB_. 

The DSI2 source code can be downloaded from github_. Get the source by

.. code-block:: bash

   $ git clone git@github.com/mattcieslak/DSI2.git
   $ cd DSI2-master
   # If you have write permission to your python distribution
   $ python setup.py install
   # otherwise
   $ export PYTHONPATH=$PYTHONPATH:`pwd`

Setting up your environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``dsi2`` requires some environment variables to be set so it can find
resources in your file system. On a \*nix system these are set using the
``export`` command.

 * ``DSI2_DATA`` is the path to the ``example_data`` directory that 
   is provided on this website (**NOTE: upload the example data**).
   This is data that dsi2 needs internally (such as standard atlas 
   labels and the MNI 152 volume).

 * ``LOCAL_TRACKDB`` is the path to your directory containing your
   locally stored .pkl files. If this variable is not set, dsi2 will
   check your home directory for a ``local_trackdb`` directory. If you 
   have downloaded example_trackdb.tar.gz, the location of its extracted
   directory 


Starting MongoDB
"""""""""""""""""
You will need to start a ``mongod`` process if you intend to use the MongoDB backend.
If you are running on a single machine and do not want to open your ports to the world
remember to pass a ``--bind_ip`` argument to ``mongod``. For example

.. code-block:: bash

  $ mongod --bind_ip 127.0.0.1

Verifying your installation
"""""""""""""""""""""""""""
Download the unit testing data_

.. _Canopy: https://enthought.com/products/canopy/
.. _SciPy: http://www.scipy.org/install.html
.. _NumPy: http://www.scipy.org/install.html
.. _TraitsUI: https://github.com/enthought/traitsui
.. _Traits: https://github.com/enthought/traits
.. _MayaVi: https://github.com/enthought/mayavi
.. _Chaco: https://github.com/enthought/chaco
.. _Scikit-Image: http://scikit-image.org/download
.. _Scikit-Learn: http://scikit-learn.org/stable/install.html
.. _PyMongo: http://scikit-learn.org/stable/install.html
.. _Nibabel: http://nipy.org/nibabel/installation.html
.. _DiPy: http://nipy.org/dipy/installation.html
.. _Matplotlib: http://matplotlib.org/users/installing.html
.. _MongoDB: http://docs.mongodb.org/manual/installation
.. _github: https://github.com/mattcieslak/DSI2
.. _data:  https://labs.psych.ucsb.edu/grafton/scott/
