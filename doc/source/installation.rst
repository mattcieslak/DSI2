Installing DSI2
================

**Note:** *Download instructions are for Mac OS users only*

Dependencies
~~~~~~~~~~~~
Here we describe how to install DSI2 on your system.  Before DSI2 will work, a number of python
packages must be installed.  The Enthought's Canopy_ is very useful if you work in academia and is highly recommended.
Otherwise you will need to install the following packages:

- NumPy_ / Scipy_
- Traits_
- TraitsUI_
- Chaco_
- MayaVi_
- Matplotlib_


Even if using Canopy, you will need to install the following packages:
-----------------------------------------------------------------------

First, install MongoDB_.

Next, use the "pip install _____" command in your terminal to install these packages. 
Make sure that you have Xcode *(from the app store)* and the command line tools installed ahead of time:

- PyMongo_ 
- Nibabel_
- DiPy_
- Matplotlib_

Next, you will be using Canopy's "Package Manager" in the welcome screen as shown below:
------------------------------------------------------------------------------------------

.. figure:: _static/welcome_canopy.png
   :scale: 58 %
   :align: center


**Search for "scikit" and install the** *top two* **packages:**  


.. figure:: _static/canopy_install.png
   :scale: 65 %
   :align: center


Below are additional links that can assist you if you run into problems or if you are not using Canopy:

- Scikit-Image_
- Scikit-Learn_
 
Open your terminal to download the DSI2 source code:
-----------------------------------------------------

The DSI2 source code can be downloaded from github_. Get the source by

.. code-block:: bash

   $ git clone git@github.com:mattcieslak/DSI2.git
   $ cd DSI2-master
   # If you have write permission to your python distribution
   $ python setup.py install
   # otherwise
   $ export PYTHONPATH=$PYTHONPATH:`pwd`

Setting up your environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a Mac, edit your .profile script to include this line, assuming you installed
dsi_studio.app in /Applications.

.. code-block:: bash

    alias dsi_studio=/Applications/dsi_studio.app/Contents/MacOS/dsi_studio

Starting MongoDB
-----------------
You will need to start a ``mongod`` process if you intend to use the MongoDB backend.
If you are running on a single machine and do not want to open your ports to the world
remember to pass a ``--bind_ip`` argument to ``mongod``. For example

.. code-block:: bash

  $ mongod --bind_ip 127.0.0.1

Verifying your installation
-----------------------------
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
.. _PyMongo: http://api.mongodb.org/python/current/installation.html
.. _Nibabel: http://nipy.org/nibabel/installation.html
.. _DiPy: http://nipy.org/dipy/installation.html
.. _Matplotlib: http://matplotlib.org/users/installing.html
.. _MongoDB: http://docs.mongodb.org/manual/installation
.. _github: https://github.com/mattcieslak/DSI2
.. _data:  https://labs.psych.ucsb.edu/grafton/scott/
