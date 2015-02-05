DSI2 Download Details
======================

**Note:** *Download instructions are for Mac OS users only*

FreeSurfer
~~~~~~~~~~~~
You will need to download the latest version of FreeSurfer_ for your operating system:

.. figure:: _static/freesurf.png
   :scale: 85 %
   :align: center

Once you've downloaded the package (above), right click to "open" and begin installation of FreeSurfer.  

The "Read Me" instructions in the downloading process will instruct you to do the following:

In your terminal, type in "open .profile" from your home directory and paste the following into your .profile:

.. code-block:: bash

	export FREESURFER_HOME=/Applications/freesurfer
	source $FREESURFER_HOME/SetUpFreeSurfer.sh  # ← Make sure you make it a ".sh" file and NOT ".csh"

Afterwards, continue with installation process.

XQuartz
--------

You *may* be required to download XQuartz_ as well:

.. figure:: _static/xquartz.png
   :scale: 85 %
   :align: center

Once you've downloaded the package (above), right click to "open" and begin installation of XQuartz.  

Follow the "Read Me" instructions before completing installation.

Easy Lausanne
~~~~~~~~~~~~~~~~~~~~~~~~~~
Easy Lausanne takes input data and creates an atlas to label streamlines for tractography of the brain.

Enter the following into your terminal from your home directory:

.. code-block:: bash

   $ git clone git@github.com:mattcieslak/easy_lausanne.git
   $ cd easy_lausanne-master
   # If you have write permission to your python distribution
   $ python setup.py install
   # otherwise
   $ export PYTHONPATH=$PYTHONPATH:`pwd`


Installing DSI Studio
~~~~~~~~~~~~~~~~~~~~~~~
Finally, download and install DSIstudio_

.. _FreeSurfer: http://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/5.3.0-HCP/
.. _XQuartz: http://xquartz.macosforge.org/landing/
.. _DSIstudio: http://dsi-studio.labsolver.org/dsi-studio-download


