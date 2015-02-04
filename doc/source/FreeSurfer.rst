Preparing Data with Freesurfer
===============================

Running FreeSurfer


Make sure you've sourced your setupfreesurfer.sh file
Make sure your environment variable for SUBJECTS_DIR is set to your desired location

.. code-block:: bash

	$ export SUBJECTS_DIR=/Users/Viktoriya/Desktop/subjects
	
Next step will take input data and copy it into SUBJECTS_DIR	

.. code-block:: bash

	$ recon-all \
	  ... -s example_data \
	  ... -i /Users/Viktoriya/Desktop/subjects/example_MPRAGE.nii.gz \
	  ... -T2 /Users/Viktoriya/Desktop/subjects/example_t2wspace.nii.gz
	
Now run the FreeSurfer reconstruction pipeline

.. code-block:: bash

	$ recon-all -s example_data -all
	
	