DSI Studio
==========

Reconstructing DWI
------------------------
DSI Studio's QSDR algorithm is very useful for reconstructing
DWI data in a standard space. The process begins with creating
a ``.src`` file

.. code-block:: bash

  $ dsi_studio \
      --action=src \
      --source=/path/to/dicom_files \
      --output=output.src.gz

Once the ``.src.gz`` is created, reconstruct it with

.. code-block:: bash

  $ dsi_studio \
      --action=rec \
      --thread=2 \
      --source=output.src.gz \
      --method=7 \
      --param0=1.25 \
      --param1-2 \
      --output_map=1

This will produce a ``map.fib.gz`` file.

Fiber Tracking
--------------
Tractography is performed on the ``map.fib.gz`` file.
Supposing the goal is to reconstruct 100,000 streamlines
with some conservative parameters. See DSI Studio's
`documentation <http://dsi-studio.labsolver.org/Manual/Fiber-Tracking>`
to tune the parameters appropriately.

.. code-block:: bash

  $ dsi_studio \
      --action=trk \
      --fiber_count=100000 \
      --turning_angle=55 \
      --min_length=10 \
      --smoothing=0.0 \
      --max_length=400 \
      --output=output.trk

These parameters work well for data from a Siemens Tim Trio 3T
scaner. Your mileage may vary.

