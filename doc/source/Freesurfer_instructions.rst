Using Freesurfer:

Run Surface and Volumes at same time with this code:

In terminal:

export SUBJECTS_DIR=/Users/Viktoriya/Desktop/subjects

cd $SUBJECTS_DIR

cd 0240c/

freeview -v \
mri/T1.mgz \
mri/wm.mgz \
mri/brainmask.mgz \
mri/aseg.mgz:colormap=lut:opacity=0.2 \
-f surf/lh.white:edgecolor=blue \
surf/lh.pial:edgecolor=red \
surf/rh.white:edgecolor=blue \
surf/rh.pial:edgecolor=red \
-f  surf/lh.pial:annot=aparc.annot:name=pial_aparc:visible=0 \
surf/lh.inflated:overlay=lh.thickness:overlay_threshold=0.1,3::name=inflated_thickness:visible=0 \
surf/lh.inflated:visible=0 \
surf/lh.white:visible=0 \
surf/lh.pial \
surf/rh.pial:annot=aparc.annot:name=pial_aparc:visible=0 \
surf/rh.inflated:overlay=lh.thickness:overlay_threshold=0.1,3::name=inflated_thickness:visible=0 \
surf/rh.inflated:visible=0 \
surf/rh.white:visible=0 \
surf/rh.pial


