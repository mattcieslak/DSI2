
export SUBJECTS_DIR=/Users/Viktoriya/Desktop/subjects

list_subj="0240c 1188q 2171v 2917d 4222y 4487t"
for subj in $list_subj
do

recon-all -s ${subj} \
-i /Users/Viktoriya/DSI2/dsi_data/${subj}/MPRAGE/${subj}_T1.nii.gz \
-T2 /Users/Viktoriya/DSI2/dsi_data/${subj}/T2w/${subj}_T2.nii.gz
recon-all -s ${subj} -all

done