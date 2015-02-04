export SUBJECTS_DIR=/Users/Viktoriya/Desktop/subjects

list_subj="1188q"
for subj in $list_subj
do

easy_lausanne \
--target_type anisotropy \
--subject_id ${subj} \
--target_volume /Users/Viktoriya/vik.DSI2/dsi_data/${subj}/CMRR_DSI/${subj}_QAO.nii.gz \
--output_dir /Users/Viktoriya/vik.DSI2/dsi_data/${subj}/CMRR_DSI

#easy_lausanne \
#--target_type anisotropy \
#--subject_id ${subj} \
#--target_volume /Users/Viktoriya/vik.DSI2/dsi_data/${subj}/SIEMENS_MBWIP/${subj}_MBWIP.src.gz.image0.nii.gz \
#--output_dir /Users/Viktoriya/vik.DSI2/dsi_data/${subj}/SIEMENS_MBWIP

done