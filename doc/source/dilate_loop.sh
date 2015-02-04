list_num="33 60 125 250"
list_subj="4487t"
for subj in $list_subj
do
for num in $list_num
do

atlas_dilate /Users/Viktoriya/vik.DSI2/dsi_data/${subj}/CMRR_DSI/ROIv_scale${num}.nii.gz \
/Users/Viktoriya/vik.DSI2/dsi_data/${subj}/CMRR_DSI/ROIv_scale${num}_thickened.nii.gz

atlas_dilate /Users/Viktoriya/vik.DSI2/dsi_data/${subj}/SIEMENS_MBWIP/ROIv_scale${num}.nii.gz \
/Users/Viktoriya/vik.DSI2/dsi_data/${subj}/SIEMENS_MBWIP/ROIv_scale${num}_thickened.nii.gz

done
done