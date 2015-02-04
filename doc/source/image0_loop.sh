cd

list_subj="1188q 2171v 2917d 4222y 4487t"
for subj in $list_subj
do

	dsi_studio --action=exp \
	    --source=/Users/Viktoriya/DSI2/dsi_data/${subj}/CMRR_DSI/${subj}_CMRR.src.gz \
	    --export=image0
	dsi_studio --action=exp \
	    --source=/Users/Viktoriya/DSI2/dsi_data/${subj}/SIEMENS_MBWIP/${subj}_MBWIP.src.gz \
	    --export=image0

done