STARTING FROM ZERO:
-Go to: https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing
-Download: OpenBHB-Training Data (Size: 42.15 GB)
-You should have a train.zip file at the end of this
-Download: OpenBHB-Validation Data (Size: 9.91 GB)
-You should have a val.zip file at the end of this


FOR TRAINING SET:
-Unzip the file
	- unzip train.zip
	-If it triggers a zip bomb error, do this: UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip train.zip and select [A]
-Remove unnecessary files
	rm train_fsl.zip train_quasiraw.zip train_roi.zip
-Unzip VBM data
	UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip train_vbm.zip
-Unzip labels
	unzip train_labels.zip
	ls *.csv 2>/dev/null || ls *.tsv 2>/dev/null || ls *label* 2>/dev/null

Repeat for validation set. Then feature extraction (GS or FDA) can begin.
