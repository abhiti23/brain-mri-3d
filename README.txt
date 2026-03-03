Pre-requisites:
-Download dataset
	1. OpenBHB Challenge-Training+Validation NumPy (Size: 44.96 GB) from 
	2. https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing

-Extract files into public_data_challenge folder. Contents should be
	1. cv_splits – JSON file (133KB)
	2. cv_splits_indices – JSON file (111KB) 
	3. external_test.npy (5,646,606KB)
	4. internal_test.npy (5,174,864KB)
	5. test.npy (10,821,469KB)
	6. train.npy (46,130,621KB)
	7. external_test.tsv (47KB)
	8. internal_test.tsv (43KB)
	9. test.tsv (95KB)
	10.train.tsv (380KB)

-Create folder titled "V2" inside public_data_challenge and run .py files in this folder


Files (run in this order):
-extract_vbm.py: Parses out VBMs from three provided datatypes
	1) Quasi-Raw 3D MRI Volumes
	2) Voxel-Based Morphometry (VBM) Maps
	3) Surface (SBM) & Regional Features (ROI)
	Note: There should be 3227 total scans

-vbm_gaussian_splatting.py: Constructs "K" number of Gaussians for each scan. Default is 500 Gaussians.
