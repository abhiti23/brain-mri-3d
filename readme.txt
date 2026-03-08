This github repository predicts age using VBM brain images obtained form the OpenBHB website.

Pre-requisites:
-Download dataset into data/raw/
	1. OpenBHB Challenge-Training+Validation NumPy (Size: 44.96 GB) from 
	2. https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing

-Extract files into data/raw/public_data_challenge folder. Contents should be
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
-extract_train_age.py
-extract_test.py

Please run all files from the root. 

The artifacts/ folder already contains coefficients_final.npz,
which has the coefficients obtained from representing the training 
VBM images as a Tensor BSpline basis. The following instructions are
for reproducing these coefficients, but
WARNING: extracting FD coefficients will take up to 1 day.
Use the following command to create an npz file with coefficients (assuming
that the previous data extraction has been performed)-
python src/analysis/fd_coef.py

Once artifacts/coefficients_final.npz exists, the make file can be run as
-make visualize_slice: Visualizing 2D slices for a random subject from
	the raw VBM file and comparison with the 3D tensor obtained from
 	the functional data basis approximation. 
	Creates artifacts/heatmap_first_slice.png
-make train_nn: Trains shallow NN using coefficients_final.npz and
	plots results/figures/training_loss.png 
