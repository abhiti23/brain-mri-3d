# 3D MRI Brain Age Prediction using Gaussian Splatting

Predicting chronological age from structural brain MRI, using a Gaussian splatting representation of each brain as features for regression. Course project for EECS 545 at the University of Michigan.

The gap between predicted and chronological brain age is an early marker of atypical aging and neurodegenerative disease. We train on healthy subjects from the OpenBHB dataset: 3227 training and 757 test subjects, using VBM data.

See [`poster.pdf`](poster.pdf) for figures and full results. To reproduce the work, start with [`SETUP.md`](SETUP.md).

## Method

**Representation.** Each brain's voxel intensities are modeled as a weighted sum of 500 learnable 3D Gaussians. These are fit with a weighted EM, in which the MRI intensity at each voxel acts as a weight in the M-step. A standard GMM clusters only the voxel coordinates. Weighting by intensity means the fitted mixing weights and covariances capture regional tissue volume and shape.

**Anchoring.** The 500 Gaussians are first fit freely to a group-average brain built from 150 subjects. Their centers are then frozen. Each subject is fit by updating only the mixing weights and covariances, starting from the average-brain solution. This makes every Gaussian correspond to the same anatomical location in every subject, so all subjects share one fixed-length feature vector (3500 features).

**Regression.** Lasso (for sparsity, since there are 3500 features and 3227 subjects) and XGBoost (for nonlinearity), combined in a 50/50 ensemble. The two models' validation errors were only moderately correlated (r = 0.55), which motivated ensembling.

## Results

| Model | Test MAE (years) |
|---|---|
| ResNet (benchmark) | 2.67 |
| Lasso | 4.17 |
| XGBoost | 3.95 |
| Ensemble | 3.70 |

Only 777 features survive Lasso. Train MAE under 10-fold CV (3.35) is close to test MAE (3.70), so the model generalizes despite being underdetermined. Predictions skew young for older subjects, likely because of age imbalance in the training data.

## Repository layout

| Folder | Contents |
|---|---|
| `GET_GS_FEATURES/` | Fit the average-brain Gaussians and extract per-subject splatting features |
| `GS_REGRESSION/` | Lasso, XGBoost and ensemble on the splatting features |
| `GET_FDA_FEATURES/` | Alternative features: 216 tensor-product B-spline coefficients per subject |
| `FDA_REGRESSION/` | Ridge, XGBoost and ensemble on the B-spline features |
| `MIXOFTWO/` | Ensembling weights between the FDA and splatting predictions |

Each folder has its own README with run instructions.

## Authors

Abhishek Koparde, Abhiti Mishra, Mohith Nagaraju, Anvay Pradhan
