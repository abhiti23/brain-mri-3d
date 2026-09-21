# FDA_REGRESSION

Age regression on the FDA features: Ridge and XGBoost with 10-fold cross-validation, error correlation analysis, and a 50/50 ensemble.

## Directory structure

```
FDA_REGRESSION/                   ← working folder
├── age_regression_ridge_cv.py
├── age_regression_xgb_cv.py
├── error_correlation_analysis.py
├── fda_scatter.py
├── train_vbm_coefs/
│   ├── coef_0000_sub-100053248969_preproc-cat12vbm_desc-gm_T1w.npy
│   └── ...
├── val_gmm_weighted_outputs/
│   ├── coef_0000_sub-1000536464191_preproc-cat12vbm_desc-gm_T1w.npy
│   └── ...
└── train_labels/
    ├── participants.tsv
    └── ...
```

## Usage

Run the scripts in this order. All outputs are written to the working folder.

1. `python age_regression_ridge_cv.py`: Ridge regression on the training data, 10-fold CV
2. `python age_regression_xgb_cv.py`: XGBoost regression on the training data, 10-fold CV
3. `python error_correlation_analysis.py`: correlation between the errors of the two models
4. `python fda_scatter.py`: predicted vs. true age scatter plot for the 50/50 ensemble
