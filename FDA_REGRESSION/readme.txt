Initial directory structure:

FDA_REGRESSION/    ← working folder
├── age_regression_ridge_cv.py
├── age_regression_xgb_cv.py
├── error_correlation_analysis.py
├── fda_scatter.py
├── train_vbm_coefs
    ├── coef_0000_sub-100053248969_preproc-cat12vbm_desc-gm_T1w.npy
    └── ...

├── val_gmm_weighted_outputs
    ├── coef_0000_sub-1000536464191_preproc-cat12vbm_desc-gm_T1w.npy
    └── ...

├── train_labels
    ├── participants.tsv
    └── ...

Run the commands in the following order (all outputs stored in the working folder):

1. python age_regression_ridge_cv.py    --    Ridge regression on training data, 10-fold cross validation
2. python age_regression_xgb_cv.py      --    XGBoost regression on training data, 10-fold cross validation
3. python error_correlation_analysis.py --    Analyze correlations between errors made by the two regression models
4. python fda_scatter.py                --    Plots the predicted age vs true age scatter plot for the 50/50 ensemble of the two regression models    
