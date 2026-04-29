Initial directory structure:

GS_REGRESSION/    ← working folder
├── age_regression_lasso_cv.py
├── age_regression_xgb_cv.py
├── analyze_lasso_features.py
├── eval_ensemble_cv.py
├── eval_ensemble.py
├── train_eval_lasso.py
├── train_eval_xgboost.py
├── gmm_features_weighted
    ├── sub-100053248969_preproc-cat12vbm_desc-gm_T1w.npy
    └── ...

├── val_gmm_weighted_outputs
    ├── sub-1000536464191_preproc-cat12vbm_desc-gm_T1w.npy
    └── ...

├── train_labels
    ├── participants.tsv
    └── ...

└── val_labels
    ├── participants.tsv
    └── ...

Run the commands in the following order:

1. python age_regression_lasso_cv.py  --
2. python age_regression_xgb_cv.py    --
3. python eval_ensemble_cv.py         --
4. python analyze_lasso_features.py   --
5. python train_eval_lasso.py         --
6. python train_eval_xgb.py           --
7. python eval_ensemble.py
