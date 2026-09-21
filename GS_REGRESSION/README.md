# GS_REGRESSION

## Initial directory structure

```
GS_REGRESSION/                    ← working folder
├── age_regression_lasso_cv.py
├── age_regression_xgb_cv.py
├── analyze_lasso_features.py
├── eval_ensemble_cv.py
├── eval_ensemble.py
├── train_eval_lasso.py
├── train_eval_xgboost.py
├── gmm_features_weighted/
│   ├── sub-100053248969_preproc-cat12vbm_desc-gm_T1w.npy
│   └── ...
├── val_gmm_weighted_outputs/
│   ├── sub-1000536464191_preproc-cat12vbm_desc-gm_T1w.npy
│   └── ...
├── train_labels/
│   ├── participants.tsv
│   └── ...
└── val_labels/
    ├── participants.tsv
    └── ...
```

## Commands

Run the commands in the following order:

1. `python age_regression_lasso_cv.py`: Lasso regression on training data, 10-fold cross validation, outputs stored in the folder `LASSOCV_RESULTS`
2. `python age_regression_xgb_cv.py`: XGBoost regression on training data, 10-fold cross validation, outputs stored in the folder `XGBCV_RESULTS`
3. `python eval_ensemble_cv.py`: Ensemble of lasso and xgboost, outputs stored in the folder `ensemble_cv`
4. `python analyze_lasso_features.py`: Gives statistics of surviving lasso features, outputs in the terminal log
5. `python train_eval_lasso.py`: Trains lasso on the full data and evaluates on the test set, outputs stored in the folder `lasso_final`
6. `python train_eval_xgb.py`: Trains xgboost on the full data and evaluates on the test set, outputs stored in the folder `xgb_final`
7. `python eval_ensemble.py`: Evaluates the ensemble of the two models on the test set, outputs stored in the folder `ensemble_final`
