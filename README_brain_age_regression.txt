File structure:
some_folder/
├── brain_age_regression.py
├── participants.tsv
└── gmm_1d_outputs/
    ├── sub-100053248969_preproc-cat12vbm_desc-gm_T1w.npy
    ├── sub-100263562592_preproc-cat12vbm_desc-gm_T1w.npy
    └── ...

Running file:
pip install numpy pandas scikit-learn matplotlib
python brain_age_regression.py