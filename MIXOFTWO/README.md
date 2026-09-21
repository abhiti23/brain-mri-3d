# MIXOFTWO

Ensembles the predictions from the FDA and Gaussian splatting (GS) approaches.

## Directory structure

```
MIXOFTWO/                         ← working folder
├── ensemble_predictions_fda.csv
├── ensemble_predictions_gs.csv
└── fda_vs_gs_bar.py
```

`ensemble_predictions_fda.csv` and `ensemble_predictions_gs.csv` contain the 10-fold CV age predictions from the two approaches. `fda_vs_gs_bar.py` finds the optimal ensembling weights for FDA and GS.

## Usage

```bash
python fda_vs_gs_bar.py
```

This produces `fda_vs_gs_mae_bar.png`.
