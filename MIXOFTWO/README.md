Initial directory structure

MIXOFTWO/    ← working folder
├── ensemble_predictions_fda.csv
├── ensemble_predictions_gs.csv
└── fda_vs_gs_bar.py

The files ensemble_predictions_fda.csv and ensemble_predictions_gs.csv contain the 10 fold cv age predictions from the two approaches. The script fda_vs_gs_bar.py evaluates the optimal ensembling weights for FDA and GS.

Run the command: python fda_vs_gs_bar.py  --  outputs the file fda_vs_gs_mae_bar.png
