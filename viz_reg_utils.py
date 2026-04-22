"""
Shared utilities for regression visualization files.
Loads feature_importance.csv and computes per-component importance scores.
"""

import numpy as np
import pandas as pd

K = 500


def load_component_scores(csv_path="feature_importance.csv"):
    """
    Load feature_importance.csv and return per-component importance arrays.

    Returns
    -------
    weight_imp   : (K,) importance from weight features only
    cov_imp      : (K,) importance from covariance features only
    total_imp    : (K,) total importance (weight + covariance)
    weight_sign  : (K,) sign of net weight importance (+1 / -1 / 0)
                   Can be used for blue/red colouring if sign data is available.
                   NOTE: XGBoost importances are unsigned, so sign is always +1 here.
    """
    df = pd.read_csv(csv_path)

    weight_imp  = np.zeros(K)
    cov_imp     = np.zeros(K)

    for _, row in df.iterrows():
        k   = int(row["component"])
        imp = float(row["importance"])
        if row["feature_type"] == "weight":
            weight_imp[k] += imp
        else:
            cov_imp[k] += imp

    total_imp = weight_imp + cov_imp

    # Normalise each to [0, 1]
    def norm(x):
        m = x.max()
        return x / m if m > 0 else x

    return norm(weight_imp), norm(cov_imp), norm(total_imp)
