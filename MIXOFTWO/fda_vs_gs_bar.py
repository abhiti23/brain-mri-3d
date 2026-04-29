"""
Bar plot of MAE for FDA, GS, and their ensembles.

Inputs:
  - ensemble_predictions_fda.csv  (uses 'pred_avg')
  - ensemble_predictions_gs.csv   (uses 'pred_ensemble')

Output:
  - fda_vs_gs_mae_bar.png
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-fda', default='ensemble_predictions_fda.csv')
    ap.add_argument('--csv-gs', default='ensemble_predictions_gs.csv')
    ap.add_argument('--pred-col-fda', default='pred_avg')
    ap.add_argument('--pred-col-gs', default='pred_ensemble')
    ap.add_argument('--out', default='fda_vs_gs_mae_bar.png')
    args = ap.parse_args()

    a = pd.read_csv(args.csv_fda)[['subject', 'true_age', args.pred_col_fda]].rename(
        columns={args.pred_col_fda: 'pred_a'})
    b = pd.read_csv(args.csv_gs)[['subject', 'true_age', args.pred_col_gs]].rename(
        columns={args.pred_col_gs: 'pred_b'})
    merged = a.merge(b, on='subject', suffixes=('_a', '_b'),
                     validate='one_to_one')

    y = merged['true_age_a'].values
    p_fda = merged['pred_a'].values
    p_gs = merged['pred_b'].values

    # 50/50 average ensemble
    p_avg = 0.5 * (p_fda + p_gs)

    # Optimal convex weight (closed form on these out-of-fold predictions)
    d = p_fda - p_gs
    var_d = float(np.sum(d ** 2))
    w = float(np.dot(p_fda - p_gs, y - p_gs) / var_d) if var_d > 1e-12 else 0.5
    w = float(np.clip(w, 0.0, 1.0))
    p_opt = w * p_fda + (1 - w) * p_gs

    maes = [
        mean_absolute_error(y, p_fda),
        mean_absolute_error(y, p_gs),
        mean_absolute_error(y, p_avg),
        mean_absolute_error(y, p_opt),
    ]
    labels = ['FDA', 'GS', '50/50 avg', f'Optimal\n(w_FDA={w:.2f})']
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, maes, color=colors)
    for b_, v in zip(bars, maes):
        ax.text(b_.get_x() + b_.get_width() / 2, v, f'{v:.3f}',
                ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('MAE (years)')
    ax.set_title('FDA vs GS — single models and ensembles')
    ax.set_ylim(0, max(maes) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"[*] Saved {args.out}")


if __name__ == "__main__":
    main()
