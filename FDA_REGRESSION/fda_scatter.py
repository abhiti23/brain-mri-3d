"""
Scatter plot of the FDA 50/50 ensemble prediction vs true age.

Input:
  - ensemble_predictions_fda.csv with columns: subject, true_age, pred_avg

Output:
  - fda_pred_vs_true.png
"""

import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='ensemble_predictions.csv')
    ap.add_argument('--pred-col', default='pred_avg')
    ap.add_argument('--out', default='fda_pred_vs_true.png')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = df['true_age'].values
    p = df[args.pred_col].values
    mae = mean_absolute_error(y, p)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, p, s=10, alpha=0.4, color='#4c72b0')

    lims = [min(y.min(), p.min()) - 2, max(y.max(), p.max()) + 2]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.6, label='y = x')
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')

    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted Age (FDA 50/50 ensemble)')
    ax.set_title(f'FDA 50/50 ensemble: Predicted vs True Age   (MAE = {mae:.2f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"[*] Saved {args.out}")


if __name__ == "__main__":
    main()
