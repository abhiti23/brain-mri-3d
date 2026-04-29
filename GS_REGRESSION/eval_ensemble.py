"""
50/50 ensemble of LassoCV and XGBoost validation predictions.

Run AFTER train_eval_lasso.py and train_eval_xgboost.py.

Inputs:
  - lasso_final/lasso_val_predictions.csv
  - xgb_final/xgb_val_predictions.csv

Outputs (in --out-dir):
  - ensemble_val_predictions.csv
  - ensemble_comparison.png
  - ensemble_metrics.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lasso', default='lasso_final/lasso_val_predictions.csv')
    parser.add_argument('--xgb', default='xgb_final/xgb_val_predictions.csv')
    parser.add_argument('--out-dir', default='ensemble_final')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    lasso = pd.read_csv(args.lasso)
    xgb = pd.read_csv(args.xgb)

    merged = lasso.merge(xgb, on='subject', suffixes=('_lasso', '_xgb'))
    print(f"[*] {len(merged)} matched validation subjects")

    y_true = merged['true_age_lasso'].values
    y_lasso = merged['predicted_age_lasso'].values
    y_xgb = merged['predicted_age_xgb'].values
    y_ens = 0.5 * y_lasso + 0.5 * y_xgb

    error_corr = np.corrcoef(merged['error_lasso'], merged['error_xgb'])[0, 1]

    results = {}
    for name, y_pred in [('LassoCV', y_lasso), ('XGBoost', y_xgb), ('Ensemble', y_ens)]:
        results[name] = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
        }

    print(f"\n    Error correlation: {error_corr:.3f}")
    print(f"\n    {'Model':<12} {'MAE':>6} {'RMSE':>6} {'R²':>6}")
    print(f"    {'-'*34}")
    for name, m in results.items():
        print(f"    {name:<12} {m['mae']:>6.2f} {m['rmse']:>6.2f} {m['r2']:>6.3f}")

    # --- Save predictions ---
    pred_df = pd.DataFrame({
        'subject': merged['subject'],
        'true_age': y_true,
        'pred_lasso': y_lasso,
        'pred_xgb': y_xgb,
        'pred_ensemble': y_ens,
        'error_lasso': y_lasso - y_true,
        'error_xgb': y_xgb - y_true,
        'error_ensemble': y_ens - y_true,
    })
    pred_path = os.path.join(args.out_dir, 'ensemble_val_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\n[*] Saved {pred_path}")

    # --- Metrics file ---
    metrics_path = os.path.join(args.out_dir, 'ensemble_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"error_correlation: {error_corr:.4f}\n")
        for name, m in results.items():
            f.write(f"{name}_mae: {m['mae']:.4f}\n")
            f.write(f"{name}_rmse: {m['rmse']:.4f}\n")
            f.write(f"{name}_r2: {m['r2']:.4f}\n")
    print(f"[*] Saved {metrics_path}")

    # --- Plot: 3 scatter plots + 1 head-to-head, each as a separate file ---
    age_lims = [y_true.min() - 3, y_true.max() + 3]

    scatter_specs = [
        (y_lasso, 'LassoCV',         'scatter_lasso.png'),
        (y_xgb,   'XGBoost',         'scatter_xgboost.png'),
        (y_ens,   'Ensemble (50/50)', 'scatter_ensemble.png'),
    ]

    for y_pred, name, fname in scatter_specs:
        m = results[name.split(' ')[0]]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, s=8, alpha=0.4, c='steelblue')
        ax.plot(age_lims, age_lims, 'r--', lw=1)
        ax.set_xlim(age_lims); ax.set_ylim(age_lims)
        ax.set_xlabel('True Age'); ax.set_ylabel('Predicted Age')
        ax.set_title(f"{name} — Test Set\nMAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R²={m['r2']:.3f}")
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved {fname}")

    # Head-to-head: per-subject error comparison
    lasso_ae = np.abs(y_lasso - y_true)
    ens_ae   = np.abs(y_ens - y_true)
    xgb_ae   = np.abs(y_xgb - y_true)
    max_ae   = max(lasso_ae.max(), xgb_ae.max(), ens_ae.max())
    ens_wins = (ens_ae < xgb_ae).sum()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xgb_ae, ens_ae, s=8, alpha=0.4, c='steelblue')
    ax.plot([0, max_ae], [0, max_ae], 'r--', lw=1)
    ax.set_xlabel('XGBoost |error| (years)')
    ax.set_ylabel('Ensemble |error| (years)')
    ax.set_title(f'Ensemble vs XGBoost — Test Set\n'
                 f'Ensemble wins: {ens_wins}/{len(merged)} ({ens_wins/len(merged)*100:.1f}%)')
    ax.set_xlim(0, max_ae + 1); ax.set_ylim(0, max_ae + 1)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'scatter_ensemble_vs_xgb.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved scatter_ensemble_vs_xgb.png")


if __name__ == "__main__":
    main()
