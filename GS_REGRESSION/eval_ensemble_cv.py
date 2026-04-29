"""
50/50 and greedy-optimal ensemble of LassoCV and XGBoost CV predictions.

Inputs:
  - LASSOCV_RESULTS/age_prediction_results.csv
  - XGBOOST_RESULTS_CV/xgb_age_prediction_results.csv

Outputs (in --out-dir):
  - ensemble_val_predictions.csv
  - ensemble_metrics.txt
  - scatter_lasso.png
  - scatter_xgboost.png
  - scatter_ensemble_50_50.png
  - scatter_ensemble_optimal.png
  - scatter_ensemble_vs_xgb.png
  - error_lasso.png
  - error_xgboost.png
  - error_ensemble_50_50.png
  - error_ensemble_optimal.png
  - weight_sweep.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def scatter_plot(y_true, y_pred, title, filename, age_lims):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4, c='steelblue')
    ax.plot(age_lims, age_lims, 'r--', lw=1)
    ax.set_xlim(age_lims); ax.set_ylim(age_lims)
    ax.set_xlabel('True Age'); ax.set_ylabel('Predicted Age')
    ax.set_title(f"{title}\nMAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"   -> Saved {filename}")


def error_plot(y_true, y_pred, title, filename, age_lims):
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(y_true, errors, s=8, alpha=0.4, c='steelblue')
    ax.axhline(0, color='r', ls='--', lw=1)
    z = np.polyfit(y_true, errors, 1)
    x_line = np.linspace(age_lims[0], age_lims[1], 100)
    ax.plot(x_line, np.polyval(z, x_line), 'orange', lw=1.5, label=f'slope={z[0]:.3f}')
    ax.set_xlim(age_lims)
    ax.set_xlabel('True Age'); ax.set_ylabel('Predicted − True Age')
    ax.set_title(f"{title}\nmean={errors.mean():.2f}  std={errors.std():.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"   -> Saved {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lasso', default='LASSOCV_RESULTS/age_prediction_results.csv')
    parser.add_argument('--xgb', default='XGBCV_RESULTS/xgb_age_prediction_results.csv')
    parser.add_argument('--out-dir', default='ensemble_cv')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    lasso = pd.read_csv(args.lasso)
    xgb_df = pd.read_csv(args.xgb)

    merged = lasso.merge(xgb_df, on='subject', suffixes=('_lasso', '_xgb'))
    print(f"[*] {len(merged)} matched subjects")

    y_true = merged['true_age_lasso'].values
    y_lasso = merged['predicted_age_lasso'].values
    y_xgb = merged['predicted_age_xgb'].values

    error_corr = np.corrcoef(merged['error_lasso'], merged['error_xgb'])[0, 1]
    print(f"    Error correlation: {error_corr:.3f}")

    # --- 50/50 ensemble ---
    y_ens50 = 0.5 * y_lasso + 0.5 * y_xgb

    # --- Greedy optimal weight ---
    weights = np.arange(0.0, 1.001, 0.01)
    maes = np.array([mean_absolute_error(y_true, w * y_lasso + (1 - w) * y_xgb)
                     for w in weights])
    best_w = weights[np.argmin(maes)]
    best_mae = maes.min()
    y_ens_opt = best_w * y_lasso + (1 - best_w) * y_xgb

    print(f"    Optimal weight: {best_w:.2f} Lasso + {1-best_w:.2f} XGBoost")

    # --- Collect all metrics ---
    models = {
        'LassoCV': y_lasso,
        'XGBoost': y_xgb,
        'Ensemble (50/50)': y_ens50,
        f'Ensemble ({best_w:.2f}/{1-best_w:.2f})': y_ens_opt,
    }

    results = {}
    for name, y_pred in models.items():
        results[name] = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
        }

    print(f"\n    {'Model':<28} {'MAE':>6} {'RMSE':>6} {'R²':>6}")
    print(f"    {'-'*50}")
    for name, m in results.items():
        print(f"    {name:<28} {m['mae']:>6.2f} {m['rmse']:>6.2f} {m['r2']:>6.3f}")

    # --- Save predictions ---
    pred_df = pd.DataFrame({
        'subject': merged['subject'],
        'true_age': y_true,
        'pred_lasso': y_lasso,
        'pred_xgb': y_xgb,
        'pred_ensemble_50_50': y_ens50,
        'pred_ensemble_optimal': y_ens_opt,
    })
    pred_path = os.path.join(args.out_dir, 'ensemble_val_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\n[*] Saved {pred_path}")

    # --- Save metrics ---
    metrics_path = os.path.join(args.out_dir, 'ensemble_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"error_correlation: {error_corr:.4f}\n")
        f.write(f"optimal_lasso_weight: {best_w:.2f}\n")
        f.write(f"optimal_xgb_weight: {1-best_w:.2f}\n\n")
        for name, m in results.items():
            f.write(f"{name}:\n")
            f.write(f"  mae: {m['mae']:.4f}\n")
            f.write(f"  rmse: {m['rmse']:.4f}\n")
            f.write(f"  r2: {m['r2']:.4f}\n\n")
    print(f"[*] Saved {metrics_path}")

    # --- Individual plots ---
    age_lims = [y_true.min() - 3, y_true.max() + 3]

    print("\n[*] Generating plots...")
    scatter_plot(y_true, y_lasso, 'LassoCV — CV',
                 os.path.join(args.out_dir, 'scatter_lasso.png'), age_lims)
    scatter_plot(y_true, y_xgb, 'XGBoost — CV',
                 os.path.join(args.out_dir, 'scatter_xgboost.png'), age_lims)
    scatter_plot(y_true, y_ens50, 'Ensemble (50/50) — CV',
                 os.path.join(args.out_dir, 'scatter_ensemble_50_50.png'), age_lims)
    scatter_plot(y_true, y_ens_opt, f'Ensemble ({best_w:.2f}/{1-best_w:.2f}) — CV',
                 os.path.join(args.out_dir, 'scatter_ensemble_optimal.png'), age_lims)

    error_plot(y_true, y_lasso, 'LassoCV Error — CV',
               os.path.join(args.out_dir, 'error_lasso.png'), age_lims)
    error_plot(y_true, y_xgb, 'XGBoost Error — CV',
               os.path.join(args.out_dir, 'error_xgboost.png'), age_lims)
    error_plot(y_true, y_ens50, 'Ensemble (50/50) Error — CV',
               os.path.join(args.out_dir, 'error_ensemble_50_50.png'), age_lims)
    error_plot(y_true, y_ens_opt, f'Ensemble ({best_w:.2f}/{1-best_w:.2f}) Error — CV',
               os.path.join(args.out_dir, 'error_ensemble_optimal.png'), age_lims)

    # --- Head-to-head: XGBoost vs 50/50 ensemble ---
    xgb_ae = np.abs(y_xgb - y_true)
    ens_ae = np.abs(y_ens50 - y_true)
    max_ae = max(xgb_ae.max(), ens_ae.max())
    ens_wins = (ens_ae < xgb_ae).sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xgb_ae, ens_ae, s=8, alpha=0.4, c='steelblue')
    ax.plot([0, max_ae], [0, max_ae], 'r--', lw=1)
    ax.set_xlabel('XGBoost |error| (years)')
    ax.set_ylabel('Ensemble |error| (years)')
    ax.set_title(f'Ensemble vs XGBoost per-subject — CV\n'
                 f'Ensemble wins: {ens_wins}/{len(merged)} ({ens_wins/len(merged)*100:.1f}%)')
    ax.set_xlim(0, max_ae + 1); ax.set_ylim(0, max_ae + 1)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'scatter_ensemble_vs_xgb.png'), dpi=200)
    plt.close()
    print(f"   -> Saved scatter_ensemble_vs_xgb.png")

    # --- Weight sweep plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(weights, maes, 'steelblue', lw=2)
    ax.axvline(0.5, color='gray', ls='--', lw=1,
               label=f'50/50: MAE={results["Ensemble (50/50)"]["mae"]:.2f}')
    ax.axvline(best_w, color='red', ls='--', lw=1,
               label=f'optimal w={best_w:.2f}: MAE={best_mae:.2f}')
    ax.axhline(results['XGBoost']['mae'], color='green', ls=':', lw=1,
               label=f'XGBoost alone: MAE={results["XGBoost"]["mae"]:.2f}')
    ax.axhline(results['LassoCV']['mae'], color='orange', ls=':', lw=1,
               label=f'Lasso alone: MAE={results["LassoCV"]["mae"]:.2f}')
    ax.set_xlabel('Lasso weight (XGBoost weight = 1 − Lasso weight)')
    ax.set_ylabel('MAE (years)')
    ax.set_title('Ensemble Weight Sweep — CV')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'weight_sweep.png'), dpi=200)
    plt.close()
    print(f"   -> Saved weight_sweep.png")

    # --- Caveat ---
    print(f"\n    NOTE: The optimal weight was found by grid search on the")
    print(f"    CV predictions. The 50/50 ensemble is the fully honest number.")


if __name__ == "__main__":
    main()
