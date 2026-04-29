"""
Compare ridge vs XGBoost age-prediction errors and assess ensemble potential.

Inputs:
  - ridge results CSV  (cols: subject, true_age, predicted_age, error, abs_error)
  - xgb   results CSV  (same schema)

Outputs:
  - error_comparison_summary.txt   plain-text report of all metrics
  - error_correlation_plots.png    4-panel diagnostic figure
  - ensemble_predictions.csv       per-subject predictions from each model + ensembles

Logic:
  1. Merge on subject.
  2. Compute correlation of signed errors and absolute errors (Pearson + Spearman).
  3. Detrend each model's errors against true age (remove regression-to-the-mean
     bias) and recompute correlation on the residuals. This is the part that
     actually matters for ensemble benefit.
  4. Build the 50/50 average ensemble and report its MAE/RMSE/R².
  5. Find the optimal convex weight w* in [0,1] that minimizes ensemble MSE on
     these out-of-fold predictions. (Caveat printed: this weight is fit on the
     same data we evaluate on — slight leakage. For an unbiased estimate of
     ensemble gain you'd nest the weight selection in a second CV layer.)
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def metrics(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': r2_score(y_true, y_pred),
    }


def detrend_against_age(errors, ages):
    """Remove linear trend in error w.r.t. true age (the regression-to-mean bias).
    Returns residuals."""
    A = np.vstack([ages, np.ones_like(ages)]).T
    coef, *_ = np.linalg.lstsq(A, errors, rcond=None)
    fitted = A @ coef
    return errors - fitted, coef  # coef = [slope, intercept]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ridge', default='age_prediction_ridge_results.csv')
    parser.add_argument('--xgb', default='age_prediction_xgb_results.csv')
    parser.add_argument('--out-prefix', default='error_comparison')
    args = parser.parse_args()

    ridge = pd.read_csv(args.ridge)
    xgb = pd.read_csv(args.xgb)

    merged = ridge.merge(
        xgb, on='subject', suffixes=('_ridge', '_xgb'), validate='one_to_one'
    )
    # Sanity: true_age should agree
    age_disagreement = (merged['true_age_ridge'] - merged['true_age_xgb']).abs().max()
    assert age_disagreement < 1e-9, "true_age differs between files — check merge."
    y = merged['true_age_ridge'].values
    e_r = merged['error_ridge'].values         # signed: pred - true
    e_x = merged['error_xgb'].values
    p_r = merged['predicted_age_ridge'].values
    p_x = merged['predicted_age_xgb'].values
    n = len(merged)

    # --- Per-model metrics (sanity check that the CSVs match prior console output) ---
    m_ridge = metrics(y, p_r)
    m_xgb = metrics(y, p_x)

    # --- Raw error correlations ---
    r_signed_pearson, p_signed_pearson = pearsonr(e_r, e_x)
    r_signed_spearman, p_signed_spearman = spearmanr(e_r, e_x)
    r_abs_pearson, p_abs_pearson = pearsonr(np.abs(e_r), np.abs(e_x))
    r_abs_spearman, p_abs_spearman = spearmanr(np.abs(e_r), np.abs(e_x))

    # --- Detrended (residual after removing age-related bias) ---
    res_r, slope_r = detrend_against_age(e_r, y)
    res_x, slope_x = detrend_against_age(e_x, y)
    r_detrended, p_detrended = pearsonr(res_r, res_x)

    # --- 50/50 average ensemble ---
    p_avg = 0.5 * (p_r + p_x)
    m_avg = metrics(y, p_avg)

    # --- Optimal convex weight (fit on same data — caveat about leakage) ---
    # min over w of E[(y - (w*p_r + (1-w)*p_x))^2]
    # Closed form: w* = cov(p_x - y, p_x - p_r) / var(p_r - p_x)
    d = p_r - p_x
    var_d = float(np.var(d))
    if var_d > 1e-12:
        w_unconstrained = float(np.dot(p_x - y, p_x - p_r) / np.sum(d ** 2)) * (n / n)
        # Re-derive cleanly:
        # ||y - (w p_r + (1-w) p_x)||^2 = ||(y - p_x) - w(p_r - p_x)||^2
        # w* = <(p_r - p_x), (y - p_x)> / ||p_r - p_x||^2
        w_unconstrained = float(np.dot(p_r - p_x, y - p_x) / np.sum(d ** 2))
    else:
        w_unconstrained = 0.5
    w_star = float(np.clip(w_unconstrained, 0.0, 1.0))
    p_opt = w_star * p_r + (1 - w_star) * p_x
    m_opt = metrics(y, p_opt)

    # --- Variance reduction prediction from theory ---
    # Var of the average error: var( (e_r+e_x)/2 ) = (var(e_r)+var(e_x)+2*cov)/4
    var_r, var_x = float(np.var(e_r)), float(np.var(e_x))
    cov_rx = float(np.cov(e_r, e_x, ddof=0)[0, 1])
    var_avg_err = (var_r + var_x + 2 * cov_rx) / 4.0
    var_best_single = min(var_r, var_x)
    pct_var_reduction = 100.0 * (1 - var_avg_err / var_best_single)

    # --- Save per-subject ensemble file ---
    out_pred = pd.DataFrame({
        'subject': merged['subject'],
        'true_age': y,
        'pred_ridge': p_r,
        'pred_xgb': p_x,
        'pred_avg': p_avg,
        'pred_optimal': p_opt,
        'err_ridge': e_r,
        'err_xgb': e_x,
        'err_avg': p_avg - y,
        'err_optimal': p_opt - y,
    })
    pred_path = 'ensemble_predictions.csv'
    out_pred.to_csv(pred_path, index=False)

    # --- Summary text ---
    lines = []
    lines.append("=" * 64)
    lines.append("RIDGE vs XGBOOST — ERROR CORRELATION & ENSEMBLE ANALYSIS")
    lines.append("=" * 64)
    lines.append(f"N subjects matched: {n}")
    lines.append("")
    lines.append("Per-model performance (out-of-fold predictions):")
    lines.append(f"  Ridge   : MAE={m_ridge['mae']:.3f}  RMSE={m_ridge['rmse']:.3f}  R²={m_ridge['r2']:.3f}")
    lines.append(f"  XGBoost : MAE={m_xgb['mae']:.3f}  RMSE={m_xgb['rmse']:.3f}  R²={m_xgb['r2']:.3f}")
    lines.append("")
    lines.append("Error correlations (raw):")
    lines.append(f"  Signed errors  Pearson  r = {r_signed_pearson:+.3f}  (p = {p_signed_pearson:.2e})")
    lines.append(f"  Signed errors  Spearman ρ = {r_signed_spearman:+.3f}  (p = {p_signed_spearman:.2e})")
    lines.append(f"  |Errors|       Pearson  r = {r_abs_pearson:+.3f}  (p = {p_abs_pearson:.2e})")
    lines.append(f"  |Errors|       Spearman ρ = {r_abs_spearman:+.3f}  (p = {p_abs_spearman:.2e})")
    lines.append("")
    lines.append("Error vs age slopes (regression-to-the-mean indicator):")
    lines.append(f"  Ridge   : err = {slope_r[0]:+.3f} * age + {slope_r[1]:+.2f}")
    lines.append(f"  XGBoost : err = {slope_x[0]:+.3f} * age + {slope_x[1]:+.2f}")
    lines.append("  (negative slope = older subjects predicted too young, classic RTM)")
    lines.append("")
    lines.append("Detrended error correlation (this is what matters for ensembling):")
    lines.append(f"  Pearson r = {r_detrended:+.3f}  (p = {p_detrended:.2e})")
    lines.append("  Lower |r| ⇒ more independent signal ⇒ more ensemble headroom.")
    lines.append("")
    lines.append("Ensemble performance:")
    lines.append(f"  50/50 average    : MAE={m_avg['mae']:.3f}  RMSE={m_avg['rmse']:.3f}  R²={m_avg['r2']:.3f}")
    lines.append(f"  Optimal convex   : MAE={m_opt['mae']:.3f}  RMSE={m_opt['rmse']:.3f}  R²={m_opt['r2']:.3f}")
    lines.append(f"  Optimal weight w*= {w_star:.3f}  (ridge weight; xgb weight = {1-w_star:.3f})")
    lines.append("")
    lines.append("Theoretical signed-error variance:")
    lines.append(f"  Var(err_ridge)        = {var_r:.3f}")
    lines.append(f"  Var(err_xgb)          = {var_x:.3f}")
    lines.append(f"  Cov(err_ridge, err_xgb)= {cov_rx:.3f}")
    lines.append(f"  Var(err_avg)          = {var_avg_err:.3f}")
    lines.append(f"  Variance reduction vs best single model: {pct_var_reduction:+.1f}%")
    lines.append("")
    lines.append("Caveats:")
    lines.append("  • Optimal weight w* is fit on the same out-of-fold predictions used")
    lines.append("    to evaluate it. The reported optimal-ensemble metrics are slightly")
    lines.append("    optimistic. For an unbiased estimate, do nested CV for w.")
    lines.append("  • If 50/50 average already improves over the better single model,")
    lines.append("    the gain is robust — no weight tuning needed.")
    lines.append("  • If 50/50 is worse than the better single model but the optimal-")
    lines.append("    weight ensemble is better, the gain depends on tuning w and is")
    lines.append("    less reliable. Validate via nested CV before trusting it.")
    lines.append("=" * 64)

    summary = "\n".join(lines)
    print(summary)
    with open(f"{args.out_prefix}_summary.txt", 'w') as f:
        f.write(summary + "\n")
    print(f"\n[*] Saved {args.out_prefix}_summary.txt")
    print(f"[*] Saved {pred_path}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # (1) Signed error scatter
    ax = axes[0, 0]
    ax.scatter(e_r, e_x, s=8, alpha=0.4)
    lim = max(np.abs(e_r).max(), np.abs(e_x).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('Ridge signed error')
    ax.set_ylabel('XGBoost signed error')
    ax.set_title(f'Signed errors (Pearson r={r_signed_pearson:+.3f})')

    # (2) Absolute error scatter
    ax = axes[0, 1]
    ax.scatter(np.abs(e_r), np.abs(e_x), s=8, alpha=0.4)
    lim = max(np.abs(e_r).max(), np.abs(e_x).max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
    ax.set_xlabel('Ridge |error|')
    ax.set_ylabel('XGBoost |error|')
    ax.set_title(f'Absolute errors (Pearson r={r_abs_pearson:+.3f})')

    # (3) Detrended scatter
    ax = axes[1, 0]
    ax.scatter(res_r, res_x, s=8, alpha=0.4)
    lim = max(np.abs(res_r).max(), np.abs(res_x).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('Ridge detrended residual')
    ax.set_ylabel('XGBoost detrended residual')
    ax.set_title(f'Detrended errors (Pearson r={r_detrended:+.3f})\n'
                 'this is the part ensembling can exploit')

    # (4) MAE comparison bar chart
    ax = axes[1, 1]
    labels = ['Ridge', 'XGBoost', '50/50 avg', f'Optimal\n(w={w_star:.2f})']
    maes = [m_ridge['mae'], m_xgb['mae'], m_avg['mae'], m_opt['mae']]
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    bars = ax.bar(labels, maes, color=colors)
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('MAE (years)')
    ax.set_title('Single models vs ensembles')
    ax.set_ylim(0, max(maes) * 1.15)

    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_plots.png", dpi=200)
    plt.close()
    print(f"[*] Saved {args.out_prefix}_plots.png")


if __name__ == "__main__":
    main()
