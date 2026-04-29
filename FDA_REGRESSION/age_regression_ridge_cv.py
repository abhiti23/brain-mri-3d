"""
Brain-age prediction using L2-regularized linear regression (Ridge)
with 10-fold cross-validation.

Inputs:
  - train_labels/participants.tsv (columns: participant_id, age, ...)
  - train_vbm_coefs/*.npy        (one .npy per subject, shape (1, D) or (D,))

Outputs:
  - age_prediction_ridge_results.csv      true vs predicted age + error per subject
  - age_prediction_ridge_fold_metrics.csv per-fold MAE / RMSE / R^2 / alpha
  - age_prediction_ridge_scatter.png      diagnostic plots
  - Console: per-fold and overall metrics

Notes:
  - Outer 10-fold CV; inner alpha selection via RidgeCV's efficient
    leave-one-out generalized CV (closed-form, very fast for ridge).
  - Features standardized with StandardScaler fit on the training fold only.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


SUB_RE = re.compile(r'(sub-[A-Za-z0-9]+)')


def extract_sub_id(fname):
    """Pull the 'sub-XXXX' token out of a filename, regardless of position."""
    m = SUB_RE.search(fname)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', default='train_vbm_coefs')
    parser.add_argument('--participants', default='train_labels/participants.tsv')
    parser.add_argument('--age-col', default='age')
    parser.add_argument('--id-col', default='participant_id')
    parser.add_argument('--k-folds', type=int, default=10)
    parser.add_argument('--n-alphas', type=int, default=100,
                        help='Number of alpha values to try (log-spaced).')
    parser.add_argument('--alpha-min', type=float, default=1e-3)
    parser.add_argument('--alpha-max', type=float, default=1e4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-prefix', default='age_prediction_ridge')
    args = parser.parse_args()

    # --- Load participants ---
    print(f"[*] Loading {args.participants}")
    df = pd.read_csv(args.participants, sep='\t')
    print(f"    {len(df)} rows, columns: {list(df.columns)}")
    assert args.age_col in df.columns, f"Column '{args.age_col}' not found"
    assert args.id_col in df.columns, f"Column '{args.id_col}' not found"

    df = df.dropna(subset=[args.age_col]).copy()
    df[args.age_col] = df[args.age_col].astype(float)

    # --- Index feature files by sub-id ---
    feature_files = [f for f in os.listdir(args.features_dir) if f.endswith('.npy')]
    file_lookup = {}
    for f in feature_files:
        sid = extract_sub_id(f)
        if sid is None:
            continue
        if sid in file_lookup:
            print(f"[!] Multiple files for {sid}; keeping {file_lookup[sid]}, "
                  f"ignoring {f}")
            continue
        file_lookup[sid] = f
    print(f"[*] Indexed {len(file_lookup)} feature files in {args.features_dir}")

    # --- Match participants -> files ---
    matched = []
    for _, row in df.iterrows():
        raw = str(row[args.id_col])
        candidates = [raw, f"sub-{raw}"] if not raw.startswith('sub-') else [raw]
        for c in candidates:
            if c in file_lookup:
                matched.append({
                    'sub_id': c,
                    'age': row[args.age_col],
                    'file': file_lookup[c],
                })
                break

    print(f"[*] Matched {len(matched)} / {len(df)} participants to feature files")
    if not matched:
        print("[!] No matches. Inspect ID formats:")
        print(f"    First 3 participant IDs : {df[args.id_col].head(3).tolist()}")
        print(f"    First 3 feature sub-ids : {list(file_lookup.keys())[:3]}")
        return

    # --- Load feature matrix ---
    print("[*] Loading features...")
    X_list = []
    for m in matched:
        arr = np.load(os.path.join(args.features_dir, m['file']))
        X_list.append(np.asarray(arr).squeeze())  # handle (1, D) -> (D,)
    X = np.vstack(X_list)
    y = np.array([m['age'] for m in matched], dtype=float)
    sub_ids = [m['sub_id'] for m in matched]

    print(f"    X shape: {X.shape}")
    print(f"    Age range: {y.min():.1f} - {y.max():.1f} (mean {y.mean():.1f})")

    # --- Alpha grid (log-spaced) ---
    alphas = np.logspace(np.log10(args.alpha_min),
                         np.log10(args.alpha_max),
                         args.n_alphas)

    # --- 10-fold CV with RidgeCV (nested) ---
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    y_pred_all = np.full_like(y, fill_value=np.nan, dtype=float)
    fold_metrics = []

    print(f"\n[*] Running {args.k_folds}-fold CV with RidgeCV (GCV)...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # cv=None => efficient leave-one-out via generalized CV (closed form)
        ridge = RidgeCV(alphas=alphas, cv=None, scoring=None)
        ridge.fit(X_tr_s, y_tr)

        y_hat = ridge.predict(X_te_s)
        y_pred_all[test_idx] = y_hat

        mae = mean_absolute_error(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        r2 = r2_score(y_te, y_hat)
        coef_l2 = float(np.linalg.norm(ridge.coef_))

        fold_metrics.append({
            'fold': fold + 1, 'n_test': len(test_idx),
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'alpha': ridge.alpha_, 'coef_l2_norm': coef_l2,
        })
        print(f"   Fold {fold+1:2d}: MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"R²={r2:.3f}  alpha={ridge.alpha_:.4g}  "
              f"||w||₂={coef_l2:.2f}")

    # --- Overall metrics ---
    overall_mae = mean_absolute_error(y, y_pred_all)
    overall_rmse = np.sqrt(mean_squared_error(y, y_pred_all))
    overall_r2 = r2_score(y, y_pred_all)
    print("\n" + "=" * 60)
    print(f"OVERALL:  MAE={overall_mae:.2f}  RMSE={overall_rmse:.2f}  "
          f"R²={overall_r2:.3f}")
    print("=" * 60)

    # Sanity check on the alpha grid: warn if RidgeCV is railing against the edge.
    chosen = np.array([m['alpha'] for m in fold_metrics])
    if np.any(chosen <= alphas[1]) or np.any(chosen >= alphas[-2]):
        print("[!] Selected alpha is at the edge of the search grid in some "
              "folds — consider widening --alpha-min / --alpha-max.")

    # --- Save per-subject errors ---
    results_df = pd.DataFrame({
        'subject': sub_ids,
        'true_age': y,
        'predicted_age': y_pred_all,
        'error': y_pred_all - y,
        'abs_error': np.abs(y_pred_all - y),
    })
    results_path = f"{args.out_prefix}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[*] Saved {results_path}")

    fold_path = f"{args.out_prefix}_fold_metrics.csv"
    pd.DataFrame(fold_metrics).to_csv(fold_path, index=False)
    print(f"[*] Saved {fold_path}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(y, y_pred_all, s=8, alpha=0.4)
    lims = [y.min() - 2, y.max() + 2]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted Age')
    ax.set_title(f'Brain Age Prediction — Ridge ({args.k_folds}-fold CV)\n'
                 f'MAE={overall_mae:.2f}  RMSE={overall_rmse:.2f}  '
                 f'R²={overall_r2:.3f}')
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')

    ax = axes[1]
    errors = y_pred_all - y
    ax.scatter(y, errors, s=8, alpha=0.4)
    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted − True Age')
    ax.set_title(f'Prediction Error vs Age\n'
                 f'mean error={errors.mean():.2f}  std={errors.std():.2f}')
    plt.tight_layout()
    fig_path = f"{args.out_prefix}_scatter.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[*] Saved {fig_path}")


if __name__ == "__main__":
    main()
