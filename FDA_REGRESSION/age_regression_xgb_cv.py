"""
Brain-age prediction using XGBoost regression with 10-fold cross-validation.

Inputs:
  - train_labels/participants.tsv (columns: participant_id, age, ...)
  - train_vbm_coefs/*.npy        (one .npy per subject, shape (1, D) or (D,))

Outputs:
  - age_prediction_xgb_results.csv      true vs predicted age + error per subject
  - age_prediction_xgb_fold_metrics.csv per-fold MAE / RMSE / R^2 / best_iter
  - age_prediction_xgb_importance.csv   mean feature importance across folds
  - age_prediction_xgb_scatter.png      diagnostic plots
  - Console: per-fold and overall metrics

Notes:
  - Outer 10-fold CV. Inside each training fold, a 90/10 train/val split is used
    for early stopping. XGBoost's best_iteration on the val split caps boosting.
  - No feature scaling: trees are scale-invariant.
  - Defaults are conservative (max_depth=4, lr=0.05, subsample=0.8). With ~216
    features and ~3200 subjects, deeper trees overfit quickly.

Requires: xgboost >= 1.6 (early_stopping_rounds is a constructor arg).
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-prefix', default='age_prediction_xgb')
    # XGBoost hyperparameters
    parser.add_argument('--n-estimators', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--min-child-weight', type=float, default=5.0)
    parser.add_argument('--reg-lambda', type=float, default=1.0)
    parser.add_argument('--early-stopping-rounds', type=int, default=50)
    parser.add_argument('--val-frac', type=float, default=0.1,
                        help='Fraction of each train fold reserved for early stopping.')
    parser.add_argument('--n-jobs', type=int, default=-1)
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
        X_list.append(np.asarray(arr).squeeze())  # (1, D) -> (D,)
    X = np.vstack(X_list).astype(np.float32)
    y = np.array([m['age'] for m in matched], dtype=float)
    sub_ids = [m['sub_id'] for m in matched]

    print(f"    X shape: {X.shape}")
    print(f"    Age range: {y.min():.1f} - {y.max():.1f} (mean {y.mean():.1f})")

    # --- 10-fold CV ---
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    y_pred_all = np.full_like(y, fill_value=np.nan, dtype=float)
    fold_metrics = []
    importances = np.zeros(X.shape[1], dtype=float)

    print(f"\n[*] Running {args.k_folds}-fold CV with XGBoost...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr_full, X_te = X[train_idx], X[test_idx]
        y_tr_full, y_te = y[train_idx], y[test_idx]

        # Inner train/val split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full,
            test_size=args.val_frac,
            random_state=args.seed + fold,
        )

        model = XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_weight=args.min_child_weight,
            reg_lambda=args.reg_lambda,
            objective='reg:squarederror',
            eval_metric='mae',
            tree_method='hist',
            early_stopping_rounds=args.early_stopping_rounds,
            n_jobs=args.n_jobs,
            random_state=args.seed + fold,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_hat = model.predict(X_te)
        y_pred_all[test_idx] = y_hat

        mae = mean_absolute_error(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        r2 = r2_score(y_te, y_hat)
        best_iter = int(getattr(model, 'best_iteration', args.n_estimators - 1))

        fold_metrics.append({
            'fold': fold + 1, 'n_test': len(test_idx),
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'best_iteration': best_iter,
        })
        print(f"   Fold {fold+1:2d}: MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"R²={r2:.3f}  best_iter={best_iter}")

        # Accumulate feature importances (gain-based, default)
        importances += model.feature_importances_

    importances /= args.k_folds

    # Warn if early stopping is hitting the n_estimators ceiling
    best_iters = np.array([m['best_iteration'] for m in fold_metrics])
    if np.any(best_iters >= args.n_estimators - 1):
        print("[!] best_iteration is at the n_estimators ceiling in some folds — "
              "increase --n-estimators.")

    # --- Overall metrics ---
    overall_mae = mean_absolute_error(y, y_pred_all)
    overall_rmse = np.sqrt(mean_squared_error(y, y_pred_all))
    overall_r2 = r2_score(y, y_pred_all)
    print("\n" + "=" * 60)
    print(f"OVERALL:  MAE={overall_mae:.2f}  RMSE={overall_rmse:.2f}  "
          f"R²={overall_r2:.3f}")
    print("=" * 60)

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

    imp_df = pd.DataFrame({
        'feature_index': np.arange(X.shape[1]),
        'mean_importance': importances,
    }).sort_values('mean_importance', ascending=False)
    imp_path = f"{args.out_prefix}_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"[*] Saved {imp_path}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(y, y_pred_all, s=8, alpha=0.4)
    lims = [y.min() - 2, y.max() + 2]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted Age')
    ax.set_title(f'Brain Age Prediction — XGBoost ({args.k_folds}-fold CV)\n'
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
