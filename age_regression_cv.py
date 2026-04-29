"""
Brain-age prediction using L1-regularized linear regression (Lasso)
with 10-fold cross-validation on 3500 GMM features.

Inputs:
  - participants.tsv with at least columns: participant_id, age
  - gmm_features_weighted/ directory with per-subject .npy files

Outputs:
  - Printed per-fold and overall MAE, RMSE, R²
  - age_prediction_results.csv with true vs predicted age per subject
  - age_prediction_scatter.png
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', default='gmm_features_weighted')
    parser.add_argument('--participants', default='participants.tsv')
    parser.add_argument('--age-col', default='age')
    parser.add_argument('--id-col', default='participant_id')
    parser.add_argument('--k-folds', type=int, default=10)
    parser.add_argument('--n-alphas', type=int, default=100,
                        help="Number of alpha values for LassoCV grid")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # --- Load participants ---
    print(f"[*] Loading {args.participants}")
    df = pd.read_csv(args.participants, sep='\t')
    print(f"    {len(df)} rows, columns: {list(df.columns)}")

    assert args.age_col in df.columns, f"Column '{args.age_col}' not found"
    assert args.id_col in df.columns, f"Column '{args.id_col}' not found"

    # Drop rows with missing age
    df = df.dropna(subset=[args.age_col])
    df[args.age_col] = df[args.age_col].astype(float)

    # --- Match feature files to participants ---
    feature_files = sorted(os.listdir(args.features_dir))
    feature_files = [f for f in feature_files if f.endswith('.npy')]

    # Extract subject ID from filename: sub-XXXXX_preproc-... -> sub-XXXXX
    def extract_id(fname):
        return fname.split('_')[0]

    file_lookup = {extract_id(f): f for f in feature_files}

    # Match
    matched = []
    for _, row in df.iterrows():
        sub_id = str(row[args.id_col])
        # Handle cases where participants.tsv has IDs with or without 'sub-' prefix
        candidates = [sub_id, f"sub-{sub_id}"]
        for c in candidates:
            if c in file_lookup:
                matched.append({
                    'sub_id': c,
                    'age': row[args.age_col],
                    'file': file_lookup[c],
                })
                break

    print(f"[*] Matched {len(matched)} / {len(df)} participants to feature files")
    if len(matched) == 0:
        print("[!] No matches. Check ID format in participants.tsv vs filenames.")
        print(f"    First 3 participant IDs: {df[args.id_col].values[:3]}")
        print(f"    First 3 feature file IDs: {list(file_lookup.keys())[:3]}")
        return

    # --- Load feature matrix ---
    print(f"[*] Loading features...")
    X = np.array([
        np.load(os.path.join(args.features_dir, m['file']))
        for m in matched
    ])
    y = np.array([m['age'] for m in matched])
    sub_ids = [m['sub_id'] for m in matched]

    print(f"    X shape: {X.shape}")
    print(f"    Age range: {y.min():.1f} - {y.max():.1f} (mean {y.mean():.1f})")

    # --- 10-fold CV with Lasso ---
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    y_pred_all = np.zeros_like(y)
    fold_metrics = []

    print(f"\n[*] Running {args.k_folds}-fold CV with LassoCV...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # LassoCV picks the best alpha via internal CV on training set
        lasso = LassoCV(
            n_alphas=args.n_alphas,
            cv=5,
            random_state=args.seed,
            max_iter=10000,
        )
        lasso.fit(X_train_s, y_train)

        y_pred = lasso.predict(X_test_s)
        y_pred_all[test_idx] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        n_nonzero = np.sum(lasso.coef_ != 0)

        fold_metrics.append({'fold': fold + 1, 'mae': mae, 'rmse': rmse,
                             'r2': r2, 'alpha': lasso.alpha_, 'n_features': n_nonzero})

        print(f"   Fold {fold+1:2d}: MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"R²={r2:.3f}  alpha={lasso.alpha_:.4f}  "
              f"features={n_nonzero}/{X.shape[1]}")

    # --- Overall metrics ---
    overall_mae = mean_absolute_error(y, y_pred_all)
    overall_rmse = np.sqrt(mean_squared_error(y, y_pred_all))
    overall_r2 = r2_score(y, y_pred_all)

    print(f"\n{'='*60}")
    print(f"OVERALL:  MAE={overall_mae:.2f}  RMSE={overall_rmse:.2f}  R²={overall_r2:.3f}")
    print(f"{'='*60}")

    avg_features = np.mean([m['n_features'] for m in fold_metrics])
    print(f"Average non-zero features: {avg_features:.0f} / {X.shape[1]}")

    # --- Save results ---
    results_df = pd.DataFrame({
        'subject': sub_ids,
        'true_age': y,
        'predicted_age': y_pred_all,
        'error': y_pred_all - y,
    })
    results_df.to_csv('age_prediction_results.csv', index=False)
    print(f"\n[*] Saved age_prediction_results.csv")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Scatter
    ax = axes[0]
    ax.scatter(y, y_pred_all, s=8, alpha=0.4)
    lims = [y.min() - 2, y.max() + 2]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted Age')
    ax.set_title(f'Brain Age Prediction (10-fold CV)\n'
                 f'MAE={overall_mae:.2f}  RMSE={overall_rmse:.2f}  R²={overall_r2:.3f}')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    # Error vs age
    ax = axes[1]
    errors = y_pred_all - y
    ax.scatter(y, errors, s=8, alpha=0.4)
    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted − True Age')
    ax.set_title(f'Prediction Error vs Age\n'
                 f'mean error={errors.mean():.2f}  std={errors.std():.2f}')

    plt.tight_layout()
    plt.savefig('age_prediction_scatter.png', dpi=200)
    plt.close()
    print(f"[*] Saved age_prediction_scatter.png")


if __name__ == "__main__":
    main()
