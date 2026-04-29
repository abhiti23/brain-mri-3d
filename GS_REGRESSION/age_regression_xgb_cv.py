"""
Brain-age prediction using XGBoost regression with 10-fold cross-validation.

Inputs:
  - participants.tsv with participant_id and age columns
  - gmm_features_weighted/ directory with per-subject .npy files

Outputs (saved to --out-dir):
  - xgb_age_prediction_results.csv
  - xgb_age_prediction_scatter.png
  - xgb_feature_importance.png
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', default='gmm_features_weighted')
    parser.add_argument('--participants', default='./train_labels/participants.tsv')
    parser.add_argument('--age-col', default='age')
    parser.add_argument('--id-col', default='participant_id')
    parser.add_argument('--k-folds', type=int, default=10)
    parser.add_argument('--out-dir', default='XGBCV_RESULTS')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load participants ---
    print(f"[*] Loading {args.participants}")
    df = pd.read_csv(args.participants, sep='\t')
    print(f"    {len(df)} rows, columns: {list(df.columns)}")

    assert args.age_col in df.columns, f"Column '{args.age_col}' not found"
    assert args.id_col in df.columns, f"Column '{args.id_col}' not found"

    df = df.dropna(subset=[args.age_col])
    df[args.age_col] = df[args.age_col].astype(float)

    # --- Match feature files to participants ---
    feature_files = sorted(os.listdir(args.features_dir))
    feature_files = [f for f in feature_files if f.endswith('.npy')]

    def extract_id(fname):
        return fname.split('_')[0]

    file_lookup = {extract_id(f): f for f in feature_files}

    matched = []
    for _, row in df.iterrows():
        sub_id = str(row[args.id_col])
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

    # --- 10-fold CV with XGBoost ---
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    y_pred_all = np.zeros_like(y)
    fold_metrics = []
    feature_importance_acc = np.zeros(X.shape[1])

    xgb_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,       # L1
        'reg_lambda': 1.0,      # L2
        'random_state': args.seed,
        'n_jobs': -1,
    }

    print(f"\n[*] Running {args.k_folds}-fold CV with XGBoost...")
    print(f"    params: n_estimators={xgb_params['n_estimators']}, "
          f"max_depth={xgb_params['max_depth']}, lr={xgb_params['learning_rate']}")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        y_pred_all[test_idx] = y_pred

        feature_importance_acc += model.feature_importances_

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else xgb_params['n_estimators']

        fold_metrics.append({'fold': fold + 1, 'mae': mae, 'rmse': rmse, 'r2': r2})
        print(f"   Fold {fold+1:2d}: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

    # --- Overall metrics ---
    overall_mae = mean_absolute_error(y, y_pred_all)
    overall_rmse = np.sqrt(mean_squared_error(y, y_pred_all))
    overall_r2 = r2_score(y, y_pred_all)

    print(f"\n{'='*60}")
    print(f"OVERALL:  MAE={overall_mae:.2f}  RMSE={overall_rmse:.2f}  R²={overall_r2:.3f}")
    print(f"{'='*60}")

    # --- Save results ---
    results_df = pd.DataFrame({
        'subject': sub_ids,
        'true_age': y,
        'predicted_age': y_pred_all,
        'error': y_pred_all - y,
    })
    results_path = os.path.join(args.out_dir, 'xgb_age_prediction_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n[*] Saved {results_path}")

    # --- Scatter plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(y, y_pred_all, s=8, alpha=0.4)
    lims = [y.min() - 2, y.max() + 2]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted Age')
    ax.set_title(f'XGBoost Brain Age Prediction (10-fold CV)\n'
                 f'MAE={overall_mae:.2f}  RMSE={overall_rmse:.2f}  R²={overall_r2:.3f}')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    ax = axes[1]
    errors = y_pred_all - y
    ax.scatter(y, errors, s=8, alpha=0.4)
    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted − True Age')
    ax.set_title(f'Prediction Error vs Age\n'
                 f'mean error={errors.mean():.2f}  std={errors.std():.2f}')

    plt.tight_layout()
    scatter_path = os.path.join(args.out_dir, 'xgb_age_prediction_scatter.png')
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"[*] Saved {scatter_path}")

    # --- Feature importance plot ---
    avg_importance = feature_importance_acc / args.k_folds

    # Label features by type
    feature_types = []
    for k in range(500):
        feature_types.append(f'w_{k}')      # pi*T
        for cov_label in ['cxx', 'cxy', 'cxz', 'cyy', 'cyz', 'czz']:
            feature_types.append(f'{cov_label}_{k}')

    # Top 100 features
    top_idx = np.argsort(avg_importance)[::-1][:100]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(100), avg_importance[top_idx][::-1])
    ax.set_yticks(range(100))
    ax.set_yticklabels([feature_types[i] for i in top_idx][::-1], fontsize=9)
    ax.set_xlabel('Mean feature importance (gain)')
    ax.set_title('Top 100 features by XGBoost importance')
    plt.tight_layout()
    importance_path = os.path.join(args.out_dir, 'xgb_feature_importance.png')
    plt.savefig(importance_path, dpi=200)
    plt.close()
    print(f"[*] Saved {importance_path}")

    # Summary: weight vs covariance importance
    weight_idx = np.arange(0, 3500, 7)
    cov_idx = np.delete(np.arange(3500), weight_idx)
    print(f"\n    Total importance from weights (π×T):  {avg_importance[weight_idx].sum():.3f}")
    print(f"    Total importance from covariances:     {avg_importance[cov_idx].sum():.3f}")


if __name__ == "__main__":
    main()
