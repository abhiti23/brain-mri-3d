"""
Train XGBoost on all training data, evaluate on external validation set.

Outputs (in --out-dir):
  - xgb_model.json
  - xgb_val_predictions.csv
  - xgb_feature_importance.csv
  - xgb_feature_importance_top50.png
  - xgb_scatter.png
  - xgb_metrics.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_dataset(features_dir, participants_tsv, age_col, id_col):
    df = pd.read_csv(participants_tsv, sep='\t')
    df = df.dropna(subset=[age_col])
    df[age_col] = df[age_col].astype(float)
    df[id_col] = df[id_col].astype(str)

    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])
    file_lookup = {f.split('_')[0]: f for f in feature_files}

    matched = []
    for _, row in df.iterrows():
        sub_id = str(row[id_col])
        for c in [sub_id, f"sub-{sub_id}"]:
            if c in file_lookup:
                matched.append({
                    'sub_id': c,
                    'age': row[age_col],
                    'file': file_lookup[c],
                })
                break

    X = np.array([np.load(os.path.join(features_dir, m['file'])) for m in matched])
    y = np.array([m['age'] for m in matched])
    ids = [m['sub_id'] for m in matched]
    return X, y, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-features', default='gmm_features_weighted')
    parser.add_argument('--train-participants', default='./train_labels/participants.tsv')
    parser.add_argument('--val-features', default='val_gmm_weighted_outputs')
    parser.add_argument('--val-participants', default='val_labels/participants.tsv')
    parser.add_argument('--age-col', default='age')
    parser.add_argument('--id-col', default='participant_id')
    parser.add_argument('--out-dir', default='xgb_final')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load data ---
    print("[*] Loading training data...")
    X_train, y_train, ids_train = load_dataset(
        args.train_features, args.train_participants, args.age_col, args.id_col)
    print(f"    Train: {X_train.shape[0]} subjects, age {y_train.min():.1f}–{y_train.max():.1f}")

    print("[*] Loading validation data...")
    X_val, y_val, ids_val = load_dataset(
        args.val_features, args.val_participants, args.age_col, args.id_col)
    print(f"    Val:   {X_val.shape[0]} subjects, age {y_val.min():.1f}–{y_val.max():.1f}")

    # --- Train ---
    xgb_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': args.seed,
        'n_jobs': -1,
    }

    print("\n[*] Training XGBoost...")
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    # --- Save model ---
    model_path = os.path.join(args.out_dir, 'xgb_model.json')
    model.save_model(model_path)
    print(f"[*] Saved {model_path}")

    # --- Predict ---
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # --- Metrics ---
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n    {'':>8} {'MAE':>6} {'RMSE':>6} {'R²':>6}")
    print(f"    {'Train':>8} {train_mae:>6.2f} {train_rmse:>6.2f} {train_r2:>6.3f}")
    print(f"    {'Val':>8} {val_mae:>6.2f} {val_rmse:>6.2f} {val_r2:>6.3f}")

    metrics_path = os.path.join(args.out_dir, 'xgb_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"model: XGBoost\n")
        f.write(f"train_n: {len(y_train)}\n")
        f.write(f"val_n: {len(y_val)}\n")
        f.write(f"train_mae: {train_mae:.4f}\n")
        f.write(f"train_rmse: {train_rmse:.4f}\n")
        f.write(f"train_r2: {train_r2:.4f}\n")
        f.write(f"val_mae: {val_mae:.4f}\n")
        f.write(f"val_rmse: {val_rmse:.4f}\n")
        f.write(f"val_r2: {val_r2:.4f}\n")
    print(f"[*] Saved {metrics_path}")

    # --- Validation predictions CSV ---
    pred_df = pd.DataFrame({
        'subject': ids_val,
        'true_age': y_val,
        'predicted_age': y_val_pred,
        'error': y_val_pred - y_val,
    })
    pred_path = os.path.join(args.out_dir, 'xgb_val_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"[*] Saved {pred_path}")

    # --- Feature importance CSV ---
    K = 500
    importances = model.feature_importances_
    rows = []
    for k in range(K):
        base = 7 * k
        labels = [
            (f'w_{k}', 'weight', base),
            (f'cxx_{k}', 'diagonal', base + 1),
            (f'cxy_{k}', 'off_diagonal', base + 2),
            (f'cxz_{k}', 'off_diagonal', base + 3),
            (f'cyy_{k}', 'diagonal', base + 4),
            (f'cyz_{k}', 'off_diagonal', base + 5),
            (f'czz_{k}', 'diagonal', base + 6),
        ]
        for name, feat_type, idx in labels:
            rows.append({
                'feature_index': idx,
                'feature_name': name,
                'component': k,
                'feature_type': feat_type,
                'importance': importances[idx],
            })

    imp_df = pd.DataFrame(rows).sort_values('importance', ascending=False).reset_index(drop=True)
    imp_path = os.path.join(args.out_dir, 'xgb_feature_importance.csv')
    imp_df.to_csv(imp_path, index=False)
    print(f"[*] Saved {imp_path}")

    weight_imp = imp_df[imp_df['feature_type'] == 'weight']['importance'].sum()
    diag_imp = imp_df[imp_df['feature_type'] == 'diagonal']['importance'].sum()
    offdiag_imp = imp_df[imp_df['feature_type'] == 'off_diagonal']['importance'].sum()
    print(f"\n    Importance: weights={weight_imp:.3f} diag={diag_imp:.3f} offdiag={offdiag_imp:.3f}")

    # --- Feature importance plot ---
    top50 = imp_df.head(50)
    color_map = {'weight': 'steelblue', 'diagonal': 'coral', 'off_diagonal': 'mediumseagreen'}
    colors = [color_map[t] for t in top50['feature_type']]

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(range(50), top50['importance'].values[::-1], color=colors[::-1])
    ax.set_yticks(range(50))
    ax.set_yticklabels(top50['feature_name'].values[::-1], fontsize=8)
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title(f'XGBoost Top 50 Features\nVal MAE={val_mae:.2f}  R²={val_r2:.3f}')

    from matplotlib.patches import Patch
    legend = [Patch(facecolor='steelblue', label='Weight (π×T)'),
              Patch(facecolor='coral', label='Diagonal covariance'),
              Patch(facecolor='mediumseagreen', label='Off-diagonal covariance')]
    ax.legend(handles=legend, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'xgb_feature_importance_top50.png'), dpi=200)
    plt.close()

    # --- Scatter plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    age_lims = [min(y_train.min(), y_val.min()) - 3, max(y_train.max(), y_val.max()) + 3]

    ax = axes[0]
    ax.scatter(y_val, y_val_pred, s=8, alpha=0.4, c='steelblue')
    ax.plot(age_lims, age_lims, 'r--', lw=1)
    ax.set_xlim(age_lims); ax.set_ylim(age_lims)
    ax.set_xlabel('True Age'); ax.set_ylabel('Predicted Age')
    ax.set_title(f'Validation\nMAE={val_mae:.2f}  RMSE={val_rmse:.2f}  R²={val_r2:.3f}')
    ax.set_aspect('equal')

    ax = axes[1]
    errors = y_val_pred - y_val
    ax.scatter(y_val, errors, s=8, alpha=0.4, c='steelblue')
    ax.axhline(0, color='r', ls='--', lw=1)
    z = np.polyfit(y_val, errors, 1)
    x_line = np.linspace(age_lims[0], age_lims[1], 100)
    ax.plot(x_line, np.polyval(z, x_line), 'orange', lw=1.5, label=f'slope={z[0]:.3f}')
    ax.set_xlim(age_lims)
    ax.set_xlabel('True Age'); ax.set_ylabel('Predicted − True Age')
    ax.set_title(f'Error vs Age\nmean={errors.mean():.2f}  std={errors.std():.2f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'xgb_scatter.png'), dpi=200)
    plt.close()
    print(f"[*] Saved plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
