"""
Train LassoCV on all training data, evaluate on external validation set.

Outputs (in --out-dir):
  - lasso_model.pkl (model + scaler)
  - lasso_val_predictions.csv
  - lasso_coefficients.csv
  - lasso_top50_coefficients.png
  - lasso_scatter.png
  - lasso_metrics.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
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
    parser.add_argument('--out-dir', default='lasso_final')
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

    # --- Standardize ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # --- Train ---
    print("\n[*] Training LassoCV (this may take a while)...")
    model = LassoCV(
        n_alphas=100,
        cv=5,
        random_state=args.seed,
        max_iter=10000,
    )
    model.fit(X_train_s, y_train)
    print(f"    alpha={model.alpha_:.4f}, "
          f"non-zero features={np.sum(model.coef_ != 0)}/{len(model.coef_)}")

    # --- Save model + scaler ---
    model_path = os.path.join(args.out_dir, 'lasso_model.pkl')
    joblib.dump({'model': model, 'scaler': scaler}, model_path)
    print(f"[*] Saved {model_path}")

    # --- Predict ---
    y_train_pred = model.predict(X_train_s)
    y_val_pred = model.predict(X_val_s)

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

    metrics_path = os.path.join(args.out_dir, 'lasso_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"model: LassoCV\n")
        f.write(f"alpha: {model.alpha_:.6f}\n")
        f.write(f"train_n: {len(y_train)}\n")
        f.write(f"val_n: {len(y_val)}\n")
        f.write(f"non_zero_features: {np.sum(model.coef_ != 0)}\n")
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
    pred_path = os.path.join(args.out_dir, 'lasso_val_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"[*] Saved {pred_path}")

    # --- Coefficients CSV ---
    K = 500
    coefs = model.coef_
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
                'coefficient': coefs[idx],
                'abs_coefficient': abs(coefs[idx]),
                'nonzero': coefs[idx] != 0,
            })

    coef_df = pd.DataFrame(rows).sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    coef_path = os.path.join(args.out_dir, 'lasso_coefficients.csv')
    coef_df.to_csv(coef_path, index=False)
    print(f"[*] Saved {coef_path}")

    n_w = coef_df[(coef_df['feature_type'] == 'weight') & coef_df['nonzero']].shape[0]
    n_d = coef_df[(coef_df['feature_type'] == 'diagonal') & coef_df['nonzero']].shape[0]
    n_o = coef_df[(coef_df['feature_type'] == 'off_diagonal') & coef_df['nonzero']].shape[0]
    print(f"\n    Non-zero: weights={n_w}/500  diagonal={n_d}/1500  off-diagonal={n_o}/1500")

    # --- Top 50 coefficients plot ---
    top50 = coef_df[coef_df['nonzero']].head(50)
    color_map = {'weight': 'steelblue', 'diagonal': 'coral', 'off_diagonal': 'mediumseagreen'}
    colors = [color_map[t] for t in top50['feature_type']]
    signs = ['pos' if c > 0 else 'neg' for c in top50['coefficient']]

    fig, ax = plt.subplots(figsize=(10, 12))
    vals = top50['coefficient'].values[::-1]
    c = colors[::-1]
    ax.barh(range(50), vals, color=c)
    ax.set_yticks(range(50))
    ax.set_yticklabels(top50['feature_name'].values[::-1], fontsize=8)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Standardized coefficient')
    ax.set_title(f'Lasso Top 50 Coefficients (by magnitude)\n'
                 f'Val MAE={val_mae:.2f}  R²={val_r2:.3f}  alpha={model.alpha_:.4f}')

    from matplotlib.patches import Patch
    legend = [Patch(facecolor='steelblue', label='Weight (π×T)'),
              Patch(facecolor='coral', label='Diagonal covariance'),
              Patch(facecolor='mediumseagreen', label='Off-diagonal covariance')]
    ax.legend(handles=legend, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'lasso_top50_coefficients.png'), dpi=200)
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
    plt.savefig(os.path.join(args.out_dir, 'lasso_scatter.png'), dpi=200)
    plt.close()
    print(f"[*] Saved plots to {args.out_dir}/")

    # --- Usage instructions ---
    print(f"\n[*] To load and predict on new data:")
    print(f"    bundle = joblib.load('{model_path}')")
    print(f"    X_new_s = bundle['scaler'].transform(X_new)")
    print(f"    y_pred = bundle['model'].predict(X_new_s)")


if __name__ == "__main__":
    main()
