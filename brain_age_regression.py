import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

GMM_DIR = 'gmm_1d_outputs'
LABELS_FILE = 'participants.tsv'

# Load labels
df = pd.read_csv(LABELS_FILE, sep='\t')
df['participant_id'] = df['participant_id'].astype(str)

# Build feature matrix and age vector
records = []
for f in sorted(glob.glob(os.path.join(GMM_DIR, 'sub-*_preproc-cat12vbm_desc-gm_T1w.npy'))):
    subject_id = os.path.basename(f).split('_')[0].replace('sub-', '')
    row = df[df['participant_id'] == subject_id]
    if row.empty:
        continue
    records.append({'subject_id': subject_id, 'age': row['age'].values[0], 'features': np.load(f)})

X = np.array([r['features'] for r in records])
y = np.array([r['age'] for r in records])
subject_ids = np.array([r['subject_id'] for r in records])

print(f"Loaded {len(X)} subjects | Feature shape: {X.shape} | Age range: {y.min():.1f}-{y.max():.1f} yrs")

# Train/val split
X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
    X, y, subject_ids, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train
model = Lasso(alpha=0.1, max_iter=5000)
model.fit(X_train, y_train)

# Evaluate
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

print(f"\nTrain  ->  R²: {r2_score(y_train, y_train_pred):.4f}  |  MAE: {mean_absolute_error(y_train, y_train_pred):.2f} yrs")
print(f"Val    ->  R²: {r2_score(y_val, y_val_pred):.4f}  |  MAE: {mean_absolute_error(y_val, y_val_pred):.2f} yrs")

# Plot predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_true, y_pred, title in zip(
    axes,
    [y_train, y_val],
    [y_train_pred, y_val_pred],
    ['Train', 'Validation']
):
    lims = [y_true.min() - 2, y_true.max() + 2]
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color='steelblue')
    ax.plot(lims, lims, 'r--', linewidth=1.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual Age (yrs)'); ax.set_ylabel('Predicted Age (yrs)')
    ax.set_title(f'{title}  |  R²={r2_score(y_true, y_pred):.4f}, MAE={mean_absolute_error(y_true, y_pred):.2f} yrs')
    ax.set_aspect('equal')

plt.suptitle('Brain Age Prediction — Lasso Regression on GMM Features')
plt.tight_layout()
plt.savefig('brain_age_prediction.png', dpi=150, bbox_inches='tight')
print("\nSaved brain_age_prediction.png")

# Save validation predictions
pd.DataFrame({
    'participant_id': ids_val,
    'actual_age': y_val,
    'predicted_age': y_val_pred,
    'error': y_val_pred - y_val
}).to_csv('val_predictions.csv', index=False)
print("Saved val_predictions.csv")