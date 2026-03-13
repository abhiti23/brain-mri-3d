# this file performs Lasso regression based on the extracted FD coefficients
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Load data ---
coef_final_path = "artifacts/coefficients_final.npz"
if os.path.exists(coef_final_path):
    X = np.load("artifacts/coefficients_final.npz")["coefficients"]  # (3227,1,512)
    X = X.squeeze(1) # (3227, 512)
    y = np.load("data/raw/public_data_challenge/VBM_extracted/train_y.npy")  # (3227,)
else:
    X = np.load("artifacts/checkpoint.npz")["coefficients"]  # (a,1,512)
    X = X.squeeze(1)  # (a, 512)
    y = np.load(
        "data/raw/public_data_challenge/VBM_extracted/train_y.npy")  # (a,)
    y = y[:X.shape[0]]

'''
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()'''

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# scale the features
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

plt.suptitle('Brain Age Prediction — Lasso Regression on FDA Features')
plt.tight_layout()
plt.savefig('results/figures/brain_age_prediction_lasso.png', dpi=150,
            bbox_inches='tight')
print("\nSaved brain_age_prediction.png")

