# this file trains a shallow neural network for regressing age on fda_coefs
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Load data ---
X = np.load("artifacts/coefficients_final.npz")["coefficients"]  # (3227,1,512)
X = X.squeeze(1) # (3227, 512)
y = np.load("data/raw/public_data_challenge/VBM_extracted/train_y.npy")  # (3227,)

# --- Preprocess ---
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
# We keep the original 'y' values in a separate variable for final error calculation
# --- CHANGE: Created y_scaled to keep the original y available ---
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_val, y_train, y_val = train_test_split(X, y_scaled, test_size=0.15, random_state=42)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# --- Model: 2 hidden layers ---
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(256, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
loss_fn = nn.L1Loss()

train_losses = []
val_losses = []
# --- ADDITION: Track MAE in actual years ---
val_mae_years_list = []

for epoch in range(300):
    # --- Train ---
    model.train()
    epoch_train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * len(xb)
    train_losses.append(epoch_train_loss / len(X_train_t))

    # --- Validate ---
    model.eval()
    with torch.no_grad():
        val_pred_scaled = model(X_val_t)
        val_loss = loss_fn(val_pred_scaled, y_val_t).item()
        
        # --- CHANGE: Convert predictions back to original scale (Years) ---
        # We use .cpu().numpy() because the scaler expects numpy arrays
        val_pred_years = scaler_y.inverse_transform(val_pred_scaled.cpu().numpy())
        y_val_years = scaler_y.inverse_transform(y_val_t.cpu().numpy())
        
        # Calculate MAE in years
        mae_years = np.mean(np.abs(val_pred_years - y_val_years))
        val_mae_years_list.append(mae_years)

    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if (epoch + 1) % 50 == 0:
        # --- CHANGE: Added "Val MAE (Years)" to the print statement ---
        print(f"Epoch {epoch+1:3d} | Train MAE: {train_losses[-1]:.4f} | Val MAE (Scaled): {val_loss:.4f} | Val MAE (Years): {mae_years:.2f}")

# --- Plot ---
plt.figure(figsize=(10, 4))
# --- CHANGE: Plotting actual years error instead of scaled error for the validation line ---
plt.plot(train_losses, label="Train loss (Scaled)")
plt.plot(val_losses, label="Val loss (Scaled)")
plt.plot(val_mae_years_list, label="Val MAE (Years)", linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Training curves (Scaled vs Years)")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/training_loss.png", dpi=150)
plt.show()
