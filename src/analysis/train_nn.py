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
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# --- Model: 2 hidden layers (sweet spot) ---
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
        val_loss = loss_fn(model(X_val_t), y_val_t).item()
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d} | Train MAE: {train_losses[-1]:.4f} | Val MAE: {val_loss:.4f}")

# --- Plot ---
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title("Training curves")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/training_loss.png", dpi=150)
plt.show()