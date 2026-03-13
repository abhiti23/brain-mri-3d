import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETTINGS & CONFIGURATION
# ==========================================
# Original brain file (used ONLY to get the correct bounding box dimensions)
brain_file = 'group_average_gm_T1w.npy' 
# The new centers file from your collaborator
centers_file = 'group_average_centers.npy'

n_components = 500

# ==========================================
# 2. LOAD DATA
# ==========================================
print("1. Loading brain data dimensions and collaborator's centers...")

# We only load this to dynamically get X_MAX, Y_MAX, Z_MAX for the plot axes
brain_data = np.squeeze(np.load(brain_file))
X_MAX, Y_MAX, Z_MAX = brain_data.shape
print(f"   -> Brain dimensions: X={X_MAX}, Y={Y_MAX}, Z={Z_MAX}")

# Load the 500 centers provided by your collaborator
coords = np.load(centers_file)
print(f"   -> Loaded {coords.shape[0]} centers from '{centers_file}'.")

# ==========================================
# 3. SAVE 3D SCATTER PLOTS TO PNG (3 VIEWS)
# ==========================================
print("\n2. Generating professional 3D plots for the report...")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Styling for a professional report (white background)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Extract X, Y, Z from the collaborator's coordinates
x = coords[:, 0]
y = coords[:, 1]
z = coords[:, 2]

# Scatter plot the centers
scatter = ax.scatter(x, y, z, c=z, cmap='plasma', s=30, alpha=0.9, edgecolors='black', linewidth=0.3)

# Format the plot to match the exact brain bounding box dynamically
ax.set_xlim(0, X_MAX)
ax.set_ylim(0, Y_MAX)
ax.set_zlim(0, Z_MAX)

# Clean report styling
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlabel('X (Left-Right)', color='black', fontsize=12)
ax.set_ylabel('Y (Posterior-Anterior)', color='black', fontsize=12)
ax.set_zlabel('Z (Inferior-Superior)', color='black', fontsize=12)

# Subdue the grid lines
ax.xaxis.pane.set_edgecolor('grey')
ax.yaxis.pane.set_edgecolor('grey')
ax.zaxis.pane.set_edgecolor('grey')
ax.tick_params(colors='black')

# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('Z-axis Depth (Superior to Inferior)', color='black', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='black')

# --- DEFINING THE THREE CAMERA ANGLES ---
# Format: (View Name, Elevation, Azimuth)
views = [
    ("Sagittal", 0, 180),  # X-axis points directly into the page (Side view)
    ("Coronal", 0, -90),   # Y-axis points directly into the page (Front view)
    ("Axial", 90, -90)     # Z-axis points directly into the page (Top-down view)
]

for view_name, elev, azim in views:
    # Move the camera
    ax.view_init(elev=elev, azim=azim)
    
    # Update the title for this specific view
    ax.set_title(f'Collaborator GMM Centers ({view_name} View)', color='black', fontsize=16)
    
    # Save the file
    filename = f"gmm_centers_report_{view_name.lower()}.png"
    print(f"   -> Saving {view_name} view to '{filename}'...")
    
    # bbox_inches='tight' trims the excess white margins around the saved image
    plt.savefig(filename, dpi=300, bbox_inches='tight')

# Close the figure to free up memory
plt.close()

print("\nAll tasks completed successfully!")