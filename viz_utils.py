"""
Shared utilities for viz_slider1 and viz_slider2.
  - load_brain_outline : extracts a sparse wireframe from the average brain mask
  - draw_brain_outline : plots it on a 3D axis
  - make_ellipsoid     : generates ellipsoid surface points from a 3x3 covariance
"""

import numpy as np


# ── brain outline ─────────────────────────────────────────────────────────────
def load_brain_outline(avg_brain_path, threshold=0.1, step=4):
    """
    Load the average brain, threshold it to get a mask, then sample
    surface voxels sparsely for a lightweight wireframe.
    Returns (xs, ys, zs) arrays of surface point coordinates.
    """
    brain = np.squeeze(np.load(avg_brain_path))
    mask  = brain > threshold

    # surface voxels: inside mask but with at least one neighbour outside
    from scipy.ndimage import binary_erosion
    eroded  = binary_erosion(mask)
    surface = mask & ~eroded

    coords = np.argwhere(surface)
    # subsample for speed
    coords = coords[::step]
    return coords[:, 0], coords[:, 1], coords[:, 2]


def draw_brain_outline(ax, xs, ys, zs, alpha=0.08, color="#888888", size=0.3):
    ax.scatter(xs, ys, zs, c=color, s=size, alpha=alpha,
               edgecolors="none", depthshade=False)


# ── ellipsoid ─────────────────────────────────────────────────────────────────
def make_ellipsoid(center, cov, scale=1.0, n=12):
    """
    Return (X, Y, Z) surface arrays for a 3D ellipsoid defined by
    the covariance matrix `cov` centred at `center`.
    `scale` multiplies the radii so they're visible at brain voxel scale.
    """
    # eigendecomposition: axes and lengths
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)   # clamp negatives from numerical noise
    radii = np.sqrt(vals) * scale

    # unit sphere
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi,     n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # stretch by radii, rotate by eigenvectors
    sphere = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)  # (3, N)
    ellipsoid = vecs @ (np.diag(radii) @ sphere)                   # (3, N)

    X = (ellipsoid[0] + center[0]).reshape(n, n)
    Y = (ellipsoid[1] + center[1]).reshape(n, n)
    Z = (ellipsoid[2] + center[2]).reshape(n, n)
    return X, Y, Z