"""
Plot the 500 weighted GMM centers in 3D with anatomical context.

Inputs:
  - centers.npy        (500 x 3)
  - group_average brain (optional, for anatomical backdrop)

Outputs:
  - centers_3d.png       — 3D scatter from three viewing angles
  - centers_projections.png — 2D projections onto sagittal / coronal / axial planes
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


def plot_3d_scatter(centers, brain_shape, filename, brain=None):
    views = [
        ('Sagittal-ish',  0, 0),
        ('Axial-ish',    90, 270),
        ('Oblique',      30, 45),
    ]

    fig = plt.figure(figsize=(18, 6))
    for i, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        # Optional anatomical context
        if brain is not None:
            mask = brain > 0.1
            coords = np.argwhere(mask)
            rng = np.random.RandomState(0)
            idx = rng.choice(len(coords), size=min(3000, len(coords)), replace=False)
            bg = coords[idx]
            ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2],
                       c='lightgray', s=1, alpha=0.15, marker='.')

        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   c='crimson', s=25, alpha=0.75,
                   edgecolors='darkred', linewidths=0.3)

        ax.set_xlim(0, brain_shape[0])
        ax.set_ylim(0, brain_shape[1])
        ax.set_zlim(0, brain_shape[2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"[*] Saved {filename}")


def plot_2d_projections(centers, brain_shape, filename, brain=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    plane_defs = [
        ('Sagittal (y-z)', 1, 2, 0),
        ('Coronal (x-z)',  0, 2, 1),
        ('Axial (x-y)',    0, 1, 2),
    ]

    for ax, (title, d1, d2, d_collapse) in zip(axes, plane_defs):
        if brain is not None:
            bg = brain.max(axis=d_collapse).T
            ax.imshow(bg, cmap='gray', origin='lower', alpha=0.5,
                      extent=[0, brain_shape[d1], 0, brain_shape[d2]])

        ax.scatter(centers[:, d1], centers[:, d2],
                   c='crimson', s=18, alpha=0.7,
                   edgecolors='darkred', linewidths=0.3)
        ax.set_xlim(0, brain_shape[d1])
        ax.set_ylim(0, brain_shape[d2])
        ax.set_aspect('equal')
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"[*] Saved {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('centers', help="Path to centers.npy (K x 3)")
    parser.add_argument('--brain', default=None,
                        help="Optional: group-average brain .npy for anatomical backdrop")
    parser.add_argument('--prefix', default='centers')
    args = parser.parse_args()

    centers = np.load(args.centers)
    print(f"[*] Loaded {centers.shape[0]} centers, shape {centers.shape}")

    brain = None
    brain_shape = (
        int(centers[:, 0].max()) + 10,
        int(centers[:, 1].max()) + 10,
        int(centers[:, 2].max()) + 10,
    )
    if args.brain:
        brain = np.squeeze(np.load(args.brain))
        brain_shape = brain.shape
        print(f"[*] Loaded brain for backdrop, shape {brain_shape}")

    plot_3d_scatter(centers, brain_shape, f'{args.prefix}_3d.png', brain=brain)
    plot_2d_projections(centers, brain_shape, f'{args.prefix}_projections.png', brain=brain)

    # Quick stats on the center cloud
    print(f"\n    x range: [{centers[:, 0].min():.1f}, {centers[:, 0].max():.1f}]")
    print(f"    y range: [{centers[:, 1].min():.1f}, {centers[:, 1].max():.1f}]")
    print(f"    z range: [{centers[:, 2].min():.1f}, {centers[:, 2].max():.1f}]")


if __name__ == "__main__":
    main()
