"""
Detailed analysis of Lasso feature survival patterns.

Questions answered:
  1. When a weight survives, do its covariances also survive (and vice versa)?
  2. Which covariance types survive: diagonal (cxx, cyy, czz) vs off-diagonal (cxy, cxz, cyz)?
  3. How many components have both weight AND at least one covariance surviving?
  4. Spatial distribution: where are the surviving components located?
"""

import numpy as np
import glob
import os
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def main():
    # --- Load and fit (same as the quick check) ---
    df = pd.read_csv('./train_labels/participants.tsv', sep='\t')
    df['participant_id'] = df['participant_id'].astype(str)

    files = sorted(glob.glob('gmm_features_weighted/*.npy'))
    X, y = [], []
    for f in files:
        sub_id = os.path.basename(f).split('_')[0].replace('sub-', '')
        row = df[df['participant_id'] == sub_id]
        if not row.empty:
            X.append(np.load(f))
            y.append(row['age'].values[0])

    X, y = np.array(X), np.array(y)
    X = StandardScaler().fit_transform(X)

    model = Lasso(alpha=0.085, max_iter=10000)
    model.fit(X, y)
    coefs = model.coef_

    K = 500
    cov_names = ['c_xx', 'c_xy', 'c_xz', 'c_yy', 'c_yz', 'c_zz']

    # --- Unpack per-component survival ---
    weight_alive = np.zeros(K, dtype=bool)
    cov_alive = np.zeros((K, 6), dtype=bool)
    weight_coefs = np.zeros(K)
    cov_coefs = np.zeros((K, 6))

    for k in range(K):
        base = 7 * k
        weight_coefs[k] = coefs[base]
        weight_alive[k] = coefs[base] != 0
        for j in range(6):
            cov_coefs[k, j] = coefs[base + 1 + j]
            cov_alive[k, j] = coefs[base + 1 + j] != 0

    any_cov_alive = cov_alive.any(axis=1)
    n_cov_per_component = cov_alive.sum(axis=1)

    # ==========================================
    # 1. CO-OCCURRENCE ANALYSIS
    # ==========================================
    both_alive = weight_alive & any_cov_alive
    weight_only = weight_alive & ~any_cov_alive
    cov_only = ~weight_alive & any_cov_alive
    neither = ~weight_alive & ~any_cov_alive

    print("=" * 60)
    print("1. CO-OCCURRENCE: weight vs covariance survival")
    print("=" * 60)
    print(f"   Both weight AND covariance(s) survive: {both_alive.sum():>4} / {K}")
    print(f"   Weight only (no covariances):          {weight_only.sum():>4} / {K}")
    print(f"   Covariance(s) only (no weight):        {cov_only.sum():>4} / {K}")
    print(f"   Neither survives:                      {neither.sum():>4} / {K}")
    print(f"\n   Components with at least something:    {(weight_alive | any_cov_alive).sum():>4} / {K}")

    # ==========================================
    # 2. COVARIANCE TYPE BREAKDOWN
    # ==========================================
    diag_idx = [0, 3, 5]       # c_xx, c_yy, c_zz
    offdiag_idx = [1, 2, 4]    # c_xy, c_xz, c_yz

    print(f"\n{'=' * 60}")
    print("2. COVARIANCE TYPE BREAKDOWN")
    print("=" * 60)
    print(f"\n   Per covariance entry survival:")
    for j, name in enumerate(cov_names):
        count = cov_alive[:, j].sum()
        kind = "diagonal" if j in diag_idx else "off-diagonal"
        print(f"     {name:>5} ({kind:>12}): {count:>4} / {K}")

    diag_total = cov_alive[:, diag_idx].sum()
    offdiag_total = cov_alive[:, offdiag_idx].sum()
    print(f"\n   Total diagonal entries surviving:     {diag_total:>4} / {K * 3}")
    print(f"   Total off-diagonal entries surviving: {offdiag_total:>4} / {K * 3}")
    print(f"   Ratio (off-diag / diagonal):         {offdiag_total / max(diag_total, 1):.2f}")

    # ==========================================
    # 3. PER-COMPONENT RICHNESS
    # ==========================================
    print(f"\n{'=' * 60}")
    print("3. HOW MANY FEATURES SURVIVE PER COMPONENT?")
    print("=" * 60)

    total_per_component = weight_alive.astype(int) + n_cov_per_component
    for n_feat in range(8):
        count = (total_per_component == n_feat).sum()
        if count > 0:
            print(f"   {n_feat} / 7 features survive: {count:>4} components")

    # ==========================================
    # 4. COEFFICIENT SIGN AND MAGNITUDE
    # ==========================================
    print(f"\n{'=' * 60}")
    print("4. WEIGHT COEFFICIENT SIGNS")
    print("=" * 60)
    alive_weight_coefs = weight_coefs[weight_alive]
    n_pos = (alive_weight_coefs > 0).sum()
    n_neg = (alive_weight_coefs < 0).sum()
    print(f"   Positive (more GM mass → older): {n_pos}")
    print(f"   Negative (more GM mass → younger): {n_neg}")
    print(f"   (Negative = regions that atrophy with age)")

    # ==========================================
    # 5. SPATIAL ANALYSIS (if centers.npy exists)
    # ==========================================
    if os.path.exists('centers.npy'):
        centers = np.load('centers.npy')

        print(f"\n{'=' * 60}")
        print("5. SPATIAL DISTRIBUTION OF SURVIVING COMPONENTS")
        print("=" * 60)

        alive_centers = centers[weight_alive | any_cov_alive]
        dead_centers = centers[neither]

        print(f"   Surviving components: {len(alive_centers)}")
        print(f"   Dead components:      {len(dead_centers)}")

        if os.path.exists('group_average_gm_T1w.npy'):
            brain = np.squeeze(np.load('group_average_gm_T1w.npy'))

            def gm_at(c):
                ci = np.clip(c[:, 0].astype(int), 0, brain.shape[0] - 1)
                cj = np.clip(c[:, 1].astype(int), 0, brain.shape[1] - 1)
                ck = np.clip(c[:, 2].astype(int), 0, brain.shape[2] - 1)
                return brain[ci, cj, ck]

            gm_alive = gm_at(alive_centers)
            gm_dead = gm_at(dead_centers)
            print(f"   Mean GM at surviving centers: {gm_alive.mean():.4f}")
            print(f"   Mean GM at dead centers:      {gm_dead.mean():.4f}")

            # Spatial plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 11))
            plane_defs = [
                ('Sagittal (y-z)', 1, 2, 0),
                ('Coronal (x-z)',  0, 2, 1),
                ('Axial (x-y)',    0, 1, 2),
            ]

            # Color by: weight sign for components where weight survived
            for row_idx, (label, marker_set, ms) in enumerate([
                ("Age-predictive components", centers[weight_alive | any_cov_alive], 18),
                ("Unused components", centers[neither], 12),
            ]):
                for col, (title, d1, d2, d_collapse) in enumerate(plane_defs):
                    ax = axes[row_idx, col]
                    bg = brain.max(axis=d_collapse).T
                    ax.imshow(bg, cmap='gray', origin='lower', alpha=0.5,
                              extent=[0, brain.shape[d1], 0, brain.shape[d2]])

                    if row_idx == 0:
                        # Color surviving components by their weight coefficient sign
                        mask_both = weight_alive | any_cov_alive
                        w_vals = weight_coefs[mask_both]
                        c_plot = centers[mask_both]
                        colors = np.where(w_vals < 0, 'blue',
                                          np.where(w_vals > 0, 'red', 'gray'))
                        ax.scatter(c_plot[:, d1], c_plot[:, d2],
                                   c=colors, s=ms, alpha=0.7, edgecolors='black',
                                   linewidths=0.3)
                    else:
                        ax.scatter(marker_set[:, d1], marker_set[:, d2],
                                   c='gray', s=ms, alpha=0.5, edgecolors='darkgray',
                                   linewidths=0.3)

                    ax.set_xlim(0, brain.shape[d1])
                    ax.set_ylim(0, brain.shape[d2])
                    ax.set_aspect('equal')
                    if row_idx == 0:
                        ax.set_title(title, fontsize=11)
                    if col == 0:
                        ax.set_ylabel(label, fontsize=11)

            # Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=8, label='Positive (↑GM → ↑age)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=8, label='Negative (↑GM → ↓age)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=8, label='Covariance only (no weight)'),
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                       fontsize=11, bbox_to_anchor=(0.5, 0.01))

            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.savefig('lasso_feature_anatomy.png', dpi=200)
            plt.close()
            print(f"\n   Saved lasso_feature_anatomy.png")

    # ==========================================
    # SUMMARY TABLE
    # ==========================================
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"   Total non-zero features: {(coefs != 0).sum()} / 3500")
    print(f"   Weights:    {weight_alive.sum()} / 500  ({weight_alive.sum()/5:.1f}%)")
    print(f"   Diagonal:   {diag_total} / 1500  ({diag_total/15:.1f}%)")
    print(f"   Off-diag:   {offdiag_total} / 1500  ({offdiag_total/15:.1f}%)")
    print(f"   Components fully dead: {neither.sum()} / 500")
    print(f"   Components fully alive (7/7): {(total_per_component == 7).sum()} / 500")


if __name__ == "__main__":
    main()
