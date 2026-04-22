"""
gif_age_weights.py — Generate a GIF scrubbing through ages (weights)

Fixed at 250 Gaussians. Each frame = one subject sorted youngest to oldest.
Size = model importance (fixed). Color = per-feature normalized intensity.

Output: age_weights.gif

Requires
--------
  feature_importance.csv
  centers.npy
  group_average_gm_T1w.npy
  training/train_gmm_weighted_outputs/
  training/train_labels/participants.tsv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from viz_utils     import load_brain_outline, draw_brain_outline
from viz_reg_utils import load_component_scores

GMM_DIR   = "training/train_gmm_weighted_outputs"
LABELS    = "training/train_labels/participants.tsv"
CENTERS   = "centers.npy"
AVG_BRAIN = "group_average_gm_T1w.npy"
IMP_CSV   = "feature_importance.csv"
OUTPUT    = "age_weights.gif"

K              = 500
FEATURES_PER_K = 7
TOP_N          = 250
FRAME_INTERVAL = 100   # ms between frames
ELEV           = 25
AZIM           = -60

centers = np.load(CENTERS)
cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]

weight_imp, cov_imp, total_imp = load_component_scores(IMP_CSV)
imp_norm         = weight_imp / (weight_imp.max() + 1e-12)
sizes_all        = 8 + imp_norm * 150
ranked_by_weight = np.argsort(weight_imp)[::-1]
top_k            = ranked_by_weight[:TOP_N]
rest_k           = ranked_by_weight[TOP_N:]

df = pd.read_csv(LABELS, sep="\t")
df["participant_id"] = df["participant_id"].astype(str)

files = sorted(glob.glob(os.path.join(GMM_DIR, "sub-*_preproc-cat12vbm_desc-gm_T1w.npy")))
assert files, f"No files found in {GMM_DIR}"

records = []
for f in files:
    pid = os.path.basename(f).split("_")[0].replace("sub-", "")
    row = df[df["participant_id"] == pid]
    if row.empty:
        continue
    age = float(row["age"].values[0])
    records.append({"path": f, "pid": pid, "age": age})

records.sort(key=lambda r: r["age"])
ages = np.array([r["age"] for r in records])
n_subjects = len(records)
print(f"Loaded {n_subjects} subjects | Age range: {ages.min():.1f} – {ages.max():.1f} yrs")

print("Preloading weights for color normalization...")
all_weights = np.zeros((n_subjects, K))
for i, r in enumerate(records):
    p = np.load(r["path"])
    all_weights[i] = p[0::FEATURES_PER_K]

col_min   = all_weights.min(axis=0)
col_max   = all_weights.max(axis=0)
col_range = col_max - col_min
col_range[col_range == 0] = 1
norm_table = (all_weights - col_min) / col_range
del all_weights
print("Done.")

print("Loading brain outline...")
bx, by, bz = load_brain_outline(AVG_BRAIN)
print("Ready. Generating frames...")

fig = plt.figure(figsize=(8, 7))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d")

XLIM = (cx.min() - 5, cx.max() + 5)
YLIM = (cy.min() - 5, cy.max() + 5)
ZLIM = (cz.min() - 5, cz.max() + 5)

sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04,
             label="Per-feature normalized GM intensity")
plt.suptitle(f"GM intensity by age  |  Top-{TOP_N} components by importance",
             fontsize=10, fontweight="bold", color="black", y=0.97)


def render_frame(i):
    ax.cla()
    ax.set_facecolor("white")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#cccccc")
    ax.grid(False)
    ax.tick_params(colors="black", labelsize=7)
    ax.set_xlabel("x", color="black", fontsize=8)
    ax.set_ylabel("y", color="black", fontsize=8)
    ax.set_zlabel("z", color="black", fontsize=8)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_zlim(ZLIM)

    draw_brain_outline(ax, bx, by, bz)

    if len(rest_k) > 0:
        ax.scatter(cx[rest_k], cy[rest_k], cz[rest_k],
                   c="#888888", s=4, alpha=0.25,
                   edgecolors="none", depthshade=False)

    colors = norm_table[i, top_k]
    ax.scatter(cx[top_k], cy[top_k], cz[top_k],
               c=colors, s=sizes_all[top_k], cmap="YlOrRd",
               alpha=0.85, edgecolors="none", depthshade=True,
               vmin=0, vmax=1)

    age = records[i]["age"]
    ax.set_title(
        f"Subject {i + 1} / {n_subjects}  |  Age: {age:.1f} yrs",
        fontsize=9, color="black", pad=6
    )

    if i % 100 == 0:
        print(f"  Frame {i + 1} / {n_subjects}")


ani = animation.FuncAnimation(fig, render_frame, frames=n_subjects,
                               interval=FRAME_INTERVAL, repeat=True)

print(f"Saving {OUTPUT} ...")
ani.save(OUTPUT, writer="pillow", fps=10, dpi=80)
print(f"Done! Saved to {OUTPUT}")
