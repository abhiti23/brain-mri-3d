"""
gif_age_covariances.py — Generate a GIF scrubbing through ages (covariances)

Fixed at 10 Gaussians. Each frame = one subject sorted youngest to oldest.
Size = model importance (fixed). Color = per-feature normalized trace.
Shape = actual per-subject covariance ellipsoid.

Output: age_covariances.gif

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
from viz_utils     import load_brain_outline, draw_brain_outline, make_ellipsoid
from viz_reg_utils import load_component_scores

GMM_DIR        = "training/train_gmm_weighted_outputs"
LABELS         = "training/train_labels/participants.tsv"
CENTERS        = "centers.npy"
AVG_BRAIN      = "group_average_gm_T1w.npy"
IMP_CSV        = "feature_importance.csv"
OUTPUT         = "age_covariances.gif"
ELLIPSOID_SCALE = 4.0

K              = 500
FEATURES_PER_K = 7
TOP_N          = 10
FRAME_INTERVAL = 100   # ms between frames
ELEV           = 25
AZIM           = -60

centers = np.load(CENTERS)
cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]

weight_imp, cov_imp, total_imp = load_component_scores(IMP_CSV)
imp_norm      = cov_imp / (cov_imp.max() + 1e-12)
ranked_by_cov = np.argsort(cov_imp)[::-1]
top_k         = ranked_by_cov[:TOP_N]
rest_k        = ranked_by_cov[TOP_N:]

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

print("Preloading traces for color normalization...")
raw_trace = np.zeros((n_subjects, K))
for i, r in enumerate(records):
    p = np.load(r["path"])
    raw_trace[i] = p[1::FEATURES_PER_K] + p[4::FEATURES_PER_K] + p[6::FEATURES_PER_K]

col_min   = raw_trace.min(axis=0)
col_max   = raw_trace.max(axis=0)
col_range = col_max - col_min
col_range[col_range == 0] = 1
norm_trace = (raw_trace - col_min) / col_range
del raw_trace
print("Done.")

print("Loading brain outline...")
bx, by, bz = load_brain_outline(AVG_BRAIN)
print("Ready. Generating frames...")

cmap = plt.cm.get_cmap("YlOrRd")

fig = plt.figure(figsize=(8, 7))
fig.patch.set_facecolor("white")
ax = fig.add_subplot(111, projection="3d")

XLIM = (cx.min() - 5, cx.max() + 5)
YLIM = (cy.min() - 5, cy.max() + 5)
ZLIM = (cz.min() - 5, cz.max() + 5)

sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04,
             label="Per-feature normalized covariance trace (0=min, 1=max)")
plt.suptitle(f"Covariance trace by age  |  Top-{TOP_N} components by importance",
             fontsize=10, fontweight="bold", color="black", y=0.97)


def load_covs(idx):
    p   = np.load(records[idx]["path"])
    cov = np.zeros((K, 3, 3))
    cov[:, 0, 0]              = p[1::FEATURES_PER_K]
    cov[:, 0, 1] = cov[:, 1, 0] = p[2::FEATURES_PER_K]
    cov[:, 0, 2] = cov[:, 2, 0] = p[3::FEATURES_PER_K]
    cov[:, 1, 1]              = p[4::FEATURES_PER_K]
    cov[:, 1, 2] = cov[:, 2, 1] = p[5::FEATURES_PER_K]
    cov[:, 2, 2]              = p[6::FEATURES_PER_K]
    return cov


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

    cov_mats = load_covs(i)
    colors   = norm_trace[i, top_k]
    for rank, k in enumerate(top_k):
        color = cmap(colors[rank])
        scale = ELLIPSOID_SCALE * (0.5 + imp_norm[k] * 0.5)
        X, Y, Z = make_ellipsoid(centers[k], cov_mats[k], scale=scale)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.75,
                        linewidth=0, antialiased=False)

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
