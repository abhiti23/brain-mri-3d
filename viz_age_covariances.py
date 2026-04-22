"""
viz_age_covariances.py — Age slider, covariance viewer (ellipsoids)

Same table logic as viz_age_weights.py but for covariances.
Each component's covariance trace (cxx + cyy + czz) is normalized
per-feature across all subjects to [0, 1]:
  - 0 = this subject has the least spread at this location vs all others
  - 1 = this subject has the most spread at this location vs all others

  - Size   = importance from feature_importance.csv (fixed)
             Bigger ellipsoid = model relies on this region's shape more
  - Color  = per-feature normalized covariance trace
             Yellow = low spread relative to others, Red = high spread
  - Shape  = actual per-subject covariance matrix (loaded on demand)

Note: covariances are loaded per subject on demand (not preloaded)
to keep RAM usage low. Expect a brief pause when changing subjects.

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
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from viz_utils     import load_brain_outline, draw_brain_outline, make_ellipsoid
from viz_reg_utils import load_component_scores

GMM_DIR        = "training/train_gmm_weighted_outputs"
LABELS         = "training/train_labels/participants.tsv"
CENTERS        = "centers.npy"
AVG_BRAIN      = "group_average_gm_T1w.npy"
IMP_CSV        = "feature_importance.csv"
ELLIPSOID_SCALE = 4.0

K              = 500
FEATURES_PER_K = 7
DEFAULT_TOP_N  = 20

centers = np.load(CENTERS)
cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]

weight_imp, cov_imp, total_imp = load_component_scores(IMP_CSV)
imp_norm      = cov_imp / (cov_imp.max() + 1e-12)
ranked_by_cov = np.argsort(cov_imp)[::-1]

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

# --- preload only traces for color normalization (~13MB) ---
print("Preloading traces for color normalization...")
raw_trace = np.zeros((n_subjects, K))
for i, r in enumerate(records):
    p = np.load(r["path"])
    raw_trace[i] = p[1::FEATURES_PER_K] + p[4::FEATURES_PER_K] + p[6::FEATURES_PER_K]

col_min   = raw_trace.min(axis=0)
col_max   = raw_trace.max(axis=0)
col_range = col_max - col_min
col_range[col_range == 0] = 1
norm_trace = (raw_trace - col_min) / col_range   # (n_subjects, K), each col in [0,1]
del raw_trace   # free memory
print("Done.")

print("Loading brain outline...")
bx, by, bz = load_brain_outline(AVG_BRAIN)
print("Ready.")

cur_idx = 0
cur_n   = DEFAULT_TOP_N
cmap    = plt.cm.get_cmap("YlOrRd")


def load_covs(idx):
    """Load covariance matrices for one subject on demand."""
    p   = np.load(records[idx]["path"])
    cov = np.zeros((K, 3, 3))
    cov[:, 0, 0]              = p[1::FEATURES_PER_K]
    cov[:, 0, 1] = cov[:, 1, 0] = p[2::FEATURES_PER_K]
    cov[:, 0, 2] = cov[:, 2, 0] = p[3::FEATURES_PER_K]
    cov[:, 1, 1]              = p[4::FEATURES_PER_K]
    cov[:, 1, 2] = cov[:, 2, 1] = p[5::FEATURES_PER_K]
    cov[:, 2, 2]              = p[6::FEATURES_PER_K]
    return cov


cov_mats = load_covs(cur_idx)


def draw():
    elev = ax.elev if hasattr(ax, "elev") else 25
    azim = ax.azim if hasattr(ax, "azim") else -60
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
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_zlim(ZLIM)

    draw_brain_outline(ax, bx, by, bz)

    top_k  = ranked_by_cov[:cur_n]
    rest_k = ranked_by_cov[cur_n:]

    if len(rest_k) > 0:
        ax.scatter(cx[rest_k], cy[rest_k], cz[rest_k],
                   c="#888888", s=4, alpha=0.25,
                   edgecolors="none", depthshade=False)

    colors = norm_trace[cur_idx, top_k]
    for rank, k in enumerate(top_k):
        color = cmap(colors[rank])
        scale = ELLIPSOID_SCALE * (0.5 + imp_norm[k] * 0.5)
        X, Y, Z = make_ellipsoid(centers[k], cov_mats[k], scale=scale)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.75,
                        linewidth=0, antialiased=False)

    age = records[cur_idx]["age"]
    ax.set_title(
        f"Subject {cur_idx + 1} / {n_subjects}  |  "
        f"ID: {records[cur_idx]['pid']}  |  Age: {age:.1f} yrs  |  "
        f"Top-{cur_n} components",
        fontsize=9, color="black", pad=6
    )
    fig.canvas.draw_idle()


fig = plt.figure(figsize=(9, 8))
fig.patch.set_facecolor("white")
plt.subplots_adjust(bottom=0.18)
ax = fig.add_subplot(111, projection="3d")

XLIM = (cx.min() - 5, cx.max() + 5)
YLIM = (cy.min() - 5, cy.max() + 5)
ZLIM = (cz.min() - 5, cz.max() + 5)

print("Rendering initial ellipsoids...")
draw()

sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04,
             label="Per-feature normalized covariance trace (0=min, 1=max across subjects)")

ax_age  = plt.axes([0.20, 0.10, 0.65, 0.03])
ax_topn = plt.axes([0.20, 0.05, 0.65, 0.03])
s_age   = Slider(ax_age,  "Age (yrs)",    ages.min(), ages.max(),
                 valinit=ages.min(),
                 valstep=(ages.max() - ages.min()) / (n_subjects - 1))
s_topn  = Slider(ax_topn, "Top-N Gauss", 1, 50, valinit=DEFAULT_TOP_N, valstep=1)


def on_age(val):
    global cur_idx, cov_mats
    cur_idx  = int(np.argmin(np.abs(ages - val)))
    cov_mats = load_covs(cur_idx)
    draw()


def on_topn(val):
    global cur_n
    cur_n = int(val)
    draw()


s_age.on_changed(on_age)
s_topn.on_changed(on_topn)


def on_key(event):
    global cur_idx, cov_mats
    if event.key == "right" and cur_idx < n_subjects - 1:
        cur_idx += 1; cov_mats = load_covs(cur_idx)
        s_age.set_val(ages[cur_idx])
    elif event.key == "left" and cur_idx > 0:
        cur_idx -= 1; cov_mats = load_covs(cur_idx)
        s_age.set_val(ages[cur_idx])


fig.canvas.mpl_connect("key_press_event", on_key)
plt.suptitle("Covariance trace by age  |  Size = model importance  |  Color = per-feature normalized trace",
             fontsize=10, fontweight="bold", color="black", y=0.97)
plt.show()