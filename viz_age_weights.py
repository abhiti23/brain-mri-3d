"""
viz_age_weights.py — Age slider, per-feature normalized intensity

Think of the data as a table: rows = subjects (sorted young to old),
columns = weight features sorted by importance (most to least).

For each column (Gaussian component), values are normalized to [0,1]
using that column's own min and max across ALL subjects.
So color tells you: for THIS region, where does this subject rank
relative to everyone else?

  - Size   = importance from feature_importance.csv (fixed)
  - color = this subject's per-feature normalized intensity (0=min, 1=max)
             Yellow = this subject has low GM here relative to others
             Red    = this subject has high GM here relative to others

Top-N slider shows only the N most important weight components.

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
from viz_utils     import load_brain_outline, draw_brain_outline
from viz_reg_utils import load_component_scores

GMM_DIR   = "training/train_gmm_weighted_outputs"
LABELS    = "training/train_labels/participants.tsv"
CENTERS   = "centers.npy"
AVG_BRAIN = "group_average_gm_T1w.npy"
IMP_CSV   = "feature_importance.csv"

K              = 500
FEATURES_PER_K = 7
DEFAULT_TOP_N  = 50

centers = np.load(CENTERS)
cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]

weight_imp, cov_imp, total_imp = load_component_scores(IMP_CSV)
imp_norm         = weight_imp / (weight_imp.max() + 1e-12)
sizes_all        = 8 + imp_norm * 150
ranked_by_weight = np.argsort(weight_imp)[::-1]   # most to least important

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

# --- build the table: shape (n_subjects, K) ---
# rows = subjects sorted young to old, columns = weight features
print("Building subject x feature table...")
raw_table = np.zeros((n_subjects, K))
for i, r in enumerate(records):
    p = np.load(r["path"])
    raw_table[i] = p[0::FEATURES_PER_K]

# --- per-feature (column-wise) normalisation to [0, 1] ---
col_min  = raw_table.min(axis=0)   # (K,) min across subjects per feature
col_max  = raw_table.max(axis=0)   # (K,) max across subjects per feature
col_range = col_max - col_min
col_range[col_range == 0] = 1      # avoid divide by zero for constant features

norm_table = (raw_table - col_min) / col_range   # (n_subjects, K), each col in [0,1]
print("Done.")

print("Loading brain outline...")
bx, by, bz = load_brain_outline(AVG_BRAIN)
print("Ready.")

cur_idx = 0
cur_n   = DEFAULT_TOP_N


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

    top_k  = ranked_by_weight[:cur_n]
    rest_k = ranked_by_weight[cur_n:]

    # dim background for excluded components
    if len(rest_k) > 0:
        ax.scatter(cx[rest_k], cy[rest_k], cz[rest_k],
                   c="#888888", s=4, alpha=0.25,
                   edgecolors="none", depthshade=False)

    # top-N: size by importance (fixed), color by per-feature normalized value
    colors = norm_table[cur_idx, top_k]   # this subject's position in [0,1] per feature
    ax.scatter(cx[top_k], cy[top_k], cz[top_k],
               c=colors, s=sizes_all[top_k], cmap="YlOrRd",
               alpha=0.85, edgecolors="none", depthshade=True,
               vmin=0, vmax=1)

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

draw()

sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04,
             label="Per-feature normalized intensity (0=min across subjects, 1=max)")

ax_age  = plt.axes([0.20, 0.10, 0.65, 0.03])
ax_topn = plt.axes([0.20, 0.05, 0.65, 0.03])
s_age   = Slider(ax_age,  "Age (yrs)",    ages.min(), ages.max(),
                 valinit=ages.min(),
                 valstep=(ages.max() - ages.min()) / (n_subjects - 1))
s_topn  = Slider(ax_topn, "Top-N Gauss", 1, K, valinit=DEFAULT_TOP_N, valstep=1)


def on_age(val):
    global cur_idx
    cur_idx = int(np.argmin(np.abs(ages - val)))
    draw()


def on_topn(val):
    global cur_n
    cur_n = int(val)
    draw()


s_age.on_changed(on_age)
s_topn.on_changed(on_topn)


def on_key(event):
    global cur_idx
    if event.key == "right" and cur_idx < n_subjects - 1:
        cur_idx += 1; s_age.set_val(ages[cur_idx])
    elif event.key == "left" and cur_idx > 0:
        cur_idx -= 1; s_age.set_val(ages[cur_idx])


fig.canvas.mpl_connect("key_press_event", on_key)
plt.suptitle("GM intensity by age  |  Size = model importance  |  color = per-feature normalized intensity",
             fontsize=10, fontweight="bold", color="black", y=0.97)
plt.show()