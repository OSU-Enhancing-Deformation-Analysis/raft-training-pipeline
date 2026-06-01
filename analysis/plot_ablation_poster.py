"""
Poster-style ablation figures v7
--------------------------------
Outputs:
  1) ablation_epoch_poster.png              transparent PNG
  2) ablation_update_poster.png             transparent PNG
  3) ablation_combined_poster.png           transparent PNG
  4) ablation_combined_poster_white.pdf     white-background PDF reference

Main changes from v6:
  - Combined figure uses shorter subplot titles to avoid overlap.
  - Combined figure uses one shared legend.
  - Equal-update budget label moved lower to avoid legend overlap.
  - PNG exports are transparent; PDF export is white background.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="poster", font_scale=0.85)
except ImportError:
    pass

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =========================================================
# Paths
# =========================================================
LOG_DIR = Path.home() / "Documents/Capstone/Raft/merged/runs"

A_LOG = LOG_DIR / "slurm-full_A_200-20246717.out"
B_LOG = LOG_DIR / "slurm-full_B_200-20246718.out"
C_LOG = LOG_DIR / "slurm-full_C_systematic_200-20316680.out"

OUT_EPOCH_PNG = LOG_DIR / "ablation_epoch_poster.png"
OUT_UPDATE_PNG = LOG_DIR / "ablation_update_poster.png"
OUT_COMBINED_PNG = LOG_DIR / "ablation_combined_poster.png"
OUT_COMBINED_PDF = LOG_DIR / "ablation_combined_poster_white.pdf"


# =========================================================
# Parsing settings
# =========================================================
PATTERN = re.compile(
    r"^epoch=(\d+) train_loss=([\d.]+) val_loss=([\d.]+) lr=([\deE.+-]+) time=([\d.]+)s"
)

A_UPD_PER_EPOCH = 50
B_UPD_PER_EPOCH = 397
C_UPD_PER_EPOCH = 694

SPIKE_THRESHOLD = 0.062


# =========================================================
# Style settings
# =========================================================
# COLOR_A = "#00A2FF"
# COLOR_B = "#FFAE00"
# COLOR_C = "#8138FF"
# COLOR_A = "#0096FF"   # brighter blue
# COLOR_B = "#FFB000"   # vivid amber / yellow-orange
# COLOR_C = "#9B5DE5"   # brighter purple

COLOR_A = "#0096FF"   # bright blue
COLOR_B = "#FF6B00"   # vivid orange-red
COLOR_C = "#9B5DE5"   # bright purple

RAW_ALPHA = 0.10
RAW_LW = 0.9
SMOOTH_LW = 3.2
MARKER_SIZE = 95

TITLE_FS = 28
COMBINED_TITLE_FS = 24
SUPTITLE_FS = 28
LABEL_FS = 24
TICK_FS = 18
LEGEND_FS = 16
COMBINED_LEGEND_FS = 16
VALUE_FS = 18
EQ_LABEL_FS = 15

EPOCH_YLIM = (0.033, 0.062)
UPDATE_YLIM = (0.033, 0.062)
UPDATE_XMAX = 40000


# =========================================================
# Utility functions
# =========================================================
def parse_log(path):
    epochs, val = [], []

    with open(path) as f:
        for line in f:
            m = PATTERN.match(line.strip())
            if m:
                epochs.append(int(m.group(1)))
                val.append(float(m.group(3)))

    if not epochs:
        raise ValueError(f"No epoch records found in: {path}")

    return np.array(epochs), np.array(val)


def mask_spikes(values, threshold):
    values = values.astype(float)
    bad = values > threshold
    n_bad = int(bad.sum())

    raw = values.copy()
    raw[bad] = np.nan

    clean = values.copy()
    if n_bad > 0:
        idx = np.arange(len(values))
        clean[bad] = np.interp(idx[bad], idx[~bad], values[~bad])

    return raw, clean, n_bad


def smooth_curve(values):
    if HAS_SCIPY:
        return gaussian_filter1d(values, sigma=2.0, mode="nearest")

    window = 10
    pad = window // 2
    padded = np.concatenate([
        np.full(pad, values[0]),
        values,
        np.full(pad, values[-1]),
    ])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def interp_value(x, y, target_x):
    return float(np.interp(target_x, x, y))


def style_axes(ax):
    ax.set_facecolor("none")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=1.0,
        labelsize=TICK_FS,
    )

    ax.grid(True, alpha=0.22)


def annotate_value(ax, x, y, text, color, dx=10, dy=8, ha="left", va="bottom"):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        color=color,
        fontsize=VALUE_FS,
        fontweight="bold",
    )


def save_transparent_png(fig, out_path):
    fig.patch.set_alpha(0.0)
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    print(f"Saved: {out_path}")


def save_white_pdf(fig, out_path):
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    fig.savefig(
        out_path,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    print(f"Saved: {out_path}")


# =========================================================
# Load and process data
# =========================================================
ea, va = parse_log(A_LOG)
eb, vb = parse_log(B_LOG)
ec, vc = parse_log(C_LOG)

va_raw, va_clean, n_a = mask_spikes(va, SPIKE_THRESHOLD)
vb_raw, vb_clean, n_b = mask_spikes(vb, SPIKE_THRESHOLD)
vc_raw, vc_clean, n_c = mask_spikes(vc, SPIKE_THRESHOLD)

va_s = smooth_curve(va_clean)
vb_s = smooth_curve(vb_clean)
vc_s = smooth_curve(vc_clean)

upd_a = ea * A_UPD_PER_EPOCH
upd_b = eb * B_UPD_PER_EPOCH
upd_c = ec * C_UPD_PER_EPOCH

a_max_upd = int(upd_a[-1])

a_final = float(np.nanmean(va_clean[-5:]))
b_final = float(np.nanmean(vb_clean[-5:]))
c_final = float(np.nanmean(vc_clean[-5:]))

a_at_equal = a_final
b_at_equal = interp_value(upd_b, vb_s, a_max_upd)
c_at_equal = interp_value(upd_c, vc_s, a_max_upd)

gap_ca_final = (a_final - c_final) / a_final * 100
gap_ca_equal = (a_at_equal - c_at_equal) / a_at_equal * 100

label_a = "A: random crop, no aug (992)"
label_b = "B: random crop, D4 aug (7,936)"
label_c = "C: systematic crop, D4 aug (13,872)"


# =========================================================
# Plot helpers
# =========================================================
def draw_epoch_plot(ax, title="Validation error over training epochs", show_legend=True):
    ax.plot(ea, va_raw, color=COLOR_A, alpha=RAW_ALPHA, linewidth=RAW_LW)
    ax.plot(eb, vb_raw, color=COLOR_B, alpha=RAW_ALPHA, linewidth=RAW_LW)
    ax.plot(ec, vc_raw, color=COLOR_C, alpha=RAW_ALPHA, linewidth=RAW_LW)

    line_a, = ax.plot(ea, va_s, color=COLOR_A, linewidth=SMOOTH_LW, label=label_a)
    line_b, = ax.plot(eb, vb_s, color=COLOR_B, linewidth=SMOOTH_LW, label=label_b)
    line_c, = ax.plot(ec, vc_s, color=COLOR_C, linewidth=SMOOTH_LW, label=label_c)

    ax.scatter([200], [a_final], color=COLOR_A, s=MARKER_SIZE, zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.scatter([200], [b_final], color=COLOR_B, s=MARKER_SIZE, zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.scatter([200], [c_final], color=COLOR_C, s=MARKER_SIZE, zorder=5,
               edgecolor="white", linewidth=1.4)

    annotate_value(ax, 200, a_final, f"{a_final:.4f}", COLOR_A,
                   dx=-12, dy=10, ha="right", va="bottom")
    annotate_value(ax, 200, b_final, f"{b_final:.4f}", COLOR_B,
                   dx=-12, dy=10, ha="right", va="bottom")
    annotate_value(ax, 200, c_final, f"{c_final:.4f}", COLOR_C,
                   dx=-12, dy=10, ha="right", va="bottom")

    ax.set_title(title, fontsize=TITLE_FS, pad=20)
    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Validation error (px)", fontsize=LABEL_FS)

    ax.set_xlim(0, 208)
    ax.set_ylim(*EPOCH_YLIM)

    style_axes(ax)

    if show_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=LEGEND_FS)

    return line_a, line_b, line_c


def draw_update_plot(ax, title="Validation error over cumulative updates", show_legend=True):
    ax.plot(upd_a, va_raw, color=COLOR_A, alpha=RAW_ALPHA, linewidth=RAW_LW)
    ax.plot(upd_b, vb_raw, color=COLOR_B, alpha=RAW_ALPHA, linewidth=RAW_LW)
    ax.plot(upd_c, vc_raw, color=COLOR_C, alpha=RAW_ALPHA, linewidth=RAW_LW)

    line_a, = ax.plot(upd_a, va_s, color=COLOR_A, linewidth=SMOOTH_LW, label=label_a)
    line_b, = ax.plot(upd_b, vb_s, color=COLOR_B, linewidth=SMOOTH_LW, label=label_b)
    line_c, = ax.plot(upd_c, vc_s, color=COLOR_C, linewidth=SMOOTH_LW, label=label_c)

    ax.axvline(
        a_max_upd,
        color="0.45",
        linestyle=":",
        linewidth=1.8,
        alpha=0.75,
        zorder=1,
    )

    ax.scatter([a_max_upd], [a_at_equal], color=COLOR_A, s=MARKER_SIZE, zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.scatter([a_max_upd], [b_at_equal], color=COLOR_B, s=MARKER_SIZE, zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.scatter([a_max_upd], [c_at_equal], color=COLOR_C, s=MARKER_SIZE, zorder=5,
               edgecolor="white", linewidth=1.4)

    annotate_value(ax, a_max_upd, a_at_equal, f"{a_at_equal:.4f}", COLOR_A,
                   dx=12, dy=10, ha="left", va="bottom")
    annotate_value(ax, a_max_upd, b_at_equal, f"{b_at_equal:.4f}", COLOR_B,
                   dx=12, dy=10, ha="left", va="bottom")
    annotate_value(ax, a_max_upd, c_at_equal, f"{c_at_equal:.4f}", COLOR_C,
                   dx=12, dy=-8, ha="left", va="top")

    # Lowered to avoid legend overlap.
    ax.text(
        a_max_upd + 1200,
        0.0550,
        "Equal-update\nbudget",
        fontsize=EQ_LABEL_FS,
        color="0.35",
        ha="left",
        va="center",
        style="italic",
    )

    ax.set_title(title, fontsize=TITLE_FS, pad=20)
    ax.set_xlabel("Cumulative gradient updates", fontsize=LABEL_FS)
    ax.set_ylabel("Validation error (px)", fontsize=LABEL_FS)

    ax.set_xlim(0, UPDATE_XMAX)
    ax.set_ylim(*UPDATE_YLIM)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: "0" if x == 0 else f"{int(x/1000)}k")
    )

    style_axes(ax)

    if show_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=LEGEND_FS)

    return line_a, line_b, line_c


# =========================================================
# 1) Separate epoch PNG
# =========================================================
fig1, ax1 = plt.subplots(figsize=(8.8, 6.2), dpi=140)
fig1.patch.set_alpha(0.0)
draw_epoch_plot(ax1)
save_transparent_png(fig1, OUT_EPOCH_PNG)
plt.close(fig1)


# =========================================================
# 2) Separate update PNG
# =========================================================
fig2, ax2 = plt.subplots(figsize=(8.8, 6.2), dpi=140)
fig2.patch.set_alpha(0.0)
draw_update_plot(ax2)
save_transparent_png(fig2, OUT_UPDATE_PNG)
plt.close(fig2)


# =========================================================
# 3) Combined transparent PNG
# =========================================================
fig3, (ax3, ax4) = plt.subplots(
    1,
    2,
    figsize=(18.5, 6.6),
    dpi=140,
)

fig3.patch.set_alpha(0.0)

handles = draw_epoch_plot(
    ax3,
    title="(a) Epoch-based comparison",
    show_legend=False,
)

draw_update_plot(
    ax4,
    title="(b) Update-based comparison",
    show_legend=False,
)

# Use smaller title size for combined subplot titles.
ax3.title.set_fontsize(COMBINED_TITLE_FS)
ax4.title.set_fontsize(COMBINED_TITLE_FS)

fig3.suptitle(
    "Validation error under augmentation and crop strategies",
    fontsize=SUPTITLE_FS,
    y=1.03,
)

fig3.legend(
    handles=handles,
    labels=[label_a, label_b, label_c],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=3,
    frameon=False,
    fontsize=COMBINED_LEGEND_FS,
)

fig3.subplots_adjust(
    top=0.78,
    wspace=0.25,
)

save_transparent_png(fig3, OUT_COMBINED_PNG)


# =========================================================
# 4) Combined white-background PDF reference
# =========================================================
# Important: set white backgrounds for PDF reference.
fig3.patch.set_facecolor("white")
fig3.patch.set_alpha(1.0)
ax3.set_facecolor("white")
ax4.set_facecolor("white")

save_white_pdf(fig3, OUT_COMBINED_PDF)
plt.close(fig3)


# =========================================================
# Print useful numbers for poster text
# =========================================================
print()
print("Final values, averaged over last 5 epochs:")
print(f"  A = {a_final:.4f}")
print(f"  B = {b_final:.4f}")
print(f"  C = {c_final:.4f}")

print()
print(f"Equal-update budget: {a_max_upd:,} updates")
print(f"  A = {a_at_equal:.4f}")
print(f"  B = {b_at_equal:.4f}")
print(f"  C = {c_at_equal:.4f}")

print()
print("Poster caption facts:")
print(f"  C vs A at 200 epochs: {gap_ca_final:.1f}% lower validation error")
print(f"  C vs A at equal-update budget: {gap_ca_equal:.1f}% lower validation error")

print()
print(f"Outliers masked above {SPIKE_THRESHOLD}: A={n_a}, B={n_b}, C={n_c}")