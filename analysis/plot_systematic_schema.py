#!/usr/bin/env python3
"""Systematic tile coverage schematic — partial-schema style (matches
trim_example.svg layout).  Shows 5 tiles across top + 5 down the left,
with dotted-arrow "..." callouts implying "the same pattern continues".

Outputs two versions:
  - schema_..._labeled.png   : full annotations + legend
  - schema_..._nolabels.png  : shapes only, transparent bg (text in Slides)
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

# Colors (poster-matching yellow + bold red, Okabe-Ito green for reliable)
YELLOW = "#f5c842"   # poster accent yellow
RED = "#d62728"      # bold red (was orange)
GREEN = "#009e73"    # Okabe-Ito green

# Real geometry
H, W = 883, 1024
TILE = 128
REAL_TRIM = 17

# Trim exaggerated for schematic visibility
DISPLAY_TRIM = 40

# Partial schema: how many tiles to draw in each direction before "..."
N_TOP = 5
N_LEFT = 5


def stretch_sem(img, low=2, high=98, gamma=0.85):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [low, high])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma


def draw_excluded_ring(ax, trim, alpha=0.45):
    kw = dict(facecolor=RED, alpha=alpha, edgecolor="none")
    ax.add_patch(Rectangle((0, 0), W, trim, **kw))
    ax.add_patch(Rectangle((0, H - trim), W, trim, **kw))
    ax.add_patch(Rectangle((0, trim), trim, H - 2 * trim, **kw))
    ax.add_patch(Rectangle((W - trim, trim), trim, H - 2 * trim, **kw))


def draw_continuation_arrow(ax, x0, y0, dx, dy, color, dot_count=3,
                            seg_len=60, dot_spacing=18, dot_radius=5,
                            lw=2.8):
    """Solid arrow from (x0,y0) toward direction (dx,dy), then 3 dots.

    dx, dy give a unit-ish direction; we scale to seg_len + dots.
    """
    norm = (dx ** 2 + dy ** 2) ** 0.5
    ux, uy = dx / norm, dy / norm

    # Solid arrow segment
    head_x = x0 + ux * seg_len
    head_y = y0 + uy * seg_len
    ax.annotate("", xy=(head_x, head_y), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    # Dots after the arrow
    for i in range(1, dot_count + 1):
        cx = head_x + ux * (dot_spacing * i)
        cy = head_y + uy * (dot_spacing * i)
        ax.plot(cx, cy, marker="o", color=color, markersize=dot_radius,
                markeredgecolor=color)


def plot_schema(sem, with_text, out_path):
    fig, ax = plt.subplots(figsize=(11, 9.5))
    fig.patch.set_alpha(0.0)
    ax.imshow(sem, cmap="gray", vmin=0, vmax=1)

    draw_excluded_ring(ax, DISPLAY_TRIM)

    # Reliable zone (dashed green)
    ax.add_patch(Rectangle(
        (DISPLAY_TRIM, DISPLAY_TRIM),
        W - 2 * DISPLAY_TRIM, H - 2 * DISPLAY_TRIM,
        facecolor="none", edgecolor=GREEN, linewidth=2.4,
        linestyle=(0, (7.4, 3.2))))

    # Partial tile schema: top row (5 tiles) + left column (5 tiles)
    # corner tile shared.  Uniform stride = TILE so they are all true squares.
    drawn = set()
    for i in range(N_TOP):
        x0 = DISPLAY_TRIM + i * TILE
        y0 = DISPLAY_TRIM
        drawn.add((y0, x0))
    for i in range(N_LEFT):
        x0 = DISPLAY_TRIM
        y0 = DISPLAY_TRIM + i * TILE
        drawn.add((y0, x0))

    for (y0, x0) in drawn:
        ax.add_patch(Rectangle(
            (x0, y0), TILE, TILE,
            facecolor="none", edgecolor=YELLOW, linewidth=3.0))

    # Highlight the tile near the crack (top row, 4th column)
    hl_x = DISPLAY_TRIM + 3 * TILE
    hl_y = DISPLAY_TRIM
    ax.add_patch(Rectangle(
        (hl_x, hl_y), TILE, TILE,
        facecolor=YELLOW, alpha=0.30, edgecolor=YELLOW, linewidth=3.5))

    # Continuation arrows ("..." callouts)
    # Top row: arrow from right edge of last top tile, going right
    top_end_x = DISPLAY_TRIM + N_TOP * TILE
    arrow_y_mid = DISPLAY_TRIM + TILE // 2
    draw_continuation_arrow(ax, top_end_x + 10, arrow_y_mid, dx=1, dy=0,
                            color=YELLOW)

    # Left column: arrow from bottom edge of last left tile, going down
    left_end_y = DISPLAY_TRIM + N_LEFT * TILE
    arrow_x_mid = DISPLAY_TRIM + TILE // 2
    draw_continuation_arrow(ax, arrow_x_mid, left_end_y + 10, dx=0, dy=1,
                            color=YELLOW)

    if with_text:
        # Highlight tile label
        ax.annotate(
            "128 × 128 training tile",
            xy=(hl_x + TILE // 2, hl_y + TILE),
            xytext=(hl_x + TILE + 50, hl_y + TILE + 130),
            fontsize=15,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="black", linewidth=1.2),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

        # Trim band label
        ax.annotate(
            "subset_size / 2  ≈ 17 px (excluded)",
            xy=(W - DISPLAY_TRIM // 2, H // 2),
            xytext=(W - 340, H // 2),
            fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor="black", linewidth=1.0),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))

        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=RED, alpha=0.45, edgecolor="none"),
            plt.Line2D([0], [0], color=GREEN, linewidth=2.4,
                       linestyle=(0, (7.4, 3.2))),
            Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=YELLOW, linewidth=3.0),
        ]
        legend_labels = [
            "Excluded border (DICe correlation unreliable)",
            "Reliable interior region",
            "128 × 128 training tiles (schematic layout)",
        ]
        ax.legend(legend_handles, legend_labels, loc="lower right",
                  fontsize=11, framealpha=0.95)

    ax.set_xlim(-2, W + 2)
    ax.set_ylim(H + 2, -2)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    plt.savefig(out_path, dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"  wrote {out_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame", default="0090")
    p.add_argument("--out-dir", default=str(Path.home() /
        "Documents/Capstone/Raft/merged/runs/schema"))
    p.add_argument("--sem-root", default=str(Path.home() /
        "Documents/Capstone/dice-automation-tools/processed_datasets/ESWG007/preprocessed"))
    a = p.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sem_raw = np.asarray(Image.open(Path(a.sem_root) / f"frame_{a.frame}.tif"))
    sem = stretch_sem(sem_raw)

    plot_schema(sem, with_text=True,
                out_path=out / f"schema_frame{a.frame}_labeled.png")
    plot_schema(sem, with_text=False,
                out_path=out / f"schema_frame{a.frame}_nolabels.png")


if __name__ == "__main__":
    main()
