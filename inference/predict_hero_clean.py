#!/usr/bin/env python3
"""Clean 3-panel hero figure: SEM | DICe | RAFT prediction.

Transparent background, no colorbars, single-line large labels, tight margins.
Also exports individual single-panel PNGs (transparent, no decoration) so the
panels can be re-arranged freely in Google Draw / Slides.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.enabled = False
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raft_model import RAFTStrainModel  # noqa: E402

TILE = 128
H, W = 883, 1024
TRIM = 17


def load_tile_pair(variant_dir: Path):
    ref = np.asarray(Image.open(variant_dir / "ref.tif"), dtype=np.float32) / 255.0
    deformed = np.asarray(Image.open(variant_dir / "def.tif"), dtype=np.float32) / 255.0
    a = np.repeat(((ref * 2.0 - 1.0))[None], 3, axis=0)
    b = np.repeat(((deformed * 2.0 - 1.0))[None], 3, axis=0)
    meta = json.loads((variant_dir / "metadata.json").read_text())
    y0, x0 = meta["original_position"]
    return a, b, (int(y0), int(x0))


def infer_pair(model, device, pair_dir: Path, chunk=8):
    tiles_a, tiles_b, positions = [], [], []
    for vd in sorted(pair_dir.iterdir()):
        if not (vd.is_dir() and vd.name.endswith("rot000_flip0")):
            continue
        a, b, pos = load_tile_pair(vd)
        tiles_a.append(a); tiles_b.append(b); positions.append(pos)

    a_full = torch.from_numpy(np.stack(tiles_a)).contiguous()
    b_full = torch.from_numpy(np.stack(tiles_b)).contiguous()

    out_flows = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(positions), chunk):
            a_b = a_full[i:i + chunk].to(device).contiguous()
            b_b = b_full[i:i + chunk].to(device).contiguous()
            out_flows.append(model(a_b, b_b, iters=12)["flow"].cpu())
    flow = torch.cat(out_flows, dim=0).numpy()

    canvas_sum = np.zeros((2, H, W), dtype=np.float32)
    canvas_cnt = np.zeros((H, W), dtype=np.float32)
    for k, (y0, x0) in enumerate(positions):
        canvas_sum[:, y0:y0 + TILE, x0:x0 + TILE] += flow[k]
        canvas_cnt[y0:y0 + TILE, x0:x0 + TILE] += 1.0
    return canvas_sum / np.maximum(canvas_cnt, 1.0)[None]


def load_dice_grid(txt_path: Path):
    df = pd.read_csv(txt_path)
    dx = np.full((H, W), np.nan, dtype=np.float32)
    dy = np.full((H, W), np.nan, dtype=np.float32)
    valid = df["SIGMA"].to_numpy() >= 0
    x = df["COORDINATE_X"].to_numpy().astype(int)[valid]
    y = df["COORDINATE_Y"].to_numpy().astype(int)[valid]
    in_b = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y = x[in_b], y[in_b]
    dx[y, x] = df["DISPLACEMENT_X"].to_numpy()[valid][in_b]
    dy[y, x] = df["DISPLACEMENT_Y"].to_numpy()[valid][in_b]
    return dx, dy


def stretch_sem(img, low=2, high=98, gamma=0.85):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [low, high])
    out = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return out ** gamma


def square_crop_reliable(arr):
    """Crop to the largest square within the reliable zone.

    For 883x1024 images the reliable zone is rows [17:866], cols [17:1007]
    (i.e. 849x990).  Largest square is 849x849, taken horizontally centered.
    """
    side = H - 2 * TRIM  # 849
    y_start = TRIM
    y_end = y_start + side
    x_center = W // 2
    x_start = x_center - side // 2
    x_end = x_start + side
    return arr[..., y_start:y_end, x_start:x_end]


def plot_combined(sem, dice_m, raft_m, def_num, cmap_name,
                  vmin, vmax, font_size, out_path):
    """3 square panels of identical size, no labels, tight gap, transparent bg.

    All three arrays are pre-cropped to 849x849 square inside the reliable
    zone.  Each axes fills its assigned figure rectangle exactly; aspect="auto"
    plus matching aspect ratios guarantees zero gutters and equal panel size.
    """
    del def_num, font_size  # labels intentionally omitted; user adds them later
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white", alpha=0.0)

    sem_sq = square_crop_reliable(sem)
    dice_sq = square_crop_reliable(dice_m)
    raft_sq = square_crop_reliable(raft_m)

    side_in = 5.5      # inches per panel (panels are square)
    gap_in = 0.05      # very narrow inter-panel gap

    fig_w = 3 * side_in + 2 * gap_in
    fig_h = side_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0.0)

    arrays = [sem_sq, dice_sq, raft_sq]
    cmaps_used = ["gray", cmap, cmap]
    ranges = [(0.0, 1.0), (vmin, vmax), (vmin, vmax)]

    for i, (arr, cm, (vmn, vmx)) in enumerate(zip(arrays, cmaps_used, ranges)):
        x_in = i * (side_in + gap_in)
        ax = fig.add_axes([x_in / fig_w, 0.0, side_in / fig_w, 1.0])
        ax.imshow(arr, cmap=cm, vmin=vmn, vmax=vmx, aspect="auto")
        ax.axis("off")
        ax.patch.set_alpha(0.0)

    plt.savefig(out_path, dpi=300, transparent=True)
    plt.close()
    print(f"  wrote {out_path.name}")


def plot_single(arr, out_path, cmap_name="gray", vmin=None, vmax=None,
                is_gray=False, dpi=300):
    """Single square panel, no decoration, transparent background, equal size."""
    if is_gray:
        cmap = plt.get_cmap("gray")
    else:
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("white", alpha=0.0)

    arr_sq = square_crop_reliable(arr)

    fig = plt.figure(figsize=(5.5, 5.5))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.patch.set_alpha(0.0)
    ax.imshow(arr_sq, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.axis("off")
    plt.savefig(out_path, dpi=dpi, transparent=True)
    plt.close()
    print(f"  wrote {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="f0060_vs_f0090")
    parser.add_argument("--cmaps", nargs="+", default=["jet", "RdBu_r", "coolwarm"])
    parser.add_argument("--font-size", type=int, default=24)
    parser.add_argument("--ckpt", default=str(Path.home() /
        "Documents/Capstone/Raft/merged/runs/full_C_systematic_200/best.pt"))
    parser.add_argument("--tiles-root", default=str(Path.home() /
        "Documents/Capstone/dice-automation-tools/training_tiles_128_systematic"))
    parser.add_argument("--dice-root", default=str(Path.home() /
        "Documents/Capstone/dice-automation-tools/demo_eswg007_systematic_absolute_grid"))
    parser.add_argument("--sem-root", default=str(Path.home() /
        "Documents/Capstone/dice-automation-tools/processed_datasets/ESWG007/preprocessed"))
    parser.add_argument("--out-dir", default=str(Path.home() /
        "Documents/Capstone/Raft/merged/runs/hero_clean"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pair = args.pair
    ref_name, def_name = pair.split("_vs_")
    def_num = def_name[1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading model + checkpoint ...")
    model = RAFTStrainModel().to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))

    print(f"\n=== {pair} ===")
    pair_dir = Path(args.tiles_root) / f"pair_{pair}"
    t0 = time.time()
    raft_flow = infer_pair(model, device, pair_dir)
    print(f"  RAFT inference: {time.time() - t0:.2f} s")

    dice_dx, _ = load_dice_grid(Path(args.dice_root) / f"DICe_solution_pair_{pair}.txt")
    raft_dx = raft_flow[0]

    sem_raw = np.asarray(Image.open(Path(args.sem_root) / f"frame_{def_num}.tif"))
    sem = stretch_sem(sem_raw, low=2, high=98, gamma=0.85)

    valid = ~np.isnan(dice_dx)
    bound = max(abs(np.percentile(dice_dx[valid], 1)),
                abs(np.percentile(dice_dx[valid], 99)))
    print(f"  Dx bounds: ±{bound:.2f} px")

    zone = np.zeros((H, W), bool); zone[TRIM:H - TRIM, TRIM:W - TRIM] = True

    dice_m = np.ma.masked_invalid(dice_dx)
    raft_m_raw = np.ma.masked_array(raft_dx, mask=~zone)

    raft_dx_masked = raft_dx.copy()
    raft_dx_masked[~valid] = np.nan
    raft_m_masked = np.ma.masked_invalid(raft_dx_masked)

    print("\n-- single panels (cmap-independent) --")
    plot_single(sem, out_dir / f"panel_sem_{def_num}.png", is_gray=True)

    print("\n-- combined + per-cmap panels --")
    for cmap_name in args.cmaps:
        plot_combined(sem, dice_m, raft_m_raw, def_num, cmap_name,
                      -bound, bound, args.font_size,
                      out_dir / f"combined_{pair}_{cmap_name}.png")
        plot_combined(sem, dice_m, raft_m_masked, def_num, cmap_name,
                      -bound, bound, args.font_size,
                      out_dir / f"combined_{pair}_{cmap_name}_MASKED.png")
        plot_single(dice_m, out_dir / f"panel_dice_{pair}_{cmap_name}.png",
                    cmap_name=cmap_name, vmin=-bound, vmax=bound)
        plot_single(raft_m_raw, out_dir / f"panel_raft_{pair}_{cmap_name}.png",
                    cmap_name=cmap_name, vmin=-bound, vmax=bound)
        plot_single(raft_m_masked,
                    out_dir / f"panel_raft_{pair}_{cmap_name}_MASKED.png",
                    cmap_name=cmap_name, vmin=-bound, vmax=bound)


if __name__ == "__main__":
    main()
