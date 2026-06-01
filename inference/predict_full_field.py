#!/usr/bin/env python3
"""RAFT prediction vs DICe ground truth on a full SEM frame pair.

Runs the wrapped RAFT model on all 56 systematic tiles for a chosen pair,
stitches the predicted flows into a full-image canvas (averaging overlap),
then plots side-by-side against the absolute-merged DICe ground truth.

Outputs one PNG per requested colormap so we can pick the best-looking one.
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
H, W = 883, 1024  # full SEM frame
TRIM = 17


def load_tile_pair(variant_dir: Path):
    ref = np.asarray(Image.open(variant_dir / "ref.tif"), dtype=np.float32) / 255.0
    deformed = np.asarray(Image.open(variant_dir / "def.tif"), dtype=np.float32) / 255.0
    ref = np.repeat(((ref * 2.0 - 1.0))[None], 3, axis=0)
    deformed = np.repeat(((deformed * 2.0 - 1.0))[None], 3, axis=0)
    with open(variant_dir / "metadata.json") as fp:
        meta = json.load(fp)
    y0, x0 = meta["original_position"]
    return ref, deformed, (int(y0), int(x0))


def infer_pair(model: torch.nn.Module, device: torch.device, pair_dir: Path,
               chunk: int = 8):
    tiles_a, tiles_b, positions = [], [], []
    variant_dirs = sorted(
        p for p in pair_dir.iterdir()
        if p.is_dir() and p.name.endswith("rot000_flip0")
    )
    for vd in variant_dirs:
        a, b, pos = load_tile_pair(vd)
        tiles_a.append(a)
        tiles_b.append(b)
        positions.append(pos)

    a_full = torch.from_numpy(np.stack(tiles_a)).contiguous()
    b_full = torch.from_numpy(np.stack(tiles_b)).contiguous()

    model.eval()
    out_flows = []
    with torch.no_grad():
        for i in range(0, len(positions), chunk):
            a_b = a_full[i:i + chunk].to(device).contiguous()
            b_b = b_full[i:i + chunk].to(device).contiguous()
            res = model(a_b, b_b, iters=12)
            out_flows.append(res["flow"].cpu())
    flow = torch.cat(out_flows, dim=0).numpy()  # (N, 2, 128, 128)

    canvas_sum = np.zeros((2, H, W), dtype=np.float32)
    canvas_count = np.zeros((H, W), dtype=np.float32)
    for k, (y0, x0) in enumerate(positions):
        canvas_sum[:, y0:y0 + TILE, x0:x0 + TILE] += flow[k]
        canvas_count[y0:y0 + TILE, x0:x0 + TILE] += 1.0
    canvas_count = np.maximum(canvas_count, 1.0)
    return canvas_sum / canvas_count[None]


def load_dice_grid(txt_path: Path):
    df = pd.read_csv(txt_path)
    dx = np.full((H, W), np.nan, dtype=np.float32)
    dy = np.full((H, W), np.nan, dtype=np.float32)
    x = df["COORDINATE_X"].to_numpy().astype(int)
    y = df["COORDINATE_Y"].to_numpy().astype(int)
    valid = (df["SIGMA"].to_numpy() >= 0)
    xv, yv = x[valid], y[valid]
    in_bounds = (yv >= 0) & (yv < H) & (xv >= 0) & (xv < W)
    xv, yv = xv[in_bounds], yv[in_bounds]
    dxv = df["DISPLACEMENT_X"].to_numpy()[valid][in_bounds]
    dyv = df["DISPLACEMENT_Y"].to_numpy()[valid][in_bounds]
    dx[yv, xv] = dxv
    dy[yv, xv] = dyv
    return dx, dy


def plot_comparison(sem_ref, sem_def, dice_dx, dice_dy, raft_dx, raft_dy,
                    cmap_name, vmin_x, vmax_x, vmin_y, vmax_y,
                    title_suffix, out_path):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white", alpha=0.0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")

    axes[0, 0].imshow(sem_ref, cmap="gray")
    axes[0, 0].set_title("SEM frame 60 (reference)", fontsize=13)
    axes[0, 0].axis("off")

    axes[1, 0].imshow(sem_def, cmap="gray")
    axes[1, 0].set_title("SEM frame 74 (deformed)", fontsize=13)
    axes[1, 0].axis("off")

    im1 = axes[0, 1].imshow(dice_dx, cmap=cmap, vmin=vmin_x, vmax=vmax_x)
    axes[0, 1].set_title("DICe ground truth  $u_x$  (px)", fontsize=13)
    axes[0, 1].axis("off")

    im2 = axes[0, 2].imshow(raft_dx, cmap=cmap, vmin=vmin_x, vmax=vmax_x)
    axes[0, 2].set_title("RAFT prediction  $u_x$  (px)", fontsize=13)
    axes[0, 2].axis("off")
    cbar1 = fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=10)

    im3 = axes[1, 1].imshow(dice_dy, cmap=cmap, vmin=vmin_y, vmax=vmax_y)
    axes[1, 1].set_title("DICe ground truth  $u_y$  (px)", fontsize=13)
    axes[1, 1].axis("off")

    im4 = axes[1, 2].imshow(raft_dy, cmap=cmap, vmin=vmin_y, vmax=vmax_y)
    axes[1, 2].set_title("RAFT prediction  $u_y$  (px)", fontsize=13)
    axes[1, 2].axis("off")
    cbar2 = fig.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=10)

    fig.suptitle(f"RAFT vs DICe   |   pair f0060 → f0074   |   cmap = {cmap_name}{title_suffix}",
                 fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="f0060_vs_f0074")
    parser.add_argument(
        "--ckpt",
        default=str(Path.home() / "Documents/Capstone/Raft/merged/runs/full_C_systematic_200/best.pt"),
    )
    parser.add_argument(
        "--tiles-root",
        default=str(Path.home() / "Documents/Capstone/dice-automation-tools/training_tiles_128_systematic"),
    )
    parser.add_argument(
        "--dice-root",
        default=str(Path.home() / "Documents/Capstone/dice-automation-tools/demo_eswg007_systematic_absolute_grid"),
    )
    parser.add_argument(
        "--sem-root",
        default=str(Path.home() / "Documents/Capstone/dice-automation-tools/processed_datasets/ESWG007/preprocessed"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path.home() / "Documents/Capstone/Raft/merged/runs"),
    )
    parser.add_argument("--cmaps", nargs="+",
                        default=["RdBu_r", "jet", "coolwarm", "seismic", "viridis"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model and checkpoint...")
    model = RAFTStrainModel().to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    pair_dir = Path(args.tiles_root) / f"pair_{args.pair}"
    print(f"Running inference on {pair_dir.name} ...")
    t0 = time.time()
    raft_flow = infer_pair(model, device, pair_dir)
    print(f"  inference took {time.time() - t0:.2f} s")
    raft_dx, raft_dy = raft_flow[0], raft_flow[1]

    print("Loading DICe ground truth ...")
    dice_dx, dice_dy = load_dice_grid(Path(args.dice_root) / f"DICe_solution_pair_{args.pair}.txt")

    ref_name, def_name = args.pair.split("_vs_")
    sem_ref = np.asarray(Image.open(Path(args.sem_root) / f"frame_{ref_name[1:]}.tif"))
    sem_def = np.asarray(Image.open(Path(args.sem_root) / f"frame_{def_name[1:]}.tif"))

    valid_x = ~np.isnan(dice_dx)
    valid_y = ~np.isnan(dice_dy)
    bound_x = max(abs(np.percentile(dice_dx[valid_x], 1)),
                  abs(np.percentile(dice_dx[valid_x], 99)))
    bound_y = max(abs(np.percentile(dice_dy[valid_y], 1)),
                  abs(np.percentile(dice_dy[valid_y], 99)))
    print(f"  DX bounds: ±{bound_x:.2f} px")
    print(f"  DY bounds: ±{bound_y:.2f} px")

    # Show DICe NaN as white, mask RAFT outside the reliable zone too for parity
    zone = np.zeros((H, W), dtype=bool)
    zone[TRIM:H - TRIM, TRIM:W - TRIM] = True
    dice_dx_m = np.ma.masked_invalid(dice_dx)
    dice_dy_m = np.ma.masked_invalid(dice_dy)
    raft_dx_m = np.ma.masked_array(raft_dx, mask=~zone)
    raft_dy_m = np.ma.masked_array(raft_dy, mask=~zone)

    # Numerical EPE on overlap (DICe-valid AND in reliable zone)
    overlap = valid_x & zone
    epe = np.sqrt(
        (raft_dx[overlap] - dice_dx[overlap]) ** 2
        + (raft_dy[overlap] - dice_dy[overlap]) ** 2
    ).mean()
    suffix = f"   |   mean EPE = {epe:.3f} px"
    print(f"  mean EPE vs DICe = {epe:.3f} px")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cmap_name in args.cmaps:
        out_path = out_dir / f"raft_vs_dice_{args.pair}_{cmap_name}.png"
        plot_comparison(sem_ref, sem_def, dice_dx_m, dice_dy_m,
                        raft_dx_m, raft_dy_m, cmap_name,
                        -bound_x, bound_x, -bound_y, bound_y,
                        suffix, out_path)


if __name__ == "__main__":
    main()
