"""Dataset adapter for ESWG007 RAFT training tiles.

Each tile variant directory holds ``ref.tif`` and ``def.tif`` (uint8
grayscale, 128x128) and ``flow.npy`` (float32, ``(128, 128, 2)``,
NaN where DICe correlation failed).  Images are lazily loaded per
``__getitem__`` and normalized to ``[-1, 1]`` float32, then repeated
across 3 channels so the RAFT wrapper can stay a pass-through on
intensity.
"""

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

LOGGER = logging.getLogger(__name__)
TILE_RE = re.compile(r"^tile_(\d{4})_rot\d{3}_flip[01]$")
ORIGINAL_RE = re.compile(r"^tile_\d{4}_rot000_flip0$")


class TileDataset(Dataset):
    """Lazy-loading dataset for 128x128 SEM tile image pairs and flow fields.

    Parameters
    ----------
    tiles_root:
        Root directory containing ``pair_*`` subdirectories.
    pair_filter:
        Optional substring match; only ``pair_*`` directories whose name
        contains the substring are kept.
    originals_only:
        If ``True``, keep only variant directories matching
        ``tile_*_rot000_flip0`` (the 992-tile pre-augmentation set).
    """

    def __init__(
        self,
        tiles_root: Path,
        pair_filter: Optional[str] = None,
        originals_only: bool = False,
    ) -> None:
        self.tiles_root = Path(tiles_root)
        self.pair_filter = pair_filter
        self.originals_only = originals_only

        if not self.tiles_root.is_dir():
            raise FileNotFoundError(f"Tiles root does not exist: {self.tiles_root}")

        self.samples: list[tuple[Path, Path]] = []
        pair_dirs = sorted(path for path in self.tiles_root.iterdir() if path.is_dir())
        for pair_dir in pair_dirs:
            if not pair_dir.name.startswith("pair_"):
                continue
            if pair_filter is not None and pair_filter not in pair_dir.name:
                continue

            variant_dirs = sorted(path for path in pair_dir.iterdir() if path.is_dir())
            for variant_dir in variant_dirs:
                if originals_only and ORIGINAL_RE.match(variant_dir.name) is None:
                    continue
                if TILE_RE.match(variant_dir.name) is None:
                    continue
                self.samples.append((pair_dir, variant_dir))

        LOGGER.info(
            "TileDataset variants kept: %d (pair_filter=%s, originals_only=%s)",
            len(self.samples),
            pair_filter,
            originals_only,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        _, variant_dir = self.samples[index]
        ref = _load_image(variant_dir / "ref.tif")
        deformed = _load_image(variant_dir / "def.tif")
        flow = np.load(variant_dir / "flow.npy").astype(np.float32, copy=False)

        valid = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
        mask = valid.astype(np.float32)[None, :, :]
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        disp = flow.transpose(2, 0, 1)

        return {
            "a": torch.from_numpy(ref),
            "b": torch.from_numpy(deformed),
            "disp": torch.from_numpy(np.ascontiguousarray(disp)),
            "mask": torch.from_numpy(mask),
        }

    def group_key(self, index: int) -> tuple[str, str]:
        """Return ``(pair_name, tile_index)`` for the given flat index.

        Exposed publicly so callers (e.g. per-pair evaluation or the
        split helper) can partition the dataset by source tile without
        re-parsing variant directory names.
        """
        pair_dir, variant_dir = self.samples[index]
        match = TILE_RE.match(variant_dir.name)
        if match is None:
            raise ValueError(f"Unexpected tile variant directory name: {variant_dir.name}")
        return pair_dir.name, match.group(1)


def split_by_tile(
    dataset: TileDataset,
    val_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Split ``dataset`` into train/val Subsets by ``(pair, tile_index)`` group.

    Splitting flat indices would leak augmentation variants of the same
    underlying tile across train and val.  Grouping by ``(pair, tile_index)``
    and allocating whole groups preserves the generalization test.  In the
    originals-only regime each group has exactly one variant, so the same
    code path handles both regimes without a special case.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        groups[dataset.group_key(index)].append(index)

    group_keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_val = max(1, int(round(len(group_keys) * val_fraction)))
    if n_val >= len(group_keys):
        n_val = len(group_keys) - 1

    val_groups = set(group_keys[:n_val])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for key in group_keys:
        target = val_indices if key in val_groups else train_indices
        target.extend(groups[key])

    train_indices.sort()
    val_indices.sort()

    LOGGER.info(
        "Split by tile: groups=%d train_groups=%d val_groups=%d "
        "train_variants=%d val_variants=%d",
        len(group_keys),
        len(group_keys) - len(val_groups),
        len(val_groups),
        len(train_indices),
        len(val_indices),
    )

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _load_image(path: Path) -> np.ndarray:
    """Load a uint8 grayscale TIF and return ``(3, H, W)`` float32 in ``[-1, 1]``."""
    with Image.open(path) as image:
        img01 = np.asarray(image, dtype=np.float32) / 255.0
    img = img01 * 2.0 - 1.0
    img = np.repeat(img[None, :, :], 3, axis=0)
    return np.ascontiguousarray(img)
