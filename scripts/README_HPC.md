# RAFT HPC Training Quickstart

## Files in this package

- `raft_model.py`, `tile_dataset.py`, `train_raft.py` — model + training code
- `plot_loss.py`, `plot_compare.py`, `plot_compare_updates.py` — analysis
- `requirements.txt` — Python deps (PyTorch installed separately)
- `setup_hpc.sh` — one-time env setup
- `train.sbatch` — SLURM job script

## Step 1 — Upload to HPC

The training tiles dataset (~1.4 GB) is in a separate archive.
Both the code package (this folder) and the tiles need to be on HPC.

Suggested layout:
```
$HOME/hpc-share/raft/
├── hpc_package/                  ← unzipped from hpc_package.zip
└── training_tiles_128_v2/        ← unzipped from training_tiles_128_v2.tar.gz
```

## Step 2 — One-time env setup (on submit node, has internet)

```bash
cd ~/hpc-share/raft/hpc_package
bash setup_hpc.sh
```

This creates `~/raft-env/` with PyTorch + deps. If your HPC's CUDA is not 12.x,
edit `setup_hpc.sh` to change the `--index-url` line (e.g. cu118 for CUDA 11.8).

## Step 3 — Quick sanity job (30 epoch, originals)

```bash
cd ~/hpc-share/raft/hpc_package
sbatch train.sbatch sanity_A_30 --epochs 30 --originals-only --batch-size 16
squeue -u $USER     # check status
tail -f logs/raft-train-<jobid>.out
```

Expected: ~15-20 min on H100. Confirms env + data + GPU all working.

## Step 4 — Full ablation (200 epoch each)

After sanity passes:

```bash
sbatch train.sbatch full_A_200 --epochs 200 --batch-size 16 --originals-only
sbatch train.sbatch full_B_200 --epochs 200 --batch-size 16
```

Both jobs run in parallel if cluster has 2 GPUs free. Outputs land in
`~/hpc-share/raft/runs/full_A_200/` and `~/hpc-share/raft/runs/full_B_200/`.

## Step 5 — Compare results

After both finish, scp the .log files back to the PC and run:

```bash
python plot_compare.py full_A_200/train.log full_B_200/train.log \
    --labels "A: originals" "B: augmented"
python plot_compare_updates.py full_A_200/train.log full_B_200/train.log \
    --labels "A: originals" "B: augmented"
```

## Common issues

- **`module load python` fails**: try `module avail python` to see versions, or
  use system `python3` directly.
- **PyTorch CUDA mismatch**: check node's CUDA with `nvidia-smi` (top-right),
  match wheel index URL accordingly (cu118, cu121, cu124).
- **Job pending forever**: partition/constraint may not exist on your cluster.
  Run `sinfo -o "%P %G %N"` to find available GPU partitions and edit
  `train.sbatch` SBATCH lines.
- **OOM at bs=16**: drop to `--batch-size 8` (mirrors the RTX 5080 run on PC).
