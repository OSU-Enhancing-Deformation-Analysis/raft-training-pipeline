# RAFT Training Pipeline (CS.057 Course Archive)

Core scripts, configuration, and trained checkpoint for the RAFT optical-flow model used in OSU CS Capstone CS.057 (Applying a Machine Learning-powered Localized Deformation Analyzer for Digital Twin Applications in Materials Testing).

The full RAFT model implementation is maintained by research partner Brock Cloutier in the [ML-Model](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model) repository. This archive contains the capstone team's training runs, inference scripts, and the trained checkpoint used for the final results presented at the OSU Engineering Expo (June 5, 2026).

## Result

![Four-panel comparison of an SEM frame pair: reference frame, deformed frame, DICe ground-truth horizontal displacement field, and the RAFT model prediction. The prediction visually matches the ground truth.](image/SEM.svg)

*Predicted displacement field (right) against the DICe ground truth (third panel) for this frame pair. Qualitatively identical across the reliable interior.*

## Contents

### `scripts/` — Training scripts (HPC-ready package)
- `train_raft.py` — Main training loop.
- `raft_model.py` — RAFT model wrapper around torchvision's `raft_large`.
- `tile_dataset.py` — PyTorch Dataset for 128×128 training tiles.
- `train.sbatch` — Base SLURM submission script (any partition).
- `train_systematic.sbatch` — Long-running variant for the systematic-coverage dataset.
- `setup_hpc.sh` — One-shot HPC environment setup script.
- `requirements.txt` — Python dependencies.
- `README_HPC.md` — HPC-specific setup and usage notes.

### `inference/` — Inference and prediction scripts
- `predict_hero_clean.py` — Runs the trained model on a single image pair to produce a clean visualization of the predicted displacement field.
- `predict_full_field.py` — Runs prediction across all systematic tiles of an image pair and stitches the result into a full-field 849×849 reliable-zone displacement field.

### `analysis/` — Result analysis and plotting
- `plot_ablation_poster.py` — Produces the 3-run ablation plot used on the OSU Engineering Expo poster (Run A: originals only; Run B: random crops + D4 augmentation; Run C: systematic coverage + D4 augmentation).
- `plot_systematic_schema.py` — Renders the systematic-tile coverage diagram (anchored last-tile placement with edge trim).

### `checkpoints/` — Trained model checkpoint
- `best.pt` — Best validation checkpoint from Run C (`full_C_systematic_200`, 200 epochs, `dgxh` H100, SLURM job 20316680). Best epoch 119; validation loss 0.0364; mean EPE 0.024 px on the `f0060_vs_f0090` pair.

### `logs/` — Training run log
- `slurm-full_C_systematic_200-20316680.out` — SLURM stdout from the final training run.

## Usage Summary

Training (on the OSU HPC cluster):
```bash
cd <path-to-this-archive>/scripts
sbatch train_systematic.sbatch <run-name>
```

Inference (locally, against a trained checkpoint):
```bash
python inference/predict_hero_clean.py --checkpoint checkpoints/best.pt --pair <pair-name>
```

For dataset format, training tile generation, and the upstream DICe pipeline, see the [`dice-automation-tools`](https://github.com/OSU-Enhancing-Deformation-Analysis/dice-automation-tools) repository.

## Contributor

- Yanghui Ren (renya@oregonstate.edu)
