#!/usr/bin/env bash
#SBATCH --job-name=raft-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# Adjust the partition / constraint below for your cluster — common options:
#   --partition=gpu            (generic GPU partition)
#   --constraint=h100          (request H100 specifically)
#   --partition=gpu-h100       (cluster-specific named partition)
# Check with: sinfo -o "%P %G %N" | grep -i gpu
#SBATCH --partition=gpu
#SBATCH --constraint=h100

# Usage:
#   sbatch train.sbatch <run_name> [extra train_raft.py args...]
#
# Examples:
#   # 30-epoch sanity run, originals only
#   sbatch train.sbatch sanity_A_30 --epochs 30 --originals-only --batch-size 16
#
#   # 200-epoch full run, augmented
#   sbatch train.sbatch full_B_200 --epochs 200 --batch-size 16
#
#   # 200-epoch full run, originals only (control)
#   sbatch train.sbatch full_A_200 --epochs 200 --batch-size 16 --originals-only

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch train.sbatch <run_name> [extra args...]"
    exit 1
fi

RUN_NAME="$1"
shift
EXTRA_ARGS=("$@")

# === Paths (edit these to match your HPC layout) ===
PACKAGE_DIR="${PACKAGE_DIR:-$HOME/hpc-share/raft/hpc_package}"
TILES_DIR="${TILES_DIR:-$HOME/hpc-share/raft/training_tiles_128_v2}"
ENV_DIR="${ENV_DIR:-$HOME/raft-env}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/hpc-share/raft/runs/$RUN_NAME}"

mkdir -p logs "$OUTPUT_DIR"

# === Activate env ===
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

cd "$PACKAGE_DIR"

# === Sanity: GPU visible ===
nvidia-smi || echo "WARNING: nvidia-smi not available"
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# === Train ===
echo "=== Starting run $RUN_NAME ==="
echo "Tiles dir: $TILES_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

python -u train_raft.py \
    --tiles-dir "$TILES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers 8 \
    "${EXTRA_ARGS[@]}"

echo "=== Run $RUN_NAME finished ==="
