#!/usr/bin/env bash
#SBATCH --job-name=raft-systematic
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --account=eecs
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Usage:
#   cd ~/hpc-share/Capstone/Raft/hpc_package
#   sbatch train_systematic.sbatch full_C_systematic_200
#
# This is a long-running variant of train.sbatch (--time=20h, --partition=dgxh).
# Default points to the systematic-crop dataset (13,872 samples, ~75% more than B).
# Override with TILES_DIR env var if needed.

set -euo pipefail

RUN_NAME="${1:-full_C_systematic_200}"
shift || true
EXTRA_ARGS=("$@")

# === Paths (matches the layout used for full_A_200 / full_B_200) ===
PACKAGE_DIR="$HOME/hpc-share/Capstone/Raft/hpc_package"
TILES_DIR="${TILES_DIR:-$HOME/hpc-share/Capstone/Raft/training_tiles_128_systematic}"
ENV_DIR="${ENV_DIR:-$HOME/hpc-share/raft-env}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/hpc-share/Capstone/Raft/runs/$RUN_NAME}"

mkdir -p logs "$OUTPUT_DIR"

# === Activate env ===
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"
cd "$PACKAGE_DIR"

# === Sanity: GPU visible ===
which python
python --version
nvidia-smi || echo "WARNING: nvidia-smi not available"
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# === Train ===
echo "=== Starting $RUN_NAME (systematic crop variant) ==="
echo "Tiles: $TILES_DIR"
echo "Output: $OUTPUT_DIR"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo ""

python -u train_raft.py \
    --tiles-dir "$TILES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers 8 \
    --epochs 200 \
    --batch-size 16 \
    "${EXTRA_ARGS[@]}"

echo "=== Run $RUN_NAME finished ==="
