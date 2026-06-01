#!/usr/bin/env bash
# setup_hpc.sh — one-time HPC environment setup for RAFT training.
#
# Run this from inside the unzipped hpc_package/ directory on the HPC submit
# node (NOT inside an sbatch job — sbatch jobs may not have internet).
#
# Usage:
#   bash setup_hpc.sh

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/raft-env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== Setting up RAFT training environment at $ENV_DIR ==="
echo "Python: $($PYTHON_BIN --version)"

# Create venv if missing.
if [[ ! -d "$ENV_DIR" ]]; then
    echo "Creating venv at $ENV_DIR ..."
    "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

# Activate.
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

pip install --upgrade pip wheel

# PyTorch: install latest stable that matches the system CUDA.
# pip will auto-pick the right wheel from pytorch.org if we point it there.
# Most OSU HPC nodes have CUDA 12.x; the cu121 wheel is broadly compatible.
echo "=== Installing PyTorch (cu121 wheels) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Other requirements.
echo "=== Installing remaining packages ==="
pip install -r requirements.txt

# Smoke test.
echo "=== Smoke test ==="
python - << 'PYEOF'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"cuda version: {torch.version.cuda}")
PYEOF

echo ""
echo "=== Setup complete. Activate with: source $ENV_DIR/bin/activate ==="
