#!/usr/bin/env bash
# One-shot environment setup. Creates a Python venv at ./venv, installs
# PyTorch with a CUDA wheel matching the host GPU, then the rest of the
# project dependencies. Targets a Blackwell GPU (RTX 50-series) by default.
#
# Usage:
#   ./setup_home.sh                # auto: cu124 if NVIDIA present, CPU otherwise
#   CUDA=cu126 ./setup_home.sh     # force a specific CUDA wheel index
#   CUDA=cpu   ./setup_home.sh     # CPU-only
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$HERE/venv"
REQ="$HERE/backend/requirements.txt"

echo ""
echo "  CXR Pneumonia Detection — environment setup"
echo "  -------------------------------------------"

# --- detect GPU ---
if [[ -n "${CUDA:-}" ]]; then
  WHEEL="$CUDA"
  echo "  Using CUDA wheel: $WHEEL (forced via CUDA env var)"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  # Read the driver-reported CUDA version, pick the matching wheel index.
  DRV_CUDA="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | awk -F. '{print $1}')"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "  Detected GPU: $GPU_NAME (driver major $DRV_CUDA)"
  # RTX 50-series / Blackwell needs CUDA 12.4+ binaries. cu124 is the safe default;
  # cu126 (PyTorch nightly path) is even better if you have a 555+ driver.
  WHEEL="cu124"
  echo "  Using CUDA wheel: $WHEEL"
else
  WHEEL="cpu"
  echo "  No NVIDIA GPU detected — installing CPU PyTorch (training will be very slow)."
fi

# --- create venv ---
if [[ ! -d "$VENV_DIR" ]]; then
  echo "  Creating venv at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -q -U pip wheel

# --- install PyTorch from the right index ---
echo "  Installing PyTorch from the $WHEEL wheel index ..."
if [[ "$WHEEL" == "cpu" ]]; then
  pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
else
  pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$WHEEL"
fi

# --- install the rest ---
echo "  Installing project dependencies from $REQ ..."
pip install -q -r "$REQ"

# --- verify ---
echo ""
echo "  Verification:"
python - <<'PY'
import torch
print(f"    torch:       {torch.__version__}")
print(f"    cuda avail:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    cuda device: {torch.cuda.get_device_name(0)}")
    print(f"    cuda cap:    {torch.cuda.get_device_capability(0)}")
import timm, albumentations, pydicom, cv2, sklearn  # noqa
print("    timm + albumentations + pydicom + cv2 + sklearn: OK")
PY

echo ""
echo "  Done. Next steps:"
echo "    1. Place Kaggle credentials at ~/.kaggle/kaggle.json (chmod 600)"
echo "    2. Open project.ipynb in Jupyter and run from the top —"
echo "       cell 0.4 will download the RSNA dataset (~3.4 GB)."
echo "    3. After training, start the backend: ./start.sh"
echo ""
