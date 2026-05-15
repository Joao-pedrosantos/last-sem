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
  # RTX 50-series (Blackwell, sm_120) needs cu128 wheels (PyTorch 2.7+).
  # Older PyTorch cu124 builds only ship kernels up to sm_90 and will fail.
  if echo "$GPU_NAME" | grep -qiE "RTX 50[0-9]{2}|Blackwell"; then
    WHEEL="cu128"
  else
    WHEEL="cu124"
  fi
  echo "  Using CUDA wheel: $WHEEL"
else
  WHEEL="cpu"
  echo "  No NVIDIA GPU detected — installing CPU PyTorch (training will be very slow)."
fi

# --- create venv ---
if [[ ! -d "$VENV_DIR" ]]; then
  echo "  Creating venv at $VENV_DIR ..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$VENV_DIR"
  else
    python -m venv "$VENV_DIR"
  fi
fi
# On Windows (Git Bash/MSYS) the venv layout is Scripts/, on Linux/Mac it's bin/
if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  ACTIVATE="$VENV_DIR/Scripts/activate"
else
  ACTIVATE="$VENV_DIR/bin/activate"
fi
# shellcheck disable=SC1090
source "$ACTIVATE"

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
