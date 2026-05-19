#!/usr/bin/env bash
#SBATCH --job-name=cxr-train
#SBATCH --partition=monstrao
#SBATCH --gres=gpu:v100-32gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#
# Multi-GPU CXR pneumonia training on `monstrao`.
#
# Two phases inside the job:
#   1. build_cache.py — single-GPU pass that pre-populates
#      outputs/preproc_cache/ with lung-segmented crops. Idempotent.
#   2. torchrun ... train.py — DDP across all 4 GPUs.
#
# Pre-reqs (run once on the LOGIN node, since compute is offline):
#     ./setup_home.sh
#     source venv/bin/activate
#     export TORCH_HOME="$PWD/.cache/torch"
#     python -c "import torchxrayvision as xrv; xrv.baseline_models.chestx_det.PSPNet()"
#
# Submit:  sbatch train.sbatch

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load cuda/12.8.1
source venv/bin/activate

mkdir -p outputs

echo "=== job info ==="
echo "  node:    $(hostname)"
echo "  jobid:   ${SLURM_JOB_ID:-n/a}"
echo "  date:    $(date)"
nvidia-smi || true

# Cache torchxrayvision weights inside the project so jobs can reuse them.
export TORCH_HOME="$SLURM_SUBMIT_DIR/.cache/torch"
mkdir -p "$TORCH_HOME"

# NCCL: be loud if something's off; tune InfiniBand off if the cluster has issues.
export NCCL_DEBUG=WARN

python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda avail: {torch.cuda.is_available()} | n_gpu: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
PY

# --- 1. Pre-populate segmentation cache (single GPU, idempotent) ----------
echo
echo "=== building segmentation cache ==="
t0=$SECONDS
python backend/build_cache.py
echo "cache build elapsed: $((SECONDS - t0))s"

# --- 2. DDP training across 4 GPUs ----------------------------------------
# --num-workers 8  per rank × 4 ranks = 32 workers, matches --cpus-per-task.
# --batch-size 32  per GPU × 4 = effective batch 128.
echo
echo "=== DDP training (4× V100 32GB) ==="
torchrun --nproc_per_node=4 --standalone backend/train.py \
  --epochs-phase1 5 \
  --epochs-phase2 8 \
  --batch-size 32 \
  --img-size 640 \
  --num-workers 8

echo "=== done ==="
date
