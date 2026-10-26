#!/bin/bash
#SBATCH --job-name=sfh_ae_full
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/sfh_ae_full_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/sfh_ae_full_%j.err

set -euo pipefail

module load python

VENV="${RUSTY_VENV:-${HOME}/.sfh_euclid}"
source "${VENV}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
DATASET="${BASE}/runs/iaaft_random_window/sfh_iaaft_window_random_window_past90_150k.h5"
SPLIT="${BASE}/archives/edfn_vislt22p0_150000/reference_splits/pair_split_ae_no_edgeon.npz"
OUTPUT="${BASE}/runs/iaaft_random_window/sfh_autoencoder_full"
RESUME_FROM="${1:-}"

cd "${REPO}"

REQUIRED_FIX=6c3d142
git merge-base --is-ancestor "${REQUIRED_FIX}" HEAD || {
    echo "Checkout does not contain required fix ${REQUIRED_FIX}" >&2
    exit 2
}

[[ -f "${DATASET}" ]] || {
    echo "Missing surrogate dataset: ${DATASET}" >&2
    exit 2
}
[[ -f "${SPLIT}" ]] || {
    echo "Missing reference split: ${SPLIT}" >&2
    exit 2
}

RESUME_ARGS=()

if [[ -n "${RESUME_FROM}" ]]; then
    [[ -f "${RESUME_FROM}" ]] || {
        echo "Missing resume checkpoint: ${RESUME_FROM}" >&2
        exit 2
    }
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
elif [[ -d "${OUTPUT}" ]] &&
     [[ -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output is not empty. Pass a checkpoint to resume:" >&2
    echo "  sbatch $0 ${OUTPUT}/checkpoints/last.ckpt" >&2
    exit 2
fi

mkdir -p "${OUTPUT}"

python - <<'PY'
import lightning
import torch

print("Torch:", torch.__version__)
print("Torch location:", torch.__file__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Lightning:", lightning.__version__)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Repository branch: $(git branch --show-current)"
echo "Repository commit: $(git rev-parse HEAD)"
echo "Dataset: ${DATASET}"
echo "Split: ${SPLIT}"
echo "Output: ${OUTPUT}"
echo "Resume checkpoint: ${RESUME_FROM:-none}"

python -u -m euclid.pretrain_sfh_autoencoder \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output-dir "${OUTPUT}" \
    --run-name euclid_sfh_autoencoder_iaaft_random_window_150k \
    --input-mode median \
    --batch-size 128 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --embed-dim 256 \
    --d-model 128 \
    --n-heads 4 \
    --encoder-layers 4 \
    --decoder-layers 2 \
    --mask-fraction 0.35 \
    --w1-weight 0.5 \
    --max-epochs 50 \
    --warmup-epochs 3 \
    --patience 10 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    "${RESUME_ARGS[@]}"

python - "${OUTPUT}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
scored = []

for path in (root / "checkpoints").glob("*.ckpt"):
    match = re.search(r"val_loss=([0-9.]+)", path.name)
    if match:
        scored.append((float(match.group(1).rstrip(".")), path))

if not scored:
    raise SystemExit("No validation-scored checkpoint found")

score, checkpoint = min(scored)

print("Best checkpoint:", checkpoint)
print("Best validation loss:", score)
print("Full SFH-autoencoder training: OK")
PY
