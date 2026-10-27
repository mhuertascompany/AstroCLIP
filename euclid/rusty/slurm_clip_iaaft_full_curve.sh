#!/bin/bash
#SBATCH --job-name=clip_iaaft_full_curve
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --requeue
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/clip_iaaft_full_curve_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/clip_iaaft_full_curve_%j.err

set -euo pipefail
module load python
VENV="${RUSTY_VENV:-${HOME}/.sfh_euclid}"
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
DATASET=${ARCHIVE}/sfh_clip_150k.h5
SFH_OVERRIDE=${BASE}/runs/iaaft_full_curve/sfh_iaaft_full_curve_150k.h5
STAMP_ROOT=${ARCHIVE}/zoobot_stamps_rmax
AE_ROOT=${BASE}/runs/iaaft_full_curve/sfh_autoencoder_full
OUTPUT=${BASE}/runs/iaaft_full_curve/clip_ae_adjacency_no_edgeon_full
RESUME_FROM="${1:-}"
export HF_HOME=${BASE}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TORCH_HOME=${BASE}/cache/torch
cd "${REPO}"

AE_CKPT=$(python - "${AE_ROOT}/checkpoints" <<'PY'
import re, sys
from pathlib import Path
scored=[]
for path in Path(sys.argv[1]).glob("*.ckpt"):
    match=re.search(r"val_loss=([0-9.]+)", path.name)
    if match:
        scored.append((float(match.group(1).rstrip(".")), path))
if not scored:
    raise SystemExit("No validation-scored SFH AE checkpoint found")
print(min(scored)[1])
PY
)
for path in "${DATASET}" "${SFH_OVERRIDE}" "${AE_CKPT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMP_ROOT}/VIS" ]] || { echo "Missing stamps" >&2; exit 2; }

RESUME_ARGS=()
if [[ -f "${OUTPUT}/COMPLETED" ]]; then
    echo "Already complete: ${OUTPUT}" >&2; exit 2
elif [[ -n "${RESUME_FROM}" ]]; then
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
elif [[ -f "${OUTPUT}/checkpoints/last.ckpt" ]]; then
    RESUME_ARGS=(--resume-from "${OUTPUT}/checkpoints/last.ckpt")
elif [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Nonempty output without checkpoint: ${OUTPUT}" >&2; exit 2
fi
mkdir -p "${OUTPUT}" "${HF_HUB_CACHE}" "${TORCH_HOME}"

echo "SFH override: ${SFH_OVERRIDE}"
echo "SFH autoencoder: ${AE_CKPT}"
python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" --sfh-override "${SFH_OVERRIDE}" \
    --stamp-root "${STAMP_ROOT}" --band VIS \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${OUTPUT}" --run-name euclid_clip_iaaft_full_curve_150k \
    --sfh-pretrained-checkpoint "${AE_CKPT}" \
    --freeze-sfh-encoder --sfh-reconstruction-weight 0 \
    --image-projection mlp --image-projection-hidden-dim 1024 --image-projection-hidden-layers 2 \
    --sfh-projection mlp --sfh-projection-hidden-dim 1024 --sfh-projection-hidden-layers 2 \
    --no-sample-posterior \
    --exclude-edge-on-axis-ratio-below 0.5 --exclude-edge-on-probability-above 0.8 \
    --batch-size 128 --soft-positive-weight 0 \
    --ae-adjacency-weight 100 --ae-adjacency-warmup-epochs 3 \
    --ae-adjacency-temperature 0.07 --queue-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK}" --image-size 224 \
    --embed-dim 256 --sfh-encoder transformer --sfh-d-model 128 \
    --sfh-n-heads 4 --sfh-n-layers 4 \
    --max-epochs 60 --warmup-epochs 3 --patience 12 \
    --lr 1e-4 --weight-decay 0.01 --unfreeze-blocks 0 \
    --accelerator gpu --devices 1 --precision 16-mixed \
    "${RESUME_ARGS[@]}"
touch "${OUTPUT}/COMPLETED"
