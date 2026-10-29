#!/bin/bash
#SBATCH --job-name=shared_private
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
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/shared_private_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/shared_private_%j.err

set -euo pipefail

# First-stage shared/private latent experiment.
#
# Usage:
#   sbatch euclid/rusty/slurm_shared_private_latent.sh test [sfh_ae.ckpt]
#   sbatch euclid/rusty/slurm_shared_private_latent.sh full [sfh_ae.ckpt]

MODE=${1:-test}
if [[ "${MODE}" != test && "${MODE}" != full ]]; then
    echo "MODE must be test or full, received: ${MODE}" >&2
    exit 2
fi

module load python
source "${RUSTY_VENV:-${HOME}/.sfh_euclid}/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
DATASET=${ARCHIVE}/sfh_clip_150k.h5
STAMP_ROOT=${ARCHIVE}/zoobot_stamps_rmax
AE_CKPT=${2:-${BASE}/models/euclid_sfh_autoencoder_v1-epoch=001-val_loss=0.02222.ckpt}
OUTPUT=${BASE}/runs/shared_private_latent_${MODE}

export HF_HOME=${BASE}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TORCH_HOME=${BASE}/cache/torch

cd "${REPO}"
for path in "${DATASET}" "${AE_CKPT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMP_ROOT}/VIS" ]] || {
    echo "Missing stamps: ${STAMP_ROOT}/VIS" >&2
    exit 2
}

if [[ -f "${OUTPUT}/COMPLETED" ]]; then
    echo "Already complete: ${OUTPUT}" >&2
    exit 2
fi
RESUME_ARGS=()
if [[ -f "${OUTPUT}/checkpoints/last.ckpt" ]]; then
    RESUME_ARGS=(--resume-from "${OUTPUT}/checkpoints/last.ckpt")
elif [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Nonempty output without a last checkpoint: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT}" "${HF_HUB_CACHE}" "${TORCH_HOME}"

MODE_ARGS=()
if [[ "${MODE}" == test ]]; then
    MODE_ARGS=(--max-pairs 4096 --max-epochs 5 --patience 5)
else
    MODE_ARGS=(--max-epochs 80 --patience 15)
fi

echo "Repository commit: $(git rev-parse HEAD)"
echo "Mode: ${MODE}"
echo "Output: ${OUTPUT}"
python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" \
    --stamp-root "${STAMP_ROOT}" --band VIS \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${OUTPUT}" \
    --run-name "euclid_shared_private_latent_${MODE}" \
    --sfh-pretrained-checkpoint "${AE_CKPT}" \
    --freeze-sfh-encoder --sfh-reconstruction-weight 0 \
    --alignment-objective shared_private \
    --shared-dim 32 --private-dim 64 \
    --shared-private-hidden-dim 256 \
    --shared-pair-weight 1 \
    --shared-variance-weight 1 \
    --shared-covariance-weight 0.01 \
    --shared-private-reconstruction-weight 1 \
    --shared-private-orthogonality-weight 0.01 \
    --shared-variance-target 1 \
    --image-projection mlp \
    --image-projection-hidden-dim 1024 \
    --image-projection-hidden-layers 2 \
    --sfh-projection mlp \
    --sfh-projection-hidden-dim 1024 \
    --sfh-projection-hidden-layers 2 \
    --no-sample-posterior \
    --exclude-edge-on-axis-ratio-below 0.5 \
    --exclude-edge-on-probability-above 0.8 \
    --batch-size 128 --queue-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK}" --image-size 224 \
    --embed-dim 256 --sfh-encoder transformer --sfh-d-model 128 \
    --sfh-n-heads 4 --sfh-n-layers 4 \
    --warmup-epochs 3 --lr 1e-4 --weight-decay 0.01 \
    --unfreeze-blocks 0 \
    --accelerator gpu --devices 1 --precision 16-mixed \
    "${MODE_ARGS[@]}" "${RESUME_ARGS[@]}"

BEST_CKPT=$(python - "${OUTPUT}/checkpoints" <<'PY'
import re
import sys
from pathlib import Path

scored = []
for path in Path(sys.argv[1]).glob('*.ckpt'):
    match = re.search(r'val_loss=([0-9.]+)', path.name)
    if match:
        scored.append((float(match.group(1).rstrip('.')), path))
if not scored:
    raise SystemExit('No validation-scored checkpoint found')
print(min(scored)[1])
PY
)

EVALUATION=${OUTPUT}/evaluation_best
mkdir -p "${EVALUATION}"
EVAL_ARGS=(--n-permutations 200 --permutation-size 5000 --shape-subset 2000)
if [[ "${MODE}" == test ]]; then
    EVAL_ARGS=(--n-permutations 20 --permutation-size 1000 --shape-subset 1000)
fi
python -u -m euclid.evaluate_zoobot_clip \
    --checkpoint "${BEST_CKPT}" --dataset "${DATASET}" \
    --skip-posterior --stamp-root "${STAMP_ROOT}" \
    --split "${OUTPUT}/pair_split.npz" --output-dir "${EVALUATION}" \
    --band VIS --batch-size 128 --num-workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 512 --device cuda "${EVAL_ARGS[@]}"

touch "${OUTPUT}/COMPLETED"
echo "Best checkpoint: ${BEST_CKPT}"
echo "Evaluation: ${EVALUATION}"
