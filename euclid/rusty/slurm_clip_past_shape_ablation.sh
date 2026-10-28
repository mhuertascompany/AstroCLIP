#!/bin/bash
#SBATCH --job-name=clip_keep_recent
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
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/clip_keep_recent_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/clip_keep_recent_%j.err

set -euo pipefail
module load python
source "${RUSTY_VENV:-${HOME}/.sfh_euclid}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
DATASET=${ARCHIVE}/sfh_clip_150k.h5
STAMP_ROOT=${ARCHIVE}/zoobot_stamps_rmax
OVERRIDE=${BASE}/runs/past_shape_ablation/sfh_recent_preserved_independent_past_150k.h5
AE_CKPT=${1:-${BASE}/models/euclid_sfh_autoencoder_v1-epoch=001-val_loss=0.02222.ckpt}
RESUME_FROM="${2:-}"
OUTPUT=${BASE}/runs/past_shape_ablation/clip_ae_adjacency_no_edgeon

export HF_HOME=${BASE}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TORCH_HOME=${BASE}/cache/torch
cd "${REPO}"
for path in "${DATASET}" "${OVERRIDE}" "${AE_CKPT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMP_ROOT}/VIS" ]] || { echo "Missing stamps: ${STAMP_ROOT}/VIS" >&2; exit 2; }

RESUME_ARGS=()
if [[ -f "${OUTPUT}/COMPLETED" ]]; then
    echo "Already complete: ${OUTPUT}" >&2
    exit 2
elif [[ -n "${RESUME_FROM}" ]]; then
    [[ -f "${RESUME_FROM}" ]] || { echo "Missing resume checkpoint: ${RESUME_FROM}" >&2; exit 2; }
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
elif [[ -f "${OUTPUT}/checkpoints/last.ckpt" ]]; then
    RESUME_ARGS=(--resume-from "${OUTPUT}/checkpoints/last.ckpt")
elif [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Nonempty output without checkpoint: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT}" "${HF_HUB_CACHE}" "${TORCH_HOME}"

echo "Repository commit: $(git rev-parse HEAD)"
echo "Recent-preserved, independent-past SFHs: ${OVERRIDE}"
python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" --sfh-override "${OVERRIDE}" \
    --stamp-root "${STAMP_ROOT}" --band VIS \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${OUTPUT}" --run-name euclid_clip_past_shape_ablation_150k \
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

BEST_CKPT=$(python - "${OUTPUT}/checkpoints" <<'PY'
import re, sys
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
EVALUATION=${OUTPUT}/evaluation_ablation_pairs
mkdir -p "${EVALUATION}"
python -u -m euclid.evaluate_zoobot_clip \
    --checkpoint "${BEST_CKPT}" --dataset "${DATASET}" \
    --sfh-override "${OVERRIDE}" --skip-posterior \
    --stamp-root "${STAMP_ROOT}" --split "${OUTPUT}/pair_split.npz" \
    --output-dir "${EVALUATION}" --band VIS \
    --batch-size 128 --num-workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 512 --n-permutations 200 --permutation-size 5000 \
    --shape-subset 2000 --device cuda
touch "${OUTPUT}/COMPLETED"
