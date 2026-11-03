#!/bin/bash
#SBATCH --job-name=euclid_jepa
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --requeue
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/multimodal_jepa_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/multimodal_jepa_%j.err

set -euo pipefail

# Usage:
#   WANDB_MODE=online sbatch euclid/rusty/slurm_multimodal_jepa.sh smoke
#   WANDB_MODE=online sbatch euclid/rusty/slurm_multimodal_jepa.sh pilot v1
#   WANDB_MODE=online sbatch euclid/rusty/slurm_multimodal_jepa.sh full v1

MODE=${1:-smoke}
RUN_TAG=${2:-v1}
if [[ "${MODE}" != smoke && "${MODE}" != pilot && "${MODE}" != full ]]; then
    echo "MODE must be smoke, pilot, or full" >&2
    exit 2
fi

module load python cuda/12.5.1 cudnn/8.9.7.29-12
source "${RUSTY_VENV:-${HOME}/.sfh_euclid}/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
DATASET=${ARCHIVE}/sfh_clip_150k.h5
PACKED=${ARCHIVE}/packed_vis_64
OUTPUT=${BASE}/runs/multimodal_jepa_${MODE}_${RUN_TAG}
BATCH_SIZE=${BATCH_SIZE:-64}
PREVIEW_EVERY=${PREVIEW_EVERY:-2}

cd "${REPO}"
[[ -f "${DATASET}" ]] || { echo "Missing ${DATASET}" >&2; exit 2; }
[[ -f "${PACKED}/vis_images.npy" ]] || { echo "Missing packed VIS data" >&2; exit 2; }

EXTRA=()
if [[ "${MODE}" == smoke ]]; then
    EXTRA=(--smoke --max-pairs 2048 --epochs 2 --batch-size 16 --embedding-samples 128 --error-samples 16)
elif [[ "${MODE}" == pilot ]]; then
    EXTRA=(--max-pairs 20000 --epochs 30 --patience "${PATIENCE:-10}" --batch-size "${BATCH_SIZE}")
else
    EXTRA=(--epochs 100 --patience "${PATIENCE:-15}" --batch-size "${BATCH_SIZE}")
fi
WANDB_ARGS=(--wandb-mode "${WANDB_MODE:-disabled}")
if [[ "${NO_EARLY_STOPPING:-0}" == 1 ]]; then
    EXTRA+=(--no-early-stopping)
fi
if [[ "${WANDB_MODE:-disabled}" != disabled ]]; then
    WANDB_ARGS+=(
        --wandb-project "${WANDB_PROJECT:-euclid-counterfactual-flow}"
        --wandb-entity "${WANDB_ENTITY:-euclid-sfh-morphology}"
    )
fi
if [[ -f "${OUTPUT}/checkpoints/last.ckpt" ]]; then
    EXTRA+=(--resume "${OUTPUT}/checkpoints/last.ckpt")
elif [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Nonempty output without last.ckpt: ${OUTPUT}" >&2
    exit 2
fi

echo "Repository commit: $(git rev-parse HEAD)"
echo "Mode: ${MODE}"
echo "Output: ${OUTPUT}"

python -u -m euclid.train_multimodal_jepa \
    --dataset "${DATASET}" \
    --packed-vis "${PACKED}" \
    --output "${OUTPUT}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --image-patch 8 --sfh-patch 5 \
    --width 256 --heads 8 --layers 6 \
    --predictor-width 192 --predictor-heads 6 --predictor-layers 3 \
    --global-weight 0.0 --ema-start 0.996 --ema-end 0.9999 \
    --lr 2e-4 --weight-decay 0.05 --control-batches 8 \
    --preview-every "${PREVIEW_EVERY}" \
    --max-vis-mag 22 \
    --exclude-edge-on-axis-ratio-below 0.5 \
    --exclude-edge-on-probability-above 0.8 \
    --accelerator gpu "${WANDB_ARGS[@]}" "${EXTRA[@]}"
