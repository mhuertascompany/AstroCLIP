#!/bin/bash
#SBATCH --job-name=pop_res_flow
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
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/pop_res_flow_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/pop_res_flow_%j.err

set -euo pipefail

# Stage 1:
#   WANDB_MODE=online WANDB_PROJECT=euclid-population-residual \
#     sbatch euclid/rusty/slurm_population_residual_flow.sh population smoke
#
# Stage 2 (only after selecting the population checkpoint):
#   POPULATION_CHECKPOINT=/path/to/population-epoch=...ckpt \
#   WANDB_MODE=online WANDB_PROJECT=euclid-population-residual \
#     sbatch euclid/rusty/slurm_population_residual_flow.sh residual smoke

PHASE=${1:-population}
MODE=${2:-smoke}
RUN_TAG=${3:-}
if [[ "${PHASE}" != population && "${PHASE}" != residual ]]; then
    echo "PHASE must be population or residual" >&2
    exit 2
fi
if [[ "${MODE}" != smoke && "${MODE}" != pilot && "${MODE}" != full ]]; then
    echo "MODE must be smoke, pilot, or full" >&2
    exit 2
fi
if [[ "${PHASE}" == residual && -z "${POPULATION_CHECKPOINT:-}" ]]; then
    echo "Set POPULATION_CHECKPOINT to the selected stage-1 checkpoint" >&2
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
OUTPUT=${BASE}/runs/population_residual_flow_${PHASE}_${MODE}${RUN_TAG:+_${RUN_TAG}}
PREVIEW_EVERY=${PREVIEW_EVERY:-2}
EMBEDDING_EVERY=${EMBEDDING_EVERY:-2}
BATCH_SIZE=${BATCH_SIZE:-64}

cd "${REPO}"
[[ -f "${DATASET}" ]] || { echo "Missing ${DATASET}" >&2; exit 2; }
[[ -f "${PACKED}/vis_images.npy" ]] || { echo "Missing packed VIS data" >&2; exit 2; }

EXTRA=()
if [[ "${MODE}" == smoke ]]; then
    EXTRA=(--smoke --max-pairs 2048 --epochs 2 --batch-size 16)
elif [[ "${MODE}" == pilot ]]; then
    EXTRA=(--max-pairs 20000 --epochs 20 --patience 8 --batch-size "${BATCH_SIZE}")
else
    EXTRA=(--epochs 100 --patience 15 --batch-size "${BATCH_SIZE}")
fi
if [[ "${PHASE}" == residual ]]; then
    EXTRA+=(--population-checkpoint "${POPULATION_CHECKPOINT}")
fi
WANDB_ARGS=(--wandb-mode "${WANDB_MODE:-disabled}")
if [[ "${WANDB_MODE:-disabled}" != disabled ]]; then
    [[ -n "${WANDB_PROJECT:-}" ]] || { echo "WANDB_PROJECT is required" >&2; exit 2; }
    WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
    [[ -z "${WANDB_ENTITY:-}" ]] || WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -f "${OUTPUT}/checkpoints/last.ckpt" ]]; then
    EXTRA+=(--resume "${OUTPUT}/checkpoints/last.ckpt")
elif [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Nonempty output without last.ckpt: ${OUTPUT}" >&2
    exit 2
fi

echo "Repository commit: $(git rev-parse HEAD)"
echo "Phase: ${PHASE}"
echo "Mode: ${MODE}"
echo "Output: ${OUTPUT}"
[[ "${PHASE}" != residual ]] || echo "Population checkpoint: ${POPULATION_CHECKPOINT}"

python -u -m euclid.train_population_residual_flow \
    --phase "${PHASE}" \
    --dataset "${DATASET}" \
    --packed-vis "${PACKED}" \
    --output "${OUTPUT}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --embedding-dim 64 \
    --image-encoder-base 16 --image-flow-base 16 \
    --sfh-d-model 96 --sfh-heads 4 --sfh-layers 3 \
    --alignment-weight 0.1 --variance-weight 1.0 \
    --condition-dropout 0.15 \
    --control-batches 8 \
    --preview-every "${PREVIEW_EVERY}" --preview-samples 6 --preview-steps 32 \
    --embedding-every "${EMBEDDING_EVERY}" --embedding-samples 512 \
    --max-vis-mag 22 \
    --exclude-edge-on-axis-ratio-below 0.5 \
    --exclude-edge-on-probability-above 0.8 \
    --accelerator gpu "${WANDB_ARGS[@]}" "${EXTRA[@]}"
