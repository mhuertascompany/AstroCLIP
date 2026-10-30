#!/bin/bash
#SBATCH --job-name=single_neighbor
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
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/single_neighbor_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/single_neighbor_%j.err

set -euo pipefail

# Usage:
#   sbatch euclid/rusty/slurm_counterfactual_single_neighbor.sh smoke [lambda_geom]
#   WANDB_MODE=online WANDB_PROJECT=euclid-counterfactual-flow \
#     sbatch euclid/rusty/slurm_counterfactual_single_neighbor.sh full 0.1

MODE=${1:-smoke}
GEOMETRIC_WEIGHT=${2:-0.1}
RUN_TAG=${3:-}
if [[ "${MODE}" != smoke && "${MODE}" != full ]]; then
    echo "MODE must be smoke or full" >&2
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
WEIGHT_TAG=${GEOMETRIC_WEIGHT/./p}
OUTPUT=${BASE}/runs/counterfactual_single_neighbor_geom${WEIGHT_TAG}_${MODE}${RUN_TAG:+_${RUN_TAG}}

cd "${REPO}"
[[ -f "${DATASET}" ]] || { echo "Missing ${DATASET}" >&2; exit 2; }
[[ -f "${PACKED}/vis_images.npy" ]] || { echo "Run slurm_pack_vis_memmap.sh first" >&2; exit 2; }
python -c 'import geomloss' || {
    echo "Missing geomloss; activate the Rusty venv and run: python -m pip install geomloss" >&2
    exit 2
}

EXTRA=()
if [[ "${MODE}" == smoke ]]; then
    EXTRA=(--smoke --max-pairs 4096 --epochs 2 --batch-size 16 --pool-size 24)
else
    EXTRA=(--epochs 100 --batch-size 32 --pool-size 64)
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
echo "Mode: ${MODE}"
echo "Geometric weight: ${GEOMETRIC_WEIGHT}"
echo "Output: ${OUTPUT}"
python -u -m euclid.train_counterfactual_single_neighbor \
    --dataset "${DATASET}" \
    --packed-vis "${PACKED}" \
    --output "${OUTPUT}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --common-dim 64 --private-dim 64 --condition-dim 128 \
    --image-encoder-base 16 --image-flow-base 16 \
    --sfh-d-model 96 --sfh-layers 3 \
    --geometric-weight "${GEOMETRIC_WEIGHT}" --sinkhorn-blur 0.01 \
    --control-batches 8 \
    --preview-every 5 --preview-samples 6 --preview-steps 32 \
    --max-vis-mag 22 \
    --exclude-edge-on-axis-ratio-below 0.5 \
    --exclude-edge-on-probability-above 0.8 \
    --accelerator gpu "${WANDB_ARGS[@]}" "${EXTRA[@]}"
