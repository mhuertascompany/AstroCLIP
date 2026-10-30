#!/bin/bash
#SBATCH --job-name=gc_sfh_vis
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --requeue
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/gc_sfh_vis_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/gc_sfh_vis_%j.err

set -euo pipefail

# Paper-faithful VIS decoder pilot based on:
# https://github.com/pablomerc/galaxy-counter @ 19679fa1c40f86cc4e2ce485660076e8add486d5
# Usage: sbatch euclid/rusty/slurm_galaxy_counter_sfh_vis.sh [smoke|pilot|overfit] [tag]

MODE=${1:-smoke}
TAG=${2:-}
[[ "${MODE}" == smoke || "${MODE}" == pilot || "${MODE}" == overfit ]] || {
    echo "MODE must be smoke, pilot, or overfit" >&2
    exit 2
}

module load python cuda/12.5.1 cudnn/8.9.7.29-12
source "${RUSTY_VENV:-${HOME}/.sfh_euclid}/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-online}

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
DATASET=${ARCHIVE}/sfh_clip_150k.h5
PACKED=${ARCHIVE}/packed_vis_64
OUTPUT=${BASE}/runs/galaxy_counter_sfh_vis_${MODE}${TAG:+_${TAG}}

cd "${REPO}"
python - <<'PY'
import diffusers, geomloss, timm
print("galaxy-counter dependencies: OK")
print("diffusers:", diffusers.__version__)
PY

if [[ "${MODE}" == smoke ]]; then
    EXTRA=(--max-pairs 4096 --epochs 2 --batch-size 8 --pool-size 24 --preview-steps 50)
    PREVIEW_EVERY=1
elif [[ "${MODE}" == pilot ]]; then
    EXTRA=(--max-pairs 20000 --epochs 15 --batch-size 16 --pool-size 64 --preview-steps 250)
    PREVIEW_EVERY=1
else
    # Deliberately memorize a tiny fixed dataset to verify that the flow can
    # causally use common and private conditioning before full-scale training.
    EXTRA=(
        --max-pairs 64 --val-fraction 0.125 --epochs 500 --batch-size 8
        --pool-size 24 --preview-steps 100 --fixed-training-donors
        --preview-training --validate-every 25 --donor-reference-all
    )
    PREVIEW_EVERY=25
fi

WANDB_ARGS=(--wandb-mode "${WANDB_MODE}")
if [[ "${WANDB_MODE}" != disabled ]]; then
    WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT:-euclid-counterfactual-flow}")
    [[ -z "${WANDB_ENTITY:-}" ]] || WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

echo "AstroCLIP commit: $(git rev-parse HEAD)"
echo "galaxy-counter upstream: 19679fa1c40f86cc4e2ce485660076e8add486d5"
echo "Output: ${OUTPUT}"
python -u -m euclid.train_galaxy_counter_sfh_vis \
    --dataset "${DATASET}" --packed-vis "${PACKED}" --output "${OUTPUT}" \
    --workers "${SLURM_CPUS_PER_TASK}" --model-channels 128 \
    --geometric-weight "${GEOMETRIC_WEIGHT:-0.0}" \
    --donor-matching "${DONOR_MATCHING:-mass_window}" \
    --mass-window "${MASS_WINDOW:-0.5}" \
    --preview-every "${PREVIEW_EVERY}" --preview-samples 6 --control-batches 8 \
    --max-vis-mag 22 --exclude-edge-on-axis-ratio-below 0.5 \
    --exclude-edge-on-probability-above 0.8 \
    "${WANDB_ARGS[@]}" "${EXTRA[@]}"
