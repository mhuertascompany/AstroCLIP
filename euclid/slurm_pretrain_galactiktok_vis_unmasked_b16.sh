#!/bin/bash
#SBATCH --job-name=galactiktok_b16
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/pretrain_galactiktok_b16_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/pretrain_galactiktok_b16_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Full-image reconstruction pretraining on the bright Euclid VIS FITS stamps.
# Compared with the original masked run, this experiment uses every image
# patch and doubles the per-patch bottleneck from 8 to 16 dimensions.
#
# Usage:
#   sbatch $0 [dataset] [fits_root] [output_dir] [galactiktok_root] \
#       [image_stats.json] [resume.ckpt]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
DATASET=${1:-${BASE_DIR}/sfh_clip_150k.h5}
FITS_ROOT=${2:-${BASE_DIR}/cutouts_run}
OUTPUT_DIR=${3:-${BASE_DIR}/pretraining_galactiktok_vis_unmasked_b16}
GALACTIKTOK_ROOT=${4:-/n03data/huertas/python/galactiktok/galactiktok}
IMAGE_STATS=${5:-${BASE_DIR}/pretraining_galactiktok_vis/image_stats.json}
RESUME_FROM=${6:-}

[[ -f "${DATASET}" ]] || { echo "Missing dataset: ${DATASET}" >&2; exit 2; }
[[ -d "${FITS_ROOT}/cutouts/VIS" ]] || { echo "Missing VIS FITS: ${FITS_ROOT}/cutouts/VIS" >&2; exit 2; }
[[ -d "${GALACTIKTOK_ROOT}/src/galactiktok" ]] || { echo "Missing GalaxyTikTok checkout: ${GALACTIKTOK_ROOT}" >&2; exit 2; }
[[ -f "${IMAGE_STATS}" ]] || { echo "Missing image statistics: ${IMAGE_STATS}" >&2; exit 2; }
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ -z "${RESUME_FROM}" ]]; then
    echo "Output is not empty; choose another path or pass resume.ckpt: ${OUTPUT_DIR}" >&2
    exit 2
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    [[ -f "${RESUME_FROM}" ]] || { echo "Missing resume checkpoint: ${RESUME_FROM}" >&2; exit 2; }
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

export PYTHONPATH=${GALACTIKTOK_ROOT}/src:${PYTHONPATH:-}
mkdir -p "${OUTPUT_DIR}"

python -u -m euclid.pretrain_galactiktok_vis \
    --dataset "${DATASET}" \
    --fits-root "${FITS_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_vis_galactiktok_unmasked_b16 \
    --image-stats "${IMAGE_STATS}" \
    --image-size 96 \
    --patch-size 8 \
    --embed-dim 512 \
    --num-heads 8 \
    --num-encoder-blocks 6 \
    --num-decoder-blocks 6 \
    --bottleneck-dim 16 \
    --mask-fraction 0.0 \
    --batch-size 64 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --max-epochs 100 \
    --warmup-epochs 5 \
    --patience 12 \
    --lr 2e-4 \
    --weight-decay 0.05 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    "${RESUME_ARGS[@]}"
