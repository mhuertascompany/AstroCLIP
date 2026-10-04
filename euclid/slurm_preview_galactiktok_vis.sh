#!/bin/bash
#SBATCH --job-name=preview_galactiktok
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/preview_galactiktok_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/preview_galactiktok_%j.err
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Valid MAE preview for an existing GalaxyTikTok tokenizer. Defaults to the
# original masked model and its exact, unsmoothed reconstruction target.
#
# Usage:
#   sbatch $0 [tokenizer] [target_sigma] [dataset] [fits_root] \
#       [image_stats.json] [output.png] [galactiktok_root]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TOKENIZER=${1:-${BASE_DIR}/pretraining_galactiktok_vis/tokenizer}
TARGET_SIGMA=${2:-0}
DATASET=${3:-${BASE_DIR}/sfh_clip_150k.h5}
FITS_ROOT=${4:-${BASE_DIR}/cutouts_run}
IMAGE_STATS=${5:-${BASE_DIR}/pretraining_galactiktok_vis/image_stats.json}
OUTPUT=${6:-$(dirname "${TOKENIZER}")/reconstruction_examples_mae.png}
GALACTIKTOK_ROOT=${7:-/n03data/huertas/python/galactiktok/galactiktok}

[[ -f "${TOKENIZER}/config.json" ]] || { echo "Missing tokenizer: ${TOKENIZER}" >&2; exit 2; }
[[ -f "${DATASET}" ]] || { echo "Missing dataset: ${DATASET}" >&2; exit 2; }
[[ -d "${FITS_ROOT}/cutouts/VIS" ]] || { echo "Missing VIS FITS: ${FITS_ROOT}/cutouts/VIS" >&2; exit 2; }
[[ -f "${IMAGE_STATS}" ]] || { echo "Missing image statistics: ${IMAGE_STATS}" >&2; exit 2; }
[[ -d "${GALACTIKTOK_ROOT}/src/galactiktok" ]] || { echo "Missing GalaxyTikTok checkout: ${GALACTIKTOK_ROOT}" >&2; exit 2; }

export PYTHONPATH=${GALACTIKTOK_ROOT}/src:${PYTHONPATH:-}
echo "GalaxyTikTok commit: $(git -C "${GALACTIKTOK_ROOT}" rev-parse HEAD)"
echo "Tokenizer: ${TOKENIZER}"
echo "Target Gaussian sigma: ${TARGET_SIGMA}"

python -u -m euclid.preview_galactiktok_vis \
    --tokenizer "${TOKENIZER}" \
    --dataset "${DATASET}" \
    --fits-root "${FITS_ROOT}" \
    --image-stats "${IMAGE_STATS}" \
    --target-gaussian-sigma "${TARGET_SIGMA}" \
    --output "${OUTPUT}"
