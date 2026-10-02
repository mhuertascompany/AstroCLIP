#!/bin/bash
#SBATCH --job-name=bright_galactiktok_test
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/pretrain_galactiktok_test_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/pretrain_galactiktok_test_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Reconstruction-only smoke test on the exact bright VIS stamps.
# Usage: sbatch $0 [dataset] [fits_root] [output_dir] [galactiktok_root]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
DATASET=${1:-${BASE_DIR}/sfh_clip_150k.h5}
FITS_ROOT=${2:-${BASE_DIR}/cutouts_run}
OUTPUT_DIR=${3:-${BASE_DIR}/pretraining_galactiktok_vis_test}
GALACTIKTOK_ROOT=${4:-/n03data/huertas/python/galactiktok}

[[ -f "${DATASET}" ]] || { echo "Missing dataset: ${DATASET}" >&2; exit 2; }
[[ -d "${FITS_ROOT}/cutouts/VIS" ]] || { echo "Missing VIS FITS: ${FITS_ROOT}/cutouts/VIS" >&2; exit 2; }
[[ -d "${GALACTIKTOK_ROOT}/src/galactiktok" ]] || { echo "Missing GalaxyTikTok checkout: ${GALACTIKTOK_ROOT}" >&2; exit 2; }
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output is not empty: ${OUTPUT_DIR}" >&2
    exit 2
fi

export PYTHONPATH=${GALACTIKTOK_ROOT}/src:${PYTHONPATH:-}
mkdir -p "${OUTPUT_DIR}"
python -u -m euclid.pretrain_galactiktok_vis \
    --dataset "${DATASET}" \
    --fits-root "${FITS_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_vis_galactiktok_test \
    --max-pairs 2048 \
    --image-size 96 \
    --patch-size 8 \
    --embed-dim 256 \
    --num-heads 8 \
    --num-encoder-blocks 3 \
    --num-decoder-blocks 3 \
    --bottleneck-dim 8 \
    --mask-fraction 0.5 \
    --batch-size 64 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --max-epochs 3 \
    --warmup-epochs 1 \
    --patience 3 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed
