#!/bin/bash
#SBATCH --job-name=bright_clip_refresh
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/refresh_clip_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/refresh_clip_%j.err
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Refresh UMAP metadata and the local explorer bundle after updating scalar
# morphology fields. Embeddings are reused; no checkpoint or GPU is needed.
#
# Usage:
#   sbatch euclid/slurm_refresh_bright_clip_explorer.sh \
#       [source_evaluation_dir] [output_dir] [sfh_clip.h5] [stamp_root]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO=/n03data/huertas/python/AstroCLIP
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
SOURCE=${1:-${BASE}/training_bright_frozen_mlp/evaluation_best}
OUTPUT=${2:-${BASE}/training_bright_frozen_mlp/evaluation_best_full_zoobot}
DATASET=${3:-${BASE}/sfh_clip_150k.h5}
STAMPS=${4:-${BASE}/zoobot_stamps_rmax}

SOURCE_ARCHIVE=${SOURCE}/euclid_clip_umap_diagnostics.npz
PER_OBJECT=${SOURCE}/per_object.csv
DIAGNOSTIC=${OUTPUT}/euclid_clip_umap_diagnostics.pdf
BUNDLE=${OUTPUT}/explorer_bundle

for path in "${DATASET}" "${SOURCE_ARCHIVE}" "${PER_OBJECT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 2
    fi
done
if [[ ! -d "${STAMPS}/VIS" ]]; then
    echo "Missing VIS stamps: ${STAMPS}/VIS" >&2
    exit 2
fi
if [[ -d "${OUTPUT}" ]] && [[ -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output directory is not empty: ${OUTPUT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT}"
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
cd "${REPO}"

python -u -m euclid.refresh_clip_diagnostics \
    --archive "${SOURCE_ARCHIVE}" \
    --dataset "${DATASET}" \
    --per-object "${PER_OBJECT}" \
    --output-archive "${DIAGNOSTIC%.pdf}.npz" \
    --output-pdf "${DIAGNOSTIC}" \
    --run-label 'bright frozen MLP; full-sample ZooBot classifications'

python -u -m euclid.export_explorer_bundle \
    --dataset "${DATASET}" \
    --stamp-root "${STAMPS}" \
    --umap "${DIAGNOSTIC%.pdf}.npz" \
    --output-dir "${BUNDLE}" \
    --band VIS

echo "Refreshed diagnostic: ${DIAGNOSTIC}"
echo "Download explorer bundle: ${BUNDLE}"
