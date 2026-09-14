#!/bin/bash
#SBATCH --job-name=euclid_clip_umap
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/umap_clip_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/umap_clip_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_umap_zoobot_clip.sh [evaluation_dir] [output_pdf]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
EVALUATION_DIR=${1:-${BASE_DIR}/training_transformer_median_v2/evaluation_best}
OUTPUT=${2:-${EVALUATION_DIR}/euclid_clip_umap_diagnostics.pdf}

export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "$(dirname "${OUTPUT}")"
cd "${REPO_DIR}"

python -u -m euclid.umap_zoobot_clip \
    --embeddings "${EVALUATION_DIR}/validation_embeddings.npz" \
    --dataset "${BASE_DIR}/sfh_clip_100k.h5" \
    --per-object "${EVALUATION_DIR}/per_object.csv" \
    --output "${OUTPUT}" \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --seed 42
