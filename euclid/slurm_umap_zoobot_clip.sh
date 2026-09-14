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
#   sbatch euclid/slurm_umap_zoobot_clip.sh [checkpoint] [evaluation_dir] [output_pdf]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
TRAINING_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2
BEST_CHECKPOINT=/n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2/checkpoints/euclid_vis_sfh_transformer_median_100k-epoch=027-val_loss=4.4625.ckpt
LAST_CHECKPOINT=/n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2/checkpoints/last.ckpt
LOG_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k/training_transformer_median_v2/logs/euclid_vis_sfh_transformer_median_100k/version_0

CHECKPOINT=${1:-${BEST_CHECKPOINT}}
EVALUATION_DIR=${2:-${TRAINING_DIR}/evaluation_best}
OUTPUT=${3:-${EVALUATION_DIR}/euclid_clip_umap_diagnostics.pdf}
EMBEDDINGS=${EVALUATION_DIR}/validation_embeddings.npz
PER_OBJECT=${EVALUATION_DIR}/per_object.csv
DATASET=${BASE_DIR}/sfh_clip_100k.h5

echo "Checkpoint provenance: ${CHECKPOINT}"
echo "Last checkpoint available: ${LAST_CHECKPOINT}"
echo "Training logs: ${LOG_DIR}"
echo "Evaluation input: ${EVALUATION_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${DATASET}" ]]; then
    echo "Missing preprocessed SFHs: ${DATASET}" >&2
    exit 2
fi
if [[ ! -f "${EMBEDDINGS}" || ! -f "${PER_OBJECT}" ]]; then
    echo "Missing evaluation outputs in ${EVALUATION_DIR}" >&2
    echo "Create them with:" >&2
    echo "sbatch euclid/slurm_evaluate_zoobot_clip.sh '${CHECKPOINT}' '${EVALUATION_DIR}'" >&2
    exit 2
fi

export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "$(dirname "${OUTPUT}")"
cd "${REPO_DIR}"

python -u -m euclid.umap_zoobot_clip \
    --embeddings "${EMBEDDINGS}" \
    --dataset "${DATASET}" \
    --per-object "${PER_OBJECT}" \
    --output "${OUTPUT}" \
    --run-label "$(basename "${CHECKPOINT}" .ckpt)" \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --seed 42
