#!/bin/bash
#SBATCH --job-name=euclid_unfr1_diag
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/unfreeze1_diagnostics_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/unfreeze1_diagnostics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# Evaluate the partial-unfreezing checkpoint and then create its UMAP atlas in
# the same allocation. set -e ensures plotting is skipped if evaluation fails.

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_unfreeze1_diagnostics.sh \
#       [checkpoint] [evaluation_dir] [output_pdf]

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
TRAINING_DIR=${BASE_DIR}/training_transformer_median_unfreeze1
DEFAULT_CHECKPOINT=${TRAINING_DIR}/checkpoints/euclid_vis_sfh_transformer_median_unfreeze1-epoch=017-val_loss=4.5082.ckpt

CHECKPOINT=${1:-${DEFAULT_CHECKPOINT}}
EVALUATION_DIR=${2:-${TRAINING_DIR}/evaluation_best}
OUTPUT_PDF=${3:-${EVALUATION_DIR}/euclid_clip_umap_diagnostics.pdf}
CHECKPOINT_DIR=${CHECKPOINT%/*}
CHECKPOINT_TRAINING_DIR=${CHECKPOINT_DIR%/*}
PAIR_SPLIT=${CHECKPOINT_TRAINING_DIR}/pair_split.npz

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${PAIR_SPLIT}" ]]; then
    echo "Missing saved train/validation split: ${PAIR_SPLIT}" >&2
    exit 2
fi

cd "${REPO_DIR}"

echo "Evaluating: ${CHECKPOINT}"
bash euclid/slurm_evaluate_zoobot_clip.sh \
    "${CHECKPOINT}" "${EVALUATION_DIR}" "${PAIR_SPLIT}"

echo "Creating UMAP diagnostics: ${OUTPUT_PDF}"
bash euclid/slurm_umap_zoobot_clip.sh \
    "${CHECKPOINT}" "${EVALUATION_DIR}" "${OUTPUT_PDF}"

echo "Metrics:        ${EVALUATION_DIR}/metrics.json"
echo "Embeddings:     ${EVALUATION_DIR}/validation_embeddings.npz"
echo "Diagnostic PDF: ${OUTPUT_PDF}"
