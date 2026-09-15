#!/bin/bash
# Submit evaluation and dependent UMAP jobs for the partial-unfreezing run.

set -euo pipefail

# Run this from the AstroCLIP repository root on a Candide login node.
# Usage:
#   bash euclid/submit_unfreeze1_diagnostics.sh [checkpoint] [evaluation_dir] [output_pdf]

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

evaluation_submission=$(sbatch --parsable \
    euclid/slurm_evaluate_zoobot_clip.sh \
    "${CHECKPOINT}" "${EVALUATION_DIR}" "${PAIR_SPLIT}")
evaluation_job=${evaluation_submission%%;*}

umap_submission=$(sbatch --parsable \
    --dependency="afterok:${evaluation_job}" \
    euclid/slurm_umap_zoobot_clip.sh \
    "${CHECKPOINT}" "${EVALUATION_DIR}" "${OUTPUT_PDF}")
umap_job=${umap_submission%%;*}

echo "Evaluation job: ${evaluation_job}"
echo "UMAP job:       ${umap_job} (starts after evaluation succeeds)"
echo "Metrics:        ${EVALUATION_DIR}/metrics.json"
echo "Embeddings:     ${EVALUATION_DIR}/validation_embeddings.npz"
echo "Diagnostic PDF: ${OUTPUT_PDF}"
