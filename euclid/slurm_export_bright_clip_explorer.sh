#!/bin/bash
#SBATCH --job-name=bright_clip_export
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/evaluate_clip_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/evaluate_clip_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_export_bright_clip_explorer.sh [checkpoint] [output_dir] [pair_split]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TRAINING_DIR=${BASE_DIR}/training_bright_frozen_mlp
CHECKPOINT=${1:-${TRAINING_DIR}/checkpoints/euclid_bright_frozen_mlp_150k-epoch=020-val_loss=4.2759.ckpt}
OUTPUT_DIR=${2:-${TRAINING_DIR}/evaluation_best}
CHECKPOINT_DIR=${CHECKPOINT%/*}
CHECKPOINT_TRAINING_DIR=${CHECKPOINT_DIR%/*}
PAIR_SPLIT=${3:-${CHECKPOINT_TRAINING_DIR}/pair_split.npz}

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${PAIR_SPLIT}" ]]; then
    echo "Missing saved train/validation split: ${PAIR_SPLIT}" >&2
    exit 2
fi
if [[ ! -f "${BASE_DIR}/sfh_clip_150k.h5" ]]; then
    echo "Missing preprocessed SFHs: ${BASE_DIR}/sfh_clip_150k.h5" >&2
    exit 2
fi
if [[ ! -d "${BASE_DIR}/zoobot_stamps_rmax/VIS" ]]; then
    echo "Missing VIS JPEG directory: ${BASE_DIR}/zoobot_stamps_rmax/VIS" >&2
    exit 2
fi

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}"

python -u -m euclid.evaluate_zoobot_clip \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${BASE_DIR}/sfh_clip_150k.h5" \
    --stamp-root "${BASE_DIR}/zoobot_stamps_rmax" \
    --split "${PAIR_SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --band VIS \
    --batch-size 128 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 512 \
    --n-permutations 200 \
    --permutation-size 5000 \
    --shape-subset 2000 \
    --posterior-subset 1000 \
    --posterior-draws 5 \
    --device cuda

# Generate all diagnostic coordinates and package the same validation objects.
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
python -u -m euclid.umap_zoobot_clip \
    --embeddings "${OUTPUT_DIR}/validation_embeddings.npz" \
    --dataset "${BASE_DIR}/sfh_clip_150k.h5" \
    --per-object "${OUTPUT_DIR}/per_object.csv" \
    --output "${OUTPUT_DIR}/euclid_clip_umap_diagnostics.pdf" \
    --run-label "$(basename "${CHECKPOINT}" .ckpt)" \
    --seed 42

python -u -m euclid.export_explorer_bundle \
    --dataset "${BASE_DIR}/sfh_clip_150k.h5" \
    --stamp-root "${BASE_DIR}/zoobot_stamps_rmax" \
    --umap "${OUTPUT_DIR}/euclid_clip_umap_diagnostics.npz" \
    --output-dir "${OUTPUT_DIR}/explorer_bundle" \
    --band VIS
