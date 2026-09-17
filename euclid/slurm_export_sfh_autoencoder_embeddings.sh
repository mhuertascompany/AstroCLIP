#!/bin/bash
#SBATCH --job-name=sfh_ae_embed
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_ae_embed_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_ae_embed_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_export_sfh_autoencoder_embeddings.sh \
#       AUTOENCODER_CKPT [output_dir]

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 AUTOENCODER_CKPT [output_dir]" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
CHECKPOINT=$1
OUTPUT_DIR=${2:-${BASE_DIR}/sfh_autoencoder_v1/explorer_validation}
DATASET=${BASE_DIR}/sfh_clip_100k.h5
SPLIT=${BASE_DIR}/training_transformer_median_v2/pair_split.npz

for path in "${CHECKPOINT}" "${DATASET}" "${SPLIT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 2
    fi
done
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output is not empty: ${OUTPUT_DIR}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd /n03data/huertas/python/AstroCLIP
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

python -u -m euclid.export_sfh_autoencoder_embeddings \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --batch-size 512 \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --seed 42
