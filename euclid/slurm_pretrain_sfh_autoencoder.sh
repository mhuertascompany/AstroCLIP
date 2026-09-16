#!/bin/bash
#SBATCH --job-name=sfh_autoencoder
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_autoencoder_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_autoencoder_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_pretrain_sfh_autoencoder.sh \
#       [dataset] [pair_split] [output_dir] [resume.ckpt]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
DATASET=${1:-${BASE_DIR}/sfh_clip_100k.h5}
SPLIT=${2:-${BASE_DIR}/training_transformer_median_v2/pair_split.npz}
OUTPUT_DIR=${3:-${BASE_DIR}/sfh_autoencoder_v1}
RESUME_FROM=${4:-}

if [[ ! -f "${DATASET}" || ! -f "${SPLIT}" ]]; then
    echo "Missing dataset or split: ${DATASET} ${SPLIT}" >&2
    exit 2
fi
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ -z "${RESUME_FROM}" ]]; then
    echo "Output is not empty; choose a new directory or pass a resume checkpoint: ${OUTPUT_DIR}" >&2
    exit 2
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    if [[ ! -f "${RESUME_FROM}" ]]; then
        echo "Missing resume checkpoint: ${RESUME_FROM}" >&2
        exit 2
    fi
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

mkdir -p "${OUTPUT_DIR}"
cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.pretrain_sfh_autoencoder \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_sfh_autoencoder_v1 \
    --batch-size 128 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --embed-dim 256 \
    --d-model 128 \
    --n-heads 4 \
    --encoder-layers 4 \
    --decoder-layers 2 \
    --mask-fraction 0.35 \
    --w1-weight 0.5 \
    --max-epochs 50 \
    --warmup-epochs 3 \
    --patience 10 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    "${RESUME_ARGS[@]}"
