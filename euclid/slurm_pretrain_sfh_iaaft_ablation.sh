#!/bin/bash
#SBATCH --job-name=sfh_iaaft_ae
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sfh_iaaft_ae_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sfh_iaaft_ae_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Train matched deterministic-median autoencoders.
# Usage: sbatch $0 control|iaaft [dataset] [pair_split] [output_dir] [resume.ckpt]

if [[ $# -lt 1 || ( "$1" != "control" && "$1" != "iaaft" ) ]]; then
    echo "Usage: sbatch $0 control|iaaft [dataset] [pair_split] [output_dir] [resume.ckpt]" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

MODE=$1
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
if [[ "${MODE}" == "control" ]]; then
    DEFAULT_DATASET=${BASE}/sfh_clip_150k.h5
    DEFAULT_OUTPUT=${BASE}/sfh_autoencoder_median_control_150k
    RUN_NAME=euclid_sfh_autoencoder_median_control_150k
else
    DEFAULT_DATASET=${BASE}/sfh_iaaft_recent10_150k.h5
    DEFAULT_OUTPUT=${BASE}/sfh_autoencoder_iaaft_recent10_150k
    RUN_NAME=euclid_sfh_autoencoder_iaaft_recent10_150k
fi
DATASET=${2:-${DEFAULT_DATASET}}
SPLIT=${3:-${BASE}/training_bright_frozen_unrestricted_ae_no_edgeon/pair_split.npz}
OUTPUT_DIR=${4:-${DEFAULT_OUTPUT}}
RESUME_FROM=${5:-}

[[ -f "${DATASET}" ]] || { echo "Missing dataset: ${DATASET}" >&2; exit 2; }
[[ -f "${SPLIT}" ]] || { echo "Missing pair split: ${SPLIT}" >&2; exit 2; }
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ -z "${RESUME_FROM}" ]]; then
    echo "Output is not empty; choose a new directory or pass a resume checkpoint: ${OUTPUT_DIR}" >&2
    exit 2
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    [[ -f "${RESUME_FROM}" ]] || { echo "Missing resume checkpoint: ${RESUME_FROM}" >&2; exit 2; }
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

mkdir -p "${OUTPUT_DIR}"
python -u -m euclid.pretrain_sfh_autoencoder \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --input-mode median \
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
