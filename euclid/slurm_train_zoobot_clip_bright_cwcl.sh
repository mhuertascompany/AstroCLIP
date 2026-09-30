#!/bin/bash
#SBATCH --job-name=bright_cwcl
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/train_bright_cwcl_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/train_bright_cwcl_%j.err
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

# Frozen backbones and unrestricted MLP adapters trained with CWCL.
# Usage: sbatch $0 AUTOENCODER_CKPT [dataset] [stamp_root] [output_dir] [resume.ckpt]

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 AUTOENCODER_CKPT [dataset] [stamp_root] [output_dir] [resume.ckpt]" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
AUTOENCODER_CKPT=$1
DATASET=${2:-${BASE_DIR}/sfh_clip_150k.h5}
STAMP_ROOT=${3:-${BASE_DIR}/zoobot_stamps_rmax}
OUTPUT_DIR=${4:-${BASE_DIR}/training_bright_frozen_cwcl}
RESUME_FROM=${5:-}

[[ -f "${AUTOENCODER_CKPT}" ]] || { echo "Missing autoencoder checkpoint: ${AUTOENCODER_CKPT}" >&2; exit 2; }
[[ -f "${DATASET}" ]] || { echo "Missing dataset: ${DATASET}" >&2; exit 2; }
[[ -d "${STAMP_ROOT}/VIS" ]] || { echo "Missing VIS stamps: ${STAMP_ROOT}/VIS" >&2; exit 2; }
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ -z "${RESUME_FROM}" ]]; then
    echo "Output is not empty; choose a new directory or pass a resume checkpoint: ${OUTPUT_DIR}" >&2
    exit 2
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    [[ -f "${RESUME_FROM}" ]] || { echo "Missing resume checkpoint: ${RESUME_FROM}" >&2; exit 2; }
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" \
    --stamp-root "${STAMP_ROOT}" \
    --band VIS \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_bright_cwcl_150k \
    --sfh-pretrained-checkpoint "${AUTOENCODER_CKPT}" \
    --freeze-sfh-encoder \
    --sfh-reconstruction-weight 0 \
    --alignment-objective cwcl \
    --cwcl-similarity-temperature 0.1 \
    --cwcl-reverse-exact-weight 1 \
    --image-projection mlp \
    --image-projection-hidden-dim 1024 \
    --image-projection-hidden-layers 2 \
    --sfh-projection mlp \
    --sfh-projection-hidden-dim 1024 \
    --sfh-projection-hidden-layers 2 \
    --no-sample-posterior \
    --batch-size 128 \
    --soft-positive-weight 0 \
    --queue-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --image-size 224 \
    --embed-dim 256 \
    --sfh-encoder transformer \
    --sfh-d-model 128 \
    --sfh-n-heads 4 \
    --sfh-n-layers 4 \
    --max-epochs 60 \
    --warmup-epochs 3 \
    --patience 12 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --unfreeze-blocks 0 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    "${RESUME_ARGS[@]}"
