#!/bin/bash
#SBATCH --job-name=euclid_sfh_ae_clip
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_sfh_ae_clip_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_sfh_ae_clip_%j.err
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
#   sbatch euclid/slurm_train_zoobot_clip_100k_sfh_autoencoder.sh \
#       AUTOENCODER_CKPT [dataset] [stamp_root] [output_dir] [resume.ckpt]

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 AUTOENCODER_CKPT [dataset] [stamp_root] [output_dir] [resume.ckpt]" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
AUTOENCODER_CKPT=$1
DATASET=${2:-${BASE_DIR}/sfh_clip_100k.h5}
STAMP_ROOT=${3:-${BASE_DIR}/zoobot_stamps_rmax}
OUTPUT_DIR=${4:-${BASE_DIR}/training_transformer_median_sfh_autoencoder}
RESUME_FROM=${5:-}

if [[ ! -f "${AUTOENCODER_CKPT}" || ! -f "${DATASET}" ]]; then
    echo "Missing autoencoder checkpoint or dataset" >&2
    exit 2
fi
if [[ ! -d "${STAMP_ROOT}/VIS" ]]; then
    echo "Missing VIS stamps: ${STAMP_ROOT}/VIS" >&2
    exit 2
fi
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ -z "${RESUME_FROM}" ]]; then
    echo "Output is not empty; choose a new directory or pass a resume checkpoint: ${OUTPUT_DIR}" >&2
    exit 2
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" \
    --stamp-root "${STAMP_ROOT}" \
    --band VIS \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_vis_sfh_autoencoder_100k \
    --sfh-pretrained-checkpoint "${AUTOENCODER_CKPT}" \
    --sfh-reconstruction-weight 0.1 \
    --sfh-reconstruction-w1-weight 0.5 \
    --sfh-decoder-layers 2 \
    --no-sample-posterior \
    --batch-size 128 \
    --queue-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --image-size 224 \
    --embed-dim 256 \
    --sfh-encoder transformer \
    --sfh-d-model 128 \
    --sfh-n-heads 4 \
    --sfh-n-layers 4 \
    --sfh-lr-scale 3 \
    --max-epochs 50 \
    --warmup-epochs 2 \
    --patience 10 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --unfreeze-blocks 0 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    "${RESUME_ARGS[@]}"
