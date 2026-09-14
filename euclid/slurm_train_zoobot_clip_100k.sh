#!/bin/bash
#SBATCH --job-name=euclid_attn_med
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_attn_median_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_attn_median_%j.err
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
#   sbatch euclid/slurm_train_zoobot_clip_100k.sh \
#       [sfh_clip.h5] [stamp_root] [zoobot_source] [output_dir] [resume.ckpt]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}
mkdir -p "${HF_HOME}"

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
DATASET=${1:-${BASE_DIR}/sfh_clip_100k.h5}
STAMP_ROOT=${2:-${BASE_DIR}/zoobot_stamps_rmax}
ZOOBOT_SOURCE=${3:-hf_hub:mwalmsley/zoobot-encoder-euclid}
OUTPUT_DIR=${4:-${BASE_DIR}/training_transformer_median_v2}
RESUME_FROM=${5:-}

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

if [[ ! -f "${DATASET}" ]]; then
    echo "Missing preprocessed SFHs: ${DATASET}" >&2
    exit 2
fi
if [[ ! -d "${STAMP_ROOT}/VIS" ]]; then
    echo "Missing VIS JPEG directory: ${STAMP_ROOT}/VIS" >&2
    exit 2
fi
if [[ -f "${ZOOBOT_SOURCE}" ]]; then
    ENCODER_ARGS=(--zoobot-ckpt "${ZOOBOT_SOURCE}")
else
    ENCODER_ARGS=(--zoobot-model-name "${ZOOBOT_SOURCE}")
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" \
    --stamp-root "${STAMP_ROOT}" \
    --band VIS \
    "${ENCODER_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_vis_sfh_transformer_median_100k \
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
