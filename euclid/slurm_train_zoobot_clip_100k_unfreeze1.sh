#!/bin/bash
#SBATCH --job-name=euclid_attn_unfr1
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_attn_unfreeze1_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_attn_unfreeze1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Controlled comparison with the frozen 100k median-SFH baseline. The final
# ConvNeXt feature stage is trainable at 5e-6; the projection and SFH
# transformer train at 1e-4. Use a new output directory for an independent run.
#
# Usage:
#   sbatch euclid/slurm_train_zoobot_clip_100k_unfreeze1.sh \
#       [sfh_clip.h5] [stamp_root] [zoobot_source] [output_dir]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}
mkdir -p "${HF_HOME}"

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
DATASET=${1:-${BASE_DIR}/sfh_clip_100k.h5}
STAMP_ROOT=${2:-${BASE_DIR}/zoobot_stamps_rmax}
ZOOBOT_SOURCE=${3:-hf_hub:mwalmsley/zoobot-encoder-euclid}
OUTPUT_DIR=${4:-${BASE_DIR}/training_transformer_median_unfreeze1}

if [[ ! -f "${DATASET}" ]]; then
    echo "Missing preprocessed SFHs: ${DATASET}" >&2
    exit 2
fi
if [[ ! -d "${STAMP_ROOT}/VIS" ]]; then
    echo "Missing VIS JPEG directory: ${STAMP_ROOT}/VIS" >&2
    exit 2
fi
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output is not empty; choose a new directory: ${OUTPUT_DIR}" >&2
    exit 2
fi
if [[ -f "${ZOOBOT_SOURCE}" ]]; then
    ENCODER_ARGS=(--zoobot-ckpt "${ZOOBOT_SOURCE}")
else
    ENCODER_ARGS=(--zoobot-model-name "${ZOOBOT_SOURCE}")
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" \
    --stamp-root "${STAMP_ROOT}" \
    --band VIS \
    "${ENCODER_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_vis_sfh_transformer_median_unfreeze1 \
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
    --sfh-lr-scale 1 \
    --max-epochs 100 \
    --warmup-epochs 5 \
    --patience 20 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --unfreeze-blocks 1 \
    --backbone-lr-scale 0.05 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed
