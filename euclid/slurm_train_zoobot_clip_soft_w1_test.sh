#!/bin/bash
#SBATCH --job-name=euclid_softw1_test
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_soft_w1_test_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_soft_w1_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Two-epoch runtime check for the SFH-aware objective.
# Usage:
#   sbatch euclid/slurm_train_zoobot_clip_soft_w1_test.sh \
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
OUTPUT_DIR=${4:-${BASE_DIR}/training_test_soft_w1}

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
    --run-name euclid_vis_sfh_transformer_median_soft_w1_smoke \
    --no-sample-posterior \
    --max-pairs 1024 \
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
    --soft-positive-weight 0.25 \
    --soft-positive-k 8 \
    --max-epochs 2 \
    --warmup-epochs 1 \
    --patience 2 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --unfreeze-blocks 0 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed
