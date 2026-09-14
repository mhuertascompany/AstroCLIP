#!/bin/bash
#SBATCH --job-name=euclid_attn_pilot
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_attn_pilot_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_attn_pilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# A 10k-pair diagnostic run. It removes the MoCo queue and uses posterior
# medians so we can test optimization without stale or false-negative keys and
# without posterior-draw noise.

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}
mkdir -p "${HF_HOME}"

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
DATASET=${1:-${BASE_DIR}/sfh_clip_100k.h5}
STAMP_ROOT=${2:-${BASE_DIR}/zoobot_stamps_rmax}
ZOOBOT_SOURCE=${3:-hf_hub:mwalmsley/zoobot-encoder-euclid}
OUTPUT_DIR=${4:-${BASE_DIR}/training_pilot_transformer_median}

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

if [[ -f "${ZOOBOT_SOURCE}" ]]; then
    ENCODER_ARGS=(--zoobot-ckpt "${ZOOBOT_SOURCE}")
else
    ENCODER_ARGS=(--zoobot-model-name "${ZOOBOT_SOURCE}")
fi

python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" \
    --stamp-root "${STAMP_ROOT}" \
    --band VIS \
    "${ENCODER_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_vis_sfh_transformer_pilot_median \
    --no-sample-posterior \
    --max-pairs 10000 \
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
    --max-epochs 30 \
    --warmup-epochs 2 \
    --patience 8 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --unfreeze-blocks 0 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed
