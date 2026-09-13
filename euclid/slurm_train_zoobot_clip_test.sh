#!/bin/bash
#SBATCH --job-name=euclid_clip_test
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_clip_test_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_clip_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
DATASET=${1:-${BASE_DIR}/sfh_clip_100k.h5}
STAMP_ROOT=${2:-${BASE_DIR}/zoobot_stamps_rmax}
ZOOBOT_CKPT=${3:-/n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/family_2.ckpt}
OUTPUT_DIR=${4:-${BASE_DIR}/training_test}

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" \
    --stamp-root "${STAMP_ROOT}" \
    --band VIS \
    --zoobot-ckpt "${ZOOBOT_CKPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_vis_sfh_smoke \
    --sample-posterior \
    --max-pairs 1024 \
    --batch-size 32 \
    --queue-size 256 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --image-size 224 \
    --embed-dim 256 \
    --max-epochs 2 \
    --warmup-epochs 1 \
    --patience 2 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --unfreeze-blocks 0 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed
