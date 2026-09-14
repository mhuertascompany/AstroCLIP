#!/bin/bash
#SBATCH --job-name=euclid_clip_eval
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/evaluate_clip_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/evaluate_clip_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_evaluate_zoobot_clip.sh [checkpoint] [output_dir]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
TRAINING_DIR=${BASE_DIR}/training_transformer_median_v2
CHECKPOINT=${1:-${TRAINING_DIR}/checkpoints/euclid_vis_sfh_transformer_median_100k-epoch=027-val_loss=4.4625.ckpt}
OUTPUT_DIR=${2:-${TRAINING_DIR}/evaluation_best}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}"

python -u -m euclid.evaluate_zoobot_clip \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${BASE_DIR}/sfh_clip_100k.h5" \
    --stamp-root "${BASE_DIR}/zoobot_stamps_rmax" \
    --split "${TRAINING_DIR}/pair_split.npz" \
    --output-dir "${OUTPUT_DIR}" \
    --band VIS \
    --batch-size 128 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 512 \
    --n-permutations 200 \
    --permutation-size 5000 \
    --shape-subset 2000 \
    --posterior-subset 1000 \
    --posterior-draws 5 \
    --device cuda
