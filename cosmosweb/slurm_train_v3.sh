#!/bin/bash
#SBATCH --job-name=cweb_clip_v3
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train_v3.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train_v3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ──────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip

mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/logs
cd ${REPO_DIR}

# Changes vs v2:
#   - Dataset: cosmosweb_dataset_v3.h5 (log-spaced SFH time grid + fill_value fix)
#   - Checkpoint/logs use v3 suffix; v2 is left untouched for comparison.

# ── run ───────────────────────────────────────────────────────────────────────
python -m cosmosweb.train \
    --dataset      ${OUTPUT_DIR}/cosmosweb_dataset_v3.h5 \
    --output       ${OUTPUT_DIR}/checkpoints/cosmosweb_clip_v3.ckpt \
    --log_dir      ${OUTPUT_DIR}/logs_v3 \
    --embed_dim    256 \
    --sfh_input_dim 50 \
    --batch_size   256 \
    --max_epochs   50 \
    --lr           1e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --num_workers  4 \
    --precision    16-mixed
