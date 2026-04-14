#!/bin/bash
#SBATCH --job-name=cweb_clip
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# ── environment ──────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip

mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/logs
cd ${REPO_DIR}

# ── run ───────────────────────────────────────────────────────────────────────
# train.py uses relative imports so must be invoked as a module from repo root
python -m cosmosweb.train \
    --dataset      ${OUTPUT_DIR}/cosmosweb_dataset.h5 \
    --output       ${OUTPUT_DIR}/checkpoints/cosmosweb_clip.ckpt \
    --log_dir      ${OUTPUT_DIR}/logs \
    --embed_dim    256 \
    --sfh_input_dim 50 \
    --batch_size   256 \
    --max_epochs   100 \
    --lr           1e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --num_workers  4 \
    --precision    16-mixed
