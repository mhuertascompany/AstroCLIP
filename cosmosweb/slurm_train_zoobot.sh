#!/bin/bash
#SBATCH --job-name=cweb_zoobot
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train_zoobot.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train_zoobot.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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

# ── ZooBOT family checkpoint (frozen backbone, richest morphology supervision) ─
ZOOBOT_CKPT=/n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/family_2.ckpt

# ── run ───────────────────────────────────────────────────────────────────────
python -m cosmosweb.train_zoobot \
    --dataset      ${OUTPUT_DIR}/cosmosweb_dataset_v2.h5 \
    --stamp_root   /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
    --filter       F277W \
    --zoobot_ckpt  ${ZOOBOT_CKPT} \
    --output       ${OUTPUT_DIR}/checkpoints/cosmosweb_zoobot_v1.ckpt \
    --log_dir      ${OUTPUT_DIR}/logs_zoobot \
    --embed_dim    256 \
    --sfh_input_dim 50 \
    --batch_size   128 \
    --max_epochs   50 \
    --lr           1e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --num_workers  4 \
    --precision    16-mixed
