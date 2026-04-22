#!/bin/bash
#SBATCH --job-name=cweb_zoobot_v2
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train_zoobot_v2.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train_zoobot_v2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ───────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip

mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/logs
cd ${REPO_DIR}

ZOOBOT_CKPT=/n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/family_2.ckpt

# ── run ───────────────────────────────────────────────────────────────────────
# Key changes vs v1:
#   --queue_size 1024      : 8x more negatives per step; smaller than 4096 so queue
#                            fills with decent embeddings before warmup ends
#   --momentum   0.995     : EMA rate for key encoders
#   --lr         1e-4      : conservative lr (3e-4 caused temperature collapse)
#   --max_epochs 100       : more epochs; harder negatives → slower convergence
#   --warmup_epochs 3      : short warmup (10 was too long: LR≈0 for first 10 epochs,
#                            queue fills with near-random embeddings → bad signal)
#   --patience   20        : more patience (early epochs settle temperature)
# logit_scale T = exp(log_temp) initialised to ~14.3, clamped to [1.0, 100.0].
python -m cosmosweb.train_zoobot \
    --dataset       ${OUTPUT_DIR}/cosmosweb_dataset_v2.h5 \
    --stamp_root    /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
    --filter        F277W \
    --zoobot_ckpt   ${ZOOBOT_CKPT} \
    --output        ${OUTPUT_DIR}/checkpoints/cosmosweb_zoobot_v2.ckpt \
    --log_dir       ${OUTPUT_DIR}/logs_zoobot_v2 \
    --run_name      cosmosweb_zoobot_v2 \
    --embed_dim     256 \
    --sfh_input_dim 50 \
    --queue_size    1024 \
    --momentum      0.995 \
    --batch_size    128 \
    --max_epochs    100 \
    --lr            1e-4 \
    --weight_decay  0.05 \
    --warmup_epochs 3 \
    --patience      20 \
    --num_workers   8 \
    --precision     16-mixed
