#!/bin/bash
#SBATCH --job-name=cweb_zoobot_v2b
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train_zoobot_v2b.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/train_zoobot_v2b.err
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

# Second independent run on the v2 dataset (identical hyperparameters, different
# random seed via PyTorch default seeding) to assess training stochasticity.
# v2 original results are preserved under cosmosweb_zoobot_v2-* and logs_zoobot_v2/.

# ── run ───────────────────────────────────────────────────────────────────────
python -m cosmosweb.train_zoobot \
    --dataset       ${OUTPUT_DIR}/cosmosweb_dataset_v2.h5 \
    --stamp_root    /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
    --filter        F277W \
    --zoobot_ckpt   ${ZOOBOT_CKPT} \
    --output        ${OUTPUT_DIR}/checkpoints/cosmosweb_zoobot_v2b.ckpt \
    --log_dir       ${OUTPUT_DIR}/logs_zoobot_v2b \
    --run_name      cosmosweb_zoobot_v2b \
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
