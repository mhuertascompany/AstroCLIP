#!/bin/bash
#SBATCH --job-name=cweb_umap_zoobot_v3
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/umap_zoobot_v3.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/umap_zoobot_v3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ───────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip
CKPT=${OUTPUT_DIR}/checkpoints/cosmosweb_zoobot_v3-epoch=082-val_loss=3.9129.ckpt

cd ${REPO_DIR}

# ── run ───────────────────────────────────────────────────────────────────────
python -m cosmosweb.umap_embeddings_zoobot \
    --checkpoint       ${CKPT} \
    --dataset          ${OUTPUT_DIR}/cosmosweb_dataset_v3.h5 \
    --stamp_root       /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
    --filter           F277W \
    --output           ${OUTPUT_DIR}/umap_plots_zoobot_v3.pdf \
    --npz_output       ${OUTPUT_DIR}/cosmosweb_umap_zoobot_v3.npz \
    --batch_size       128 \
    --n_neighbors      15 \
    --min_dist         0.1 \
    --clustering       kmeans \
    --n_clusters       12 \
    --device           cuda
