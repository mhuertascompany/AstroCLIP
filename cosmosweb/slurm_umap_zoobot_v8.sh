#!/bin/bash
#SBATCH --job-name=cweb_umap_zoobot_v8
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/umap_zoobot_v8.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/umap_zoobot_v8.err
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
CKPT=${OUTPUT_DIR}/checkpoints/cosmosweb_zoobot_v8-epoch=066-val_loss=3.9846.ckpt

cd ${REPO_DIR}

# ── run ───────────────────────────────────────────────────────────────────────
python -m cosmosweb.umap_embeddings_zoobot_v8 \
    --checkpoint       ${CKPT} \
    --dataset          ${OUTPUT_DIR}/cosmosweb_dataset_v6.h5 \
    --stamp_root       /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
    --filter           F277W \
    --z_low            1.0 \
    --z_high           3.0 \
    --output           ${OUTPUT_DIR}/umap_plots_zoobot_v8.pdf \
    --npz_output       ${OUTPUT_DIR}/cosmosweb_umap_zoobot_v8.npz \
    --batch_size       128 \
    --n_neighbors      15 \
    --min_dist         0.1 \
    --clustering           pca_hdbscan \
    --n_pca_components     50 \
    --min_cluster_size     200 \
    --min_samples          30 \
    --device               cuda
