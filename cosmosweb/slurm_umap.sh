#!/bin/bash
#SBATCH --job-name=cweb_umap
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/umap.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/umap.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ───────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip
# Update CKPT to the best v2 checkpoint after retraining
CKPT=${OUTPUT_DIR}/checkpoints/cosmosweb_clip_v2-epoch=NNN-val_loss=N.NNNN.ckpt

cd ${REPO_DIR}

# ── run ───────────────────────────────────────────────────────────────────────
# Script uses relative imports → must be invoked as a module from repo root
python -m cosmosweb.umap_embeddings \
    --checkpoint  ${CKPT} \
    --dataset     ${OUTPUT_DIR}/cosmosweb_dataset_v2.h5 \
    --output      ${OUTPUT_DIR}/umap_plots_v2.pdf \
    --npz_output  ${OUTPUT_DIR}/cosmosweb_umap_v2.npz \
    --batch_size  512 \
    --n_neighbors 15 \
    --min_dist    0.1 \
    --device      cuda
