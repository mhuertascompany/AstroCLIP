#!/bin/bash
#SBATCH --job-name=cweb_rising_sfh
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/rising_sfh_retrieval.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/rising_sfh_retrieval.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ───────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip

cd ${REPO_DIR}

# ── run ───────────────────────────────────────────────────────────────────────
python -m cosmosweb.rising_sfh_retrieval \
    --npz          ${OUTPUT_DIR}/cosmosweb_umap_zoobot_v8.npz \
    --h5           ${OUTPUT_DIR}/cosmosweb_dataset_v6.h5 \
    --output       ${OUTPUT_DIR}/rising_sfh_retrieval.pdf \
    --n_queries    20 \
    --k            9 \
    --qi_threshold -0.03 \
    --seed         42
