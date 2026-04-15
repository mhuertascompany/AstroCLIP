#!/bin/bash
#SBATCH --job-name=cweb_viz
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/visualize.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/visualize.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ───────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip

# ── run ───────────────────────────────────────────────────────────────────────
python cosmosweb/visualize_pairs.py \
    --dataset  ${OUTPUT_DIR}/cosmosweb_dataset_v2.h5 \
    --output   ${OUTPUT_DIR}/pair_examples_v2.pdf \
    --n_pages  5 \
    --seed     42
