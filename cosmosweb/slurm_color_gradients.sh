#!/bin/bash
#SBATCH --job-name=cweb_color_gradients
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/color_gradients.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/color_gradients.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ───────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip
MASTER_CAT=/n23data2/cosmosweb-public/DR1/data/COSMOSWeb_mastercatalog_v1.fits

cd ${REPO_DIR}

# ── step 1: compute gradients ─────────────────────────────────────────────────
python -m cosmosweb.compute_color_gradients \
    --catalog    ${MASTER_CAT} \
    --output     ${OUTPUT_DIR}/color_gradients.fits \
    --chi2_max   5.0 \
    --BT_min     0.05 \
    --BT_max     0.95 \
    --merge_npz  ${OUTPUT_DIR}/cosmosweb_umap_zoobot_v2.npz
