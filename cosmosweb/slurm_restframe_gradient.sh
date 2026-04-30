#!/bin/bash
#SBATCH --job-name=cweb_rf_gradient
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/rf_gradient.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/rf_gradient.err
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
CAT=/n23data2/cosmosweb/catalogs/DR1/data/catalog/COSMOSWeb_mastercatalog_v1.fits

cd ${REPO_DIR}

python -m cosmosweb.compute_restframe_gradient \
    --catalog  ${CAT} \
    --npz      ${OUTPUT_DIR}/cosmosweb_umap_zoobot_v2.npz
