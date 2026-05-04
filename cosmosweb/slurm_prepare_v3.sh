#!/bin/bash
#SBATCH --job-name=cweb_prepare_v3
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/prepare_v3.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/prepare_v3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ───────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip

mkdir -p ${OUTPUT_DIR}
cd ${REPO_DIR}

CAT_DIR=/n23data2/cosmosweb/catalogs/DR1/data/catalog

# Changes vs v2:
#   - SFH_T_FRAC: log-spaced geomspace(1e-3, 1, 50) instead of linspace(0, 1, 50)
#   - fill_value: (sfr_frac[0], 0) instead of 0 — extends most-recent CIGALE
#     bin to t=0 rather than treating it as no star formation
# Output is cosmosweb_dataset_v3.h5; v2 is left untouched for comparison.

python cosmosweb/prepare_dataset.py \
    --morpho_cat  /n03data/huertas/COSMOS-Web/ilbert_finetune/ilbert_visual_zoobot_morphology.fits \
    --photom_cat  ${CAT_DIR}/COSMOSWeb_mastercatalog_v1_photom_primary.fits \
    --lephare_cat ${CAT_DIR}/COSMOSWeb_mastercatalog_v1_lephare.fits \
    --cigale_cat  ${CAT_DIR}/COSMOSWeb_mastercatalog_v1_cigale.fits \
    --img_dir     /n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/ \
    --output      ${OUTPUT_DIR}/cosmosweb_dataset_v3.h5 \
    --stamp_size  64
