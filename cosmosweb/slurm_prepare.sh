#!/bin/bash
#SBATCH --job-name=cweb_prepare
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/prepare.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/prepare.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ──────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip

mkdir -p ${OUTPUT_DIR}
cd ${REPO_DIR}

CAT_DIR=/n23data2/cosmosweb/catalogs/DR1/data/catalog

# ── run ───────────────────────────────────────────────────────────────────────
python cosmosweb/prepare_dataset.py \
    --morpho_db   /n07data/ilbert/COSMOS-Web/photoz_MASTER_v3.1.0/MORPHO/visualmorpho_COSMOSWeb_v7.db \
    --photom_cat  ${CAT_DIR}/COSMOSWeb_mastercatalog_v1_photom_primary.fits \
    --lephare_cat ${CAT_DIR}/COSMOSWeb_mastercatalog_v1_lephare.fits \
    --cigale_cat  ${CAT_DIR}/COSMOSWeb_mastercatalog_v1_cigale.fits \
    --img_dir     /n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/ \
    --output      ${OUTPUT_DIR}/cosmosweb_dataset.h5 \
    --stamp_size  64
