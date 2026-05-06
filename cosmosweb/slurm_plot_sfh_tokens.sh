#!/bin/bash
#SBATCH --job-name=cweb_sfh_tokens
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/plot_sfh_tokens.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/plot_sfh_tokens.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip
CAT=/n23data2/cosmosweb/catalogs/DR1/data/catalog/COSMOSWeb_mastercatalog_v1.fits

cd ${REPO_DIR}

python -m cosmosweb.plot_sfh_tokens \
    --h5        ${OUTPUT_DIR}/cosmosweb_dataset_v4.h5 \
    --catalog   ${CAT} \
    --output    ${OUTPUT_DIR}/sfh_tokens_diagnostics.pdf \
    --n_gal     20 \
    --n_samples  8 \
    --n_pop     300 \
    --seed      42
