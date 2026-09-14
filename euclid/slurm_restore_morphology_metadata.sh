#!/bin/bash
#SBATCH --job-name=euclid_add_morph
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/restore_morphology_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/restore_morphology_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
DATASET=${1:-${BASE_DIR}/sfh_clip_100k.h5}
CLEAN_CATALOG=${2:-${BASE_DIR}/sfh_edfn100k/catalog_sfh_100k.fits}
MER_CATALOG=${3:-${BASE_DIR}/sfh_edfn100k/morphology_catalog_sfh_100k.fits}

cd "${REPO_DIR}"

python -u -m euclid.restore_morphology_metadata \
    --dataset "${DATASET}" \
    --catalog "${CLEAN_CATALOG}" \
    --catalog "${MER_CATALOG}"
