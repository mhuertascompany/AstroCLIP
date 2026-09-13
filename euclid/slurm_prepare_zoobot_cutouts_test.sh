#!/bin/bash
#SBATCH --job-name=euclid_zoobot_test
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/zoobot_test_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/zoobot_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_prepare_zoobot_cutouts_test.sh \
#       [cutout_root] [catalog.fits] [output_directory] [sample_size]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
DEFAULT_CUTOUT_ROOT=/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_edfn100k/edfn_100k_cutouts
DEFAULT_CATALOG=/n03data/huertas/euclid/sfh_clip/edfn_100k/sfh_edfn100k/morphology_catalog_sfh_100k.fits
DEFAULT_OUTPUT=/n03data/huertas/euclid/sfh_clip/edfn_100k/zoobot_stamps_rmax_test100
CUTOUT_ROOT=${1:-${DEFAULT_CUTOUT_ROOT}}
CATALOG=${2:-${DEFAULT_CATALOG}}
OUTPUT=${3:-${DEFAULT_OUTPUT}}
SAMPLE_SIZE=${4:-100}

cd "${REPO_DIR}"

if [[ ! -f "${CUTOUT_ROOT}/manifest.csv" ]]; then
    echo "Missing ${CUTOUT_ROOT}/manifest.csv. CUTOUT_ROOT must be the parent of cutouts/VIS." >&2
    exit 2
fi
if [[ ! -d "${CUTOUT_ROOT}/cutouts/VIS" ]]; then
    echo "Missing ${CUTOUT_ROOT}/cutouts/VIS." >&2
    exit 2
fi
if [[ ! -f "${CATALOG}" ]]; then
    echo "Missing morphology catalog: ${CATALOG}" >&2
    exit 2
fi
python -c 'import skimage; print("scikit-image", skimage.__version__)'

python -u -m euclid.prepare_zoobot_cutouts \
    --cutout-root "${CUTOUT_ROOT}" \
    --catalog "${CATALOG}" \
    --output "${OUTPUT}" \
    --band VIS \
    --image-size 224 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --limit "${SAMPLE_SIZE}" \
    --resume
