#!/bin/bash
#SBATCH --job-name=edfn_bright_sfh
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_bright_sample_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_bright_sample_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 MAX_VIS_MAG N_OBJECTS [OUTPUT_DIR]" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE=/n03data/huertas/euclid/sfh_clip
SFH_FILE=/n17data/wozny/These/science_DR1/SFHs/ready_to_use_sfhs/EDFN_2fwhm_aper.h5
UTILS_DIR=/home/wozny/jobs/These/DR1_science/SFH/utils
MAX_VIS_MAG=$1
N_OBJECTS=$2
TAG=${MAX_VIS_MAG/./p}
OUTPUT=${3:-${BASE}/edfn_vislt${TAG}_${N_OBJECTS}}

cd "${REPO_DIR}"
python -u -m euclid.sample_edfn_sfhs \
    --utils-dir "${UTILS_DIR}" \
    --field EDFN \
    --phot-type 2fwhm_aper \
    --sfh-files "${SFH_FILE}" \
    --output "${OUTPUT}" \
    --n "${N_OBJECTS}" \
    --max-vis-mag "${MAX_VIS_MAG}" \
    --seed 42 \
    --batch-size 512

echo "Bright SFH sample: ${OUTPUT}"
