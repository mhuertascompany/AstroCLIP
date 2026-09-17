#!/bin/bash
#SBATCH --job-name=edfn_bright_census
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_bright_census_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_bright_census_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
SFH_FILE=/n17data/wozny/These/science_DR1/SFHs/ready_to_use_sfhs/EDFN_2fwhm_aper.h5
UTILS_DIR=/home/wozny/jobs/These/DR1_science/SFH/utils

cd "${REPO_DIR}"
python -u -m euclid.sample_edfn_sfhs \
    --utils-dir "${UTILS_DIR}" \
    --field EDFN \
    --phot-type 2fwhm_aper \
    --sfh-files "${SFH_FILE}" \
    --report-vis-limits 20.5 21.0 21.5 22.0 22.5 23.0 23.5 \
    --dry-run
