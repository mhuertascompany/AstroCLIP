#!/bin/bash
#SBATCH --job-name=edfn_sfh_100k
#SBATCH --output=/n03data/huertas/python/AstroCLIP/euclid/sample_edfn_sfhs_100k_%j.out
#SBATCH --error=/n03data/huertas/python/AstroCLIP/euclid/sample_edfn_sfhs_100k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# Submit from the repository root on candide:
#   sbatch euclid/slurm_sample_edfn_sfhs_100k.sh
# Additional sampler arguments can follow the script name.

set -eo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
set -u

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
SFH_FILE=/n17data/wozny/These/science_DR1/SFHs/ready_to_use_sfhs/EDFN_2fwhm_aper.h5
UTILS_DIR=/home/wozny/jobs/These/DR1_science/SFH/utils

cd "${REPO_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# The 100k sample is separate from the 10k pilot. The output directory may be
# new or empty, but existing sample files are never overwritten. All SFH
# realizations and the photometric columns required for image matching are kept.
python -u -m euclid.sample_edfn_sfhs \
    --utils-dir  "${UTILS_DIR}" \
    --field      EDFN \
    --phot-type  2fwhm_aper \
    --sfh-files  "${SFH_FILE}" \
    --output     "${OUTPUT_DIR}" \
    --n          100000 \
    --seed       42 \
    --batch-size 512 \
    "$@"
