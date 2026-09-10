#!/bin/bash
#SBATCH --job-name=edfn_sfh_10k
#SBATCH --output=/n03data/huertas/python/AstroCLIP/euclid/sample_edfn_sfhs_%j.out
#SBATCH --error=/n03data/huertas/python/AstroCLIP/euclid/sample_edfn_sfhs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# Submit from the repository root on candide:
#   sbatch euclid/slurm_sample_edfn_sfhs.sh
# Additional sampler arguments can follow the script name, for example:
#   sbatch euclid/slurm_sample_edfn_sfhs.sh --output /path/to/new_sample
#   sbatch euclid/slurm_sample_edfn_sfhs.sh --sfh-files /path/to/EDFN.h5

set -eo pipefail

# Same environment as the existing COSMOS-Web preparation jobs.
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
set -u

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/euclid/sfh_clip/edfn_10k
SFH_DIR=/n17data/wozny/These/science_DR1/SFHs/ready_to_use_sfhs
UTILS_DIR=/home/wozny/jobs/These/DR1_science/SFH/utils

cd "${REPO_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# The sampler creates OUTPUT_DIR itself and refuses to overwrite an existing
# sample. Do not mkdir OUTPUT_DIR here. It preserves all SFH realizations and
# exports catalog.fits with the photometric columns needed for image cutouts.
# Extra arguments override defaults (argparse uses the last supplied value).
python -u -m euclid.sample_edfn_sfhs \
    --utils-dir  "${UTILS_DIR}" \
    --field      EDFN \
    --phot-type  2fwhm_aper \
    --sfh-dir    "${SFH_DIR}" \
    --output     "${OUTPUT_DIR}" \
    --n          10000 \
    --seed       42 \
    --batch-size 128 \
    "$@"
