#!/bin/bash
#SBATCH --job-name=inspect_vis_fits
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/inspect_vis_fits_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/inspect_vis_fits_%j.err
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

FITS_ROOT=${1:-/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/cutouts_run}
LIMIT=${2:-1000}

python -u -m euclid.inspect_vis_cutouts \
    --fits-root "${FITS_ROOT}" \
    --limit "${LIMIT}"
