#!/bin/bash
#SBATCH --job-name=euclid_bright_morph
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/restore_morphology_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/restore_morphology_%j.err
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_restore_morphology_metadata_bright.sh \
#       [sfh_clip.h5] [mer_zoobot_morphology.fits]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
DATASET=${1:-${BASE}/sfh_clip_150k.h5}
MORPHOLOGY_CATALOG=${2:-${BASE}/mer_zoobot_morphology_deep.fits}

cd /n03data/huertas/python/AstroCLIP

python -u -m euclid.restore_morphology_metadata \
    --dataset "${DATASET}" \
    --catalog "${MORPHOLOGY_CATALOG}" \
    --allow-missing
