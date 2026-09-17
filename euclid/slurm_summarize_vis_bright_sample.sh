#!/bin/bash
#SBATCH --job-name=euclid_vis_census
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/vis_census_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/vis_census_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage: sbatch euclid/slurm_summarize_vis_bright_sample.sh [max_vis_mag]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_100k
MAX_VIS_MAG=${1:-22.0}
TAG=${MAX_VIS_MAG/./p}
OUTPUT=${BASE}/bright_samples/vis_lt_${TAG}.fits

cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.vis_selection \
    --dataset "${BASE}/sfh_clip_100k.h5" \
    --catalog "${BASE}/sfh_edfn100k/catalog_sfh_100k.fits" \
    --stamp-root "${BASE}/zoobot_stamps_rmax" \
    --limits 20.5 21.0 21.5 22.0 22.5 \
    --max-vis-mag "${MAX_VIS_MAG}" \
    --output "${OUTPUT}" \
    --overwrite

echo "Bright-sample ID catalog: ${OUTPUT}"
