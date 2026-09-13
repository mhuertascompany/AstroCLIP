#!/bin/bash
#SBATCH --job-name=euclid_sfh_clip
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/preprocess_sfhs_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/preprocess_sfhs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
SAMPLE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
INPUT=${1:-${SAMPLE_DIR}/sfh_000.h5}
OUTPUT=${2:-${SAMPLE_DIR}/sfh_clip_100k.h5}

cd "${REPO_DIR}"

python -m euclid.preprocess_sfhs \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --n-bins 250 \
    --batch-size 256
