#!/bin/bash
#SBATCH --job-name=euclid_zoobot_bundle
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/zoobot_bundle_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/zoobot_bundle_%j.err
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_export_zoobot_image_explorer.sh \
#       [sfh_clip.h5] [stamp_root] [umap.npz] [output_dir]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
DATASET=${1:-${BASE}/sfh_clip_150k.h5}
STAMP_ROOT=${2:-${BASE}/zoobot_stamps_rmax}
UMAP=${3:-${BASE}/zoobot_image_embedding_30k/zoobot_image_umap.npz}
OUTPUT=${4:-${BASE}/explorer_zoobot_image_30k}

cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.export_explorer_bundle \
    --dataset "${DATASET}" \
    --stamp-root "${STAMP_ROOT}" \
    --umap "${UMAP}" \
    --output-dir "${OUTPUT}" \
    --band VIS
