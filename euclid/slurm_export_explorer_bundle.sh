#!/bin/bash
#SBATCH --job-name=euclid_explorer_export
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/explorer_export_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/explorer_export_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_export_explorer_bundle.sh OUTPUT_DIR UMAP_NPZ [UMAP_NPZ ...]

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 OUTPUT_DIR UMAP_NPZ [UMAP_NPZ ...]" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_100k
OUTPUT_DIR=$1
shift

UMAP_ARGS=()
for path in "$@"; do
    UMAP_ARGS+=(--umap "${path}")
done

cd "${REPO_DIR}"
python -u -m euclid.export_explorer_bundle \
    --dataset "${BASE_DIR}/sfh_clip_100k.h5" \
    --stamp-root "${BASE_DIR}/zoobot_stamps_rmax" \
    --output-dir "${OUTPUT_DIR}" \
    --band VIS \
    "${UMAP_ARGS[@]}"
