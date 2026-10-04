#!/bin/bash
#SBATCH --job-name=compare_image_encoders
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/compare_image_encoders_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/compare_image_encoders_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Frozen GalaxyTikTok versus frozen ZooBot on the exact IDs already present in
# the ZooBot 30k diagnostic archive.
# Usage: sbatch $0 [tokenizer] [zoobot_archive] [output_dir] [galactiktok_root]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TOKENIZER=${1:-${BASE}/pretraining_galactiktok_vis/tokenizer}
DEFAULT_ZOOBOT=${BASE}/zoobot_image_embedding_30k_with_morphology_verified/zoobot_image_umap.npz
if [[ ! -f "${DEFAULT_ZOOBOT}" ]]; then
    DEFAULT_ZOOBOT=${BASE}/zoobot_image_embedding_30k/zoobot_image_umap.npz
fi
ZOOBOT_ARCHIVE=${2:-${DEFAULT_ZOOBOT}}
OUTPUT=${3:-${BASE}/galactiktok_vs_zoobot_30k}
GALACTIKTOK_ROOT=${4:-/n03data/huertas/python/galactiktok/galactiktok}
FITS_ROOT=${BASE}/cutouts_run
IMAGE_STATS=${BASE}/pretraining_galactiktok_vis/image_stats.json

[[ -f "${TOKENIZER}/config.json" ]] || { echo "Missing tokenizer: ${TOKENIZER}" >&2; exit 2; }
[[ -f "${ZOOBOT_ARCHIVE}" ]] || { echo "Missing ZooBot archive: ${ZOOBOT_ARCHIVE}" >&2; exit 2; }
[[ -d "${FITS_ROOT}/cutouts/VIS" ]] || { echo "Missing VIS FITS: ${FITS_ROOT}" >&2; exit 2; }
[[ -f "${IMAGE_STATS}" ]] || { echo "Missing image statistics: ${IMAGE_STATS}" >&2; exit 2; }
[[ -d "${GALACTIKTOK_ROOT}/src/galactiktok" ]] || { echo "Missing GalaxyTikTok checkout: ${GALACTIKTOK_ROOT}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "Output already exists: ${OUTPUT}" >&2; exit 2; }

export PYTHONPATH=${GALACTIKTOK_ROOT}/src:${PYTHONPATH:-}
echo "GalaxyTikTok commit: $(git -C "${GALACTIKTOK_ROOT}" rev-parse HEAD)"

python -u -m euclid.compare_frozen_image_encoders \
    --tokenizer "${TOKENIZER}" \
    --fits-root "${FITS_ROOT}" \
    --image-stats "${IMAGE_STATS}" \
    --zoobot-archive "${ZOOBOT_ARCHIVE}" \
    --output-dir "${OUTPUT}" \
    --batch-size 256 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --seed 42 \
    --device cuda

tar -C "${OUTPUT}" -czf "${OUTPUT}/galactiktok_vs_zoobot_compact.tar.gz" \
    galactiktok_image_umap.npz \
    zoobot_image_umap_compact.npz \
    galactiktok_vs_zoobot_umap.pdf \
    manifest.json
