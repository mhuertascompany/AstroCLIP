#!/bin/bash
#SBATCH --job-name=jepa_csls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/jepa_csls_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/jepa_csls_%j.err

set -euo pipefail

module load python
source "${HOME}/.sfh_euclid/bin/activate"

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
RUN=${BASE}/runs/multimodal_jepa_full_v1
RETRIEVAL=${RETRIEVAL:-${RUN}/sfh_to_vis_retrieval_h100_final/retrieval_results.npz}
OUTPUT=${OUTPUT:-${RUN}/sfh_to_vis_csls}

cd "${REPO}"
for path in \
    "${RETRIEVAL}" \
    "${RUN}/explorer_properties.npz" \
    "${ARCHIVE}/packed_vis_64/vis_images.npy" \
    "${ARCHIVE}/packed_vis_64/sfh_clr.npy"
do
    [[ -e "${path}" ]] || { echo "Missing ${path}" >&2; exit 2; }
done

echo "Repository commit: $(git rev-parse HEAD)"
echo "Retrieval input: ${RETRIEVAL}"
echo "Output: ${OUTPUT}"

python -u -m euclid.evaluate_jepa_csls \
    --retrieval "${RETRIEVAL}" \
    --packed-vis "${ARCHIVE}/packed_vis_64" \
    --properties "${RUN}/explorer_properties.npz" \
    --output "${OUTPUT}" \
    --k 10 --k 20 --k 50 --k 100 \
    --device cuda \
    --gallery-chunk "${GALLERY_CHUNK:-1024}" \
    --query-chunk "${QUERY_CHUNK:-512}" \
    --top-k 100 \
    --morphology-queries 1000 \
    --morphology-k 20 \
    --examples 8 \
    --seed 42
