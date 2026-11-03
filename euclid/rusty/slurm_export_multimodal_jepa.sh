#!/bin/bash
#SBATCH --job-name=export_jepa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/export_jepa_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/export_jepa_%j.err

set -euo pipefail

module load python
source "${HOME}/.sfh_euclid/bin/activate"

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
RUN=${BASE}/runs/multimodal_jepa_full_v1
IDS=${IDS:-${RUN}/explorer_galaxy_ids.npy}
PROPERTIES=${PROPERTIES:-${RUN}/explorer_properties.npz}
OUTPUT=${OUTPUT:-${RUN}/multimodal_jepa_umap_diagnostics.npz}

cd "${REPO}"
for path in \
    "${RUN}/checkpoints/last.ckpt" \
    "${RUN}/data_transform.npz" \
    "${ARCHIVE}/sfh_clip_150k.h5" \
    "${ARCHIVE}/packed_vis_64/vis_images.npy" \
    "${IDS}" \
    "${PROPERTIES}"
do
    [[ -e "${path}" ]] || { echo "Missing ${path}" >&2; exit 2; }
done

echo "Repository commit: $(git rev-parse HEAD)"
echo "Checkpoint: ${RUN}/checkpoints/last.ckpt"
echo "Output: ${OUTPUT}"

python -u -m euclid.export_multimodal_jepa_embeddings \
    --checkpoint "${RUN}/checkpoints/last.ckpt" \
    --dataset "${ARCHIVE}/sfh_clip_150k.h5" \
    --packed-vis "${ARCHIVE}/packed_vis_64" \
    --data-transform "${RUN}/data_transform.npz" \
    --ids "${IDS}" \
    --properties "${PROPERTIES}" \
    --output "${OUTPUT}" \
    --device cuda \
    --batch-size "${BATCH_SIZE:-128}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --n-neighbors "${N_NEIGHBORS:-15}" \
    --min-dist "${MIN_DIST:-0.1}" \
    --seed 42
