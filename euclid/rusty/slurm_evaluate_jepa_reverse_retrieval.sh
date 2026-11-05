#!/bin/bash
#SBATCH --job-name=jepa_vis2sfh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/jepa_vis2sfh_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/jepa_vis2sfh_%j.err

set -euo pipefail

module load python
source "${HOME}/.sfh_euclid/bin/activate"

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
RUN=${BASE}/runs/multimodal_jepa_full_v1
OUTPUT=${OUTPUT:-${RUN}/vis_to_sfh_csls}

cd "${REPO}"
for path in \
    "${RUN}/checkpoints/last.ckpt" \
    "${RUN}/data_transform.npz" \
    "${ARCHIVE}/sfh_clip_150k.h5" \
    "${ARCHIVE}/packed_vis_64/vis_images.npy" \
    "${ARCHIVE}/packed_vis_64/sfh_clr.npy"
do
    [[ -e "${path}" ]] || { echo "Missing ${path}" >&2; exit 2; }
done

echo "Repository commit: $(git rev-parse HEAD)"
echo "Checkpoint: ${RUN}/checkpoints/last.ckpt"
echo "Output: ${OUTPUT}"

python -u -m euclid.evaluate_multimodal_jepa_reverse_retrieval \
    --checkpoint "${RUN}/checkpoints/last.ckpt" \
    --dataset "${ARCHIVE}/sfh_clip_150k.h5" \
    --packed-vis "${ARCHIVE}/packed_vis_64" \
    --data-transform "${RUN}/data_transform.npz" \
    --output "${OUTPUT}" \
    --split val \
    --device cuda \
    --batch-size "${BATCH_SIZE:-128}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --gallery-chunk "${GALLERY_CHUNK:-1024}" \
    --query-chunk "${QUERY_CHUNK:-512}" \
    --top-k "${TOP_K:-100}" \
    --k 10 --k 20 --k 50 --k 100 \
    --seed 42
