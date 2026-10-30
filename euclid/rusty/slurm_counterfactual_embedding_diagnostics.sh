#!/bin/bash
#SBATCH --job-name=counterflow_umap
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/counterflow_umap_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/counterflow_umap_%j.err

set -euo pipefail

module load python cuda/12.5.1 cudnn/8.9.7.29-12
source "${RUSTY_VENV:-${HOME}/.sfh_euclid}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR=/tmp/mhuertas-matplotlib-${SLURM_JOB_ID}

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
RUN=${BASE}/runs/counterfactual_raw_vis_sfh_full_monitored_v1
CHECKPOINT=${1:-${RUN}/checkpoints/counterfactual-epoch=039-val_loss=0.24470.ckpt}
OUTPUT=${2:-${RUN}/embedding_diagnostics_epoch039}

cd "${REPO}"
for path in \
    "${CHECKPOINT}" \
    "${ARCHIVE}/sfh_clip_150k.h5" \
    "${RUN}/matched_donors.npz"; do
    [[ -f "${path}" ]] || { echo "Missing ${path}" >&2; exit 2; }
done
[[ -d "${ARCHIVE}/zoobot_stamps_rmax/VIS" ]] || {
    echo "Missing VIS stamps" >&2; exit 2;
}
[[ ! -e "${OUTPUT}" ]] || { echo "Output exists: ${OUTPUT}" >&2; exit 2; }

echo "Repository commit: $(git rev-parse HEAD)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT}"

python -u -m euclid.export_counterfactual_embeddings \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${ARCHIVE}/sfh_clip_150k.h5" \
    --stamps "${ARCHIVE}/zoobot_stamps_rmax" \
    --split "${RUN}/matched_donors.npz" \
    --output-dir "${OUTPUT}" \
    --max-objects 10000 \
    --batch-size 128 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --image-size 64 \
    --donors 8 \
    --device cuda

python -u -m euclid.umap_zoobot_clip \
    --embeddings "${OUTPUT}/common_embeddings.npz" \
    --dataset "${ARCHIVE}/sfh_clip_150k.h5" \
    --output "${OUTPUT}/common_embedding_umap.pdf" \
    --npz-output "${OUTPUT}/common_embedding_umap.npz" \
    --run-label "Counterfactual flow epoch 39: common representations"

python -u -m euclid.umap_zoobot_clip \
    --embeddings "${OUTPUT}/prior_embeddings.npz" \
    --dataset "${ARCHIVE}/sfh_clip_150k.h5" \
    --output "${OUTPUT}/prior_embedding_umap.pdf" \
    --npz-output "${OUTPUT}/prior_embedding_umap.npz" \
    --run-label "Counterfactual flow epoch 39: private/prior representations" \
    --skip-shared-manifold

echo "Diagnostics complete: ${OUTPUT}"
