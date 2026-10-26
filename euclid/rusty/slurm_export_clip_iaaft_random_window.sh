#!/bin/bash
#SBATCH --job-name=export_iaaft_umap
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/export_iaaft_umap_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/export_iaaft_umap_%j.err

set -euo pipefail

module load python
source "${RUSTY_VENV:-${HOME}/.sfh_euclid}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE="${BASE}/archives/edfn_vislt22p0_150000"
RUN="${BASE}/runs/iaaft_random_window/clip_ae_adjacency_no_edgeon_full"
CHECKPOINT="${RUN}/checkpoints/euclid_clip_iaaft_random_window_150k-epoch=002-val_loss=4.4243.ckpt"
DATASET="${ARCHIVE}/sfh_clip_150k.h5"
SFH_OVERRIDE="${BASE}/runs/iaaft_random_window/sfh_iaaft_window_random_window_past90_150k.h5"
STAMPS="${ARCHIVE}/zoobot_stamps_rmax"
SPLIT="${RUN}/pair_split.npz"
OUTPUT="${BASE}/runs/iaaft_random_window/explorer_comparison/random_window_eval"

export HF_HOME="${BASE}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${BASE}/cache/torch"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"

cd "${REPO}"
for path in "${CHECKPOINT}" "${DATASET}" "${SFH_OVERRIDE}" "${SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMPS}/VIS" ]] || { echo "Missing VIS stamps: ${STAMPS}/VIS" >&2; exit 2; }
mkdir -p "${OUTPUT}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

if [[ ! -f "${OUTPUT}/validation_embeddings.npz" ]]; then
    python -u -m euclid.evaluate_zoobot_clip \
        --checkpoint "${CHECKPOINT}" \
        --dataset "${DATASET}" \
        --sfh-override "${SFH_OVERRIDE}" \
        --stamp-root "${STAMPS}" \
        --split "${SPLIT}" \
        --output-dir "${OUTPUT}" \
        --band VIS \
        --batch-size 128 \
        --num-workers "${SLURM_CPUS_PER_TASK}" \
        --chunk-size 512 \
        --n-permutations 200 \
        --permutation-size 5000 \
        --shape-subset 2000 \
        --skip-posterior \
        --device cuda
else
    echo "Reusing ${OUTPUT}/validation_embeddings.npz"
fi

if [[ ! -f "${OUTPUT}/random_window_euclid_clip_umap_diagnostics.npz" ]]; then
    python -u -m euclid.umap_zoobot_clip \
        --embeddings "${OUTPUT}/validation_embeddings.npz" \
        --dataset "${DATASET}" \
        --per-object "${OUTPUT}/per_object.csv" \
        --output "${OUTPUT}/random_window_euclid_clip_umap_diagnostics.pdf" \
        --run-label "IAAFT random past window: AE adjacency, edge-on filtered" \
        --seed 42
else
    echo "Reusing ${OUTPUT}/random_window_euclid_clip_umap_diagnostics.npz"
fi

echo "Checkpoint: ${CHECKPOINT}"
echo "UMAP archive: ${OUTPUT}/random_window_euclid_clip_umap_diagnostics.npz"
