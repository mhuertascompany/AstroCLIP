#!/bin/bash
#SBATCH --job-name=bright_full_umap
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/full_umap_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/full_umap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=pscomp
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TRAINING=${BASE}/training_bright_frozen_mlp
INPUT=${1:-${TRAINING}/full_sample_explorer}
OUTPUT=${2:-${TRAINING}/full_sample_explorer_bundle}
DATASET=${BASE}/sfh_clip_150k.h5
STAMPS=${BASE}/zoobot_stamps_rmax
EMBEDDINGS=${INPUT}/full_embeddings.npz
DIAGNOSTIC=${INPUT}/euclid_clip_full_umap_diagnostics.pdf

[[ -f "${EMBEDDINGS}" ]] || { echo "Missing embeddings: ${EMBEDDINGS}" >&2; exit 2; }
[[ -f "${DATASET}" ]] || { echo "Missing dataset: ${DATASET}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "Output already exists: ${OUTPUT}" >&2; exit 2; }

export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

python -u -m euclid.umap_zoobot_clip \
  --embeddings "${EMBEDDINGS}" \
  --dataset "${DATASET}" \
  --output "${DIAGNOSTIC}" \
  --npz-output "${INPUT}/euclid_clip_full_umap_diagnostics.npz" \
  --run-label "bright frozen MLP; all stamp-paired galaxies" \
  --skip-shared-manifold \
  --seed 42

python -u -m euclid.export_explorer_bundle \
  --dataset "${DATASET}" \
  --stamp-root "${STAMPS}" \
  --umap "${INPUT}/euclid_clip_full_umap_diagnostics.npz" \
  --output-dir "${OUTPUT}" \
  --band VIS

tar -C "$(dirname "${OUTPUT}")" -cf "${OUTPUT}.tar" "$(basename "${OUTPUT}")"
echo "Download: ${OUTPUT}.tar"
