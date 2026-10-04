#!/bin/bash
#SBATCH --job-name=gtt_vs_zoo_aligned
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/export_gtt_vs_zoo_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/export_gtt_vs_zoo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Evaluate GalaxyTikTok and ZooBot AE-adjacency checkpoints on the exact same
# held-out IDs, fit aligned UMAPs, and package one explorer.
# Usage: sbatch $0 [GALACTIKTOK_CKPT] [ZOOBOT_CKPT] [OUTPUT_DIR] [GALACTIKTOK_ROOT]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

REPO=/n03data/huertas/python/AstroCLIP
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
GTT_TRAINING=${BASE}/training_bright_galactiktok_ae_adjacency
ZOO_TRAINING=${BASE}/training_bright_frozen_unrestricted_ae_fns
# Use last.ckpt deliberately. Early best checkpoints can precede activation of
# the adjacency term after its three-epoch warm-up.
GTT_CKPT=${1:-${GTT_TRAINING}/checkpoints/last.ckpt}
ZOO_CKPT=${2:-${ZOO_TRAINING}/checkpoints/last.ckpt}
OUTPUT=${3:-${BASE}/explorer_galactiktok_vs_zoobot_ae_adjacency}
GALACTIKTOK_ROOT=${4:-/n03data/huertas/python/galactiktok/galactiktok}

DATASET=${BASE}/sfh_clip_150k.h5
FITS_ROOT=${BASE}/cutouts_run
IMAGE_STATS=${BASE}/pretraining_galactiktok_vis/image_stats.json
JPEG_STAMPS=${BASE}/zoobot_stamps_rmax
GTT_SPLIT=${GTT_TRAINING}/pair_split.npz
COMMON_SPLIT=${ZOO_TRAINING}/pair_split.npz
GTT_EVAL=${OUTPUT}/galactiktok
ZOO_EVAL=${OUTPUT}/zoobot
BUNDLE=${OUTPUT}/explorer_bundle

for path in "${GTT_CKPT}" "${ZOO_CKPT}" "${DATASET}" "${IMAGE_STATS}" "${GTT_SPLIT}" "${COMMON_SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${FITS_ROOT}/cutouts/VIS" ]] || { echo "Missing VIS FITS: ${FITS_ROOT}" >&2; exit 2; }
[[ -d "${JPEG_STAMPS}/VIS" ]] || { echo "Missing VIS JPEGs: ${JPEG_STAMPS}" >&2; exit 2; }
[[ -d "${GALACTIKTOK_ROOT}/src/galactiktok" ]] || { echo "Missing GalaxyTikTok checkout: ${GALACTIKTOK_ROOT}" >&2; exit 2; }

export PYTHONPATH=${GALACTIKTOK_ROOT}/src:${PYTHONPATH:-}
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "${GTT_EVAL}" "${ZOO_EVAL}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
cd "${REPO}"

# GalaxyTikTok reads native FITS but was eligibility-filtered to the ZooBot
# JPEG population. Require exact train and validation equality before comparing.
python - "${GTT_SPLIT}" "${COMMON_SPLIT}" <<'PY'
import sys
import numpy as np
gtt = np.load(sys.argv[1])
common = np.load(sys.argv[2])
same_train = np.array_equal(common['train_ids'], gtt['train_ids'])
same_val = np.array_equal(common['val_ids'], gtt['val_ids'])
if not same_train or not same_val:
    raise SystemExit(
        'GalaxyTikTok and ZooBot do not use identical train/validation IDs: '
        f'same_train={same_train}, same_val={same_val}'
    )
print(
    f"Verified identical samples: train={len(common['train_ids']):,}, "
    f"validation={len(common['val_ids']):,}"
)
PY

evaluate_galactiktok() {
    if [[ -f "${GTT_EVAL}/validation_embeddings.npz" ]]; then
        echo "Reusing GalaxyTikTok evaluation in ${GTT_EVAL}"
        return
    fi
    python -u -m euclid.evaluate_zoobot_clip \
        --checkpoint "${GTT_CKPT}" \
        --dataset "${DATASET}" \
        --stamp-root "${FITS_ROOT}" \
        --image-format fits \
        --image-stats "${IMAGE_STATS}" \
        --image-size 96 \
        --split "${COMMON_SPLIT}" \
        --output-dir "${GTT_EVAL}" \
        --band VIS \
        --batch-size 128 \
        --num-workers "${SLURM_CPUS_PER_TASK}" \
        --chunk-size 512 \
        --n-permutations 200 \
        --permutation-size 5000 \
        --shape-subset 2000 \
        --posterior-subset 1000 \
        --posterior-draws 5 \
        --device cuda
}

evaluate_zoobot() {
    if [[ -f "${ZOO_EVAL}/validation_embeddings.npz" ]]; then
        echo "Reusing ZooBot evaluation in ${ZOO_EVAL}"
        return
    fi
    python -u -m euclid.evaluate_zoobot_clip \
        --checkpoint "${ZOO_CKPT}" \
        --dataset "${DATASET}" \
        --stamp-root "${JPEG_STAMPS}" \
        --split "${COMMON_SPLIT}" \
        --output-dir "${ZOO_EVAL}" \
        --band VIS \
        --batch-size 128 \
        --num-workers "${SLURM_CPUS_PER_TASK}" \
        --chunk-size 512 \
        --n-permutations 200 \
        --permutation-size 5000 \
        --shape-subset 2000 \
        --posterior-subset 1000 \
        --posterior-draws 5 \
        --device cuda
}

make_umap() {
    local evaluation=$1
    local label=$2
    if [[ -f "${evaluation}/euclid_clip_umap_diagnostics.npz" ]]; then
        echo "Reusing UMAP in ${evaluation}"
        return
    fi
    python -u -m euclid.umap_zoobot_clip \
        --embeddings "${evaluation}/validation_embeddings.npz" \
        --dataset "${DATASET}" \
        --per-object "${evaluation}/per_object.csv" \
        --output "${evaluation}/euclid_clip_umap_diagnostics.pdf" \
        --run-label "${label}" \
        --seed 42
}

evaluate_galactiktok
evaluate_zoobot
make_umap "${GTT_EVAL}" "GalaxyTikTok + AE-adjacency CLIP"
make_umap "${ZOO_EVAL}" "Frozen ZooBot + AE-adjacency CLIP"

if [[ ! -f "${BUNDLE}/manifest.json" ]]; then
    python -u -m euclid.export_explorer_bundle \
        --dataset "${DATASET}" \
        --stamp-root "${JPEG_STAMPS}" \
        --umap "${GTT_EVAL}/euclid_clip_umap_diagnostics.npz" \
        --umap "${ZOO_EVAL}/euclid_clip_umap_diagnostics.npz" \
        --output-dir "${BUNDLE}" \
        --band VIS
fi

tar -C "${OUTPUT}" -cf "${OUTPUT}/explorer_bundle.tar" explorer_bundle
echo "GalaxyTikTok checkpoint: ${GTT_CKPT}"
echo "ZooBot checkpoint: ${ZOO_CKPT}"
echo "Comparison bundle: ${OUTPUT}/explorer_bundle.tar"
