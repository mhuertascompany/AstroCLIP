#!/bin/bash
#SBATCH --job-name=export_edge_ablation
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/export_edge_ablation_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/export_edge_ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Compare the original and edge-on-filtered ZooBot AE-adjacency models on the
# exact filtered validation set. Optional arguments: original checkpoint,
# filtered checkpoint, output directory.

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

REPO=/n03data/huertas/python/AstroCLIP
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
ORIGINAL_TRAINING=${BASE}/training_bright_frozen_unrestricted_ae_fns
FILTERED_TRAINING=${BASE}/training_bright_frozen_unrestricted_ae_no_edgeon
ORIGINAL_CKPT=${1:-${ORIGINAL_TRAINING}/checkpoints/last.ckpt}
FILTERED_CKPT=${2:-${FILTERED_TRAINING}/checkpoints/last.ckpt}
OUTPUT=${3:-${BASE}/explorer_ae_adjacency_edgeon_ablation}

DATASET=${BASE}/sfh_clip_150k.h5
STAMPS=${BASE}/zoobot_stamps_rmax
ORIGINAL_SPLIT=${ORIGINAL_TRAINING}/pair_split.npz
FILTERED_SPLIT=${FILTERED_TRAINING}/pair_split.npz
ORIGINAL_EVAL=${OUTPUT}/original
FILTERED_EVAL=${OUTPUT}/edgeon_filtered
BUNDLE=${OUTPUT}/explorer_bundle

for path in "${ORIGINAL_CKPT}" "${FILTERED_CKPT}" "${DATASET}" "${ORIGINAL_SPLIT}" "${FILTERED_SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMPS}/VIS" ]] || { echo "Missing VIS stamps: ${STAMPS}/VIS" >&2; exit 2; }

mkdir -p "${ORIGINAL_EVAL}" "${FILTERED_EVAL}"
cd "${REPO}"

python - "${ORIGINAL_SPLIT}" "${FILTERED_SPLIT}" <<'PY'
import sys
import numpy as np
original = np.load(sys.argv[1])
filtered = np.load(sys.argv[2])
train_leak = np.setdiff1d(filtered['train_ids'], original['train_ids'])
val_leak = np.setdiff1d(filtered['val_ids'], original['val_ids'])
cross_leak = np.intersect1d(filtered['val_ids'], original['train_ids'])
if len(train_leak) or len(val_leak) or len(cross_leak):
    raise SystemExit(
        'Filtered split is not a clean subset of the original split: '
        f'train={len(train_leak)}, val={len(val_leak)}, cross={len(cross_leak)}'
    )
excluded = int(np.asarray(filtered['n_edge_on_excluded']))
print(
    f"Verified filtered split: train={len(filtered['train_ids']):,}, "
    f"validation={len(filtered['val_ids']):,}, excluded={excluded:,}"
)
PY

evaluate_checkpoint() {
    local checkpoint=$1
    local destination=$2
    if [[ -f "${destination}/validation_embeddings.npz" ]]; then
        echo "Reusing evaluation in ${destination}"
        return
    fi
    python -u -m euclid.evaluate_zoobot_clip \
        --checkpoint "${checkpoint}" \
        --dataset "${DATASET}" \
        --stamp-root "${STAMPS}" \
        --split "${FILTERED_SPLIT}" \
        --output-dir "${destination}" \
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

export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

evaluate_checkpoint "${ORIGINAL_CKPT}" "${ORIGINAL_EVAL}"
evaluate_checkpoint "${FILTERED_CKPT}" "${FILTERED_EVAL}"
make_umap "${ORIGINAL_EVAL}" "AE adjacency: original training"
make_umap "${FILTERED_EVAL}" "AE adjacency: edge-on intersection removed"

if [[ ! -f "${BUNDLE}/manifest.json" ]]; then
    python -u -m euclid.export_explorer_bundle \
        --dataset "${DATASET}" \
        --stamp-root "${STAMPS}" \
        --umap "${ORIGINAL_EVAL}/euclid_clip_umap_diagnostics.npz" \
        --umap "${FILTERED_EVAL}/euclid_clip_umap_diagnostics.npz" \
        --output-dir "${BUNDLE}" \
        --band VIS
fi

tar -C "${OUTPUT}" -cf "${OUTPUT}/explorer_bundle.tar" explorer_bundle
echo "Original checkpoint: ${ORIGINAL_CKPT}"
echo "Filtered checkpoint: ${FILTERED_CKPT}"
echo "Comparison bundle: ${OUTPUT}/explorer_bundle.tar"
