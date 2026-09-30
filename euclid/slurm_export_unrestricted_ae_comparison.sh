#!/bin/bash
#SBATCH --job-name=free_vs_ae_export
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/export_free_vs_ae_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/export_free_vs_ae_%j.err
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

# Evaluate the unrestricted-adapter and post-warmup AE-adjacency checkpoints,
# fit their UMAPs, and create one comparison explorer bundle. The AE run uses
# last.ckpt deliberately: its nominal best checkpoints precede activation of
# the adjacency term.
#
# Optional positional arguments:
#   1 unrestricted checkpoint
#   2 AE-adjacency checkpoint
#   3 comparison output directory

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

REPO=/n03data/huertas/python/AstroCLIP
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
FREE_TRAINING=${BASE}/training_bright_frozen_unrestricted_adapters
AE_TRAINING=${BASE}/training_bright_frozen_unrestricted_ae_fns

FREE_CKPT=${1:-${FREE_TRAINING}/checkpoints/euclid_bright_unrestricted_adapters_150k-epoch=024-val_loss=4.2556.ckpt}
AE_CKPT=${2:-${AE_TRAINING}/checkpoints/last.ckpt}
OUTPUT=${3:-${BASE}/explorer_unrestricted_vs_ae_adjacency}

DATASET=${BASE}/sfh_clip_150k.h5
STAMPS=${BASE}/zoobot_stamps_rmax
FREE_SPLIT=${FREE_TRAINING}/pair_split.npz
AE_SPLIT=${AE_TRAINING}/pair_split.npz
FREE_EVAL=${OUTPUT}/unrestricted
AE_EVAL=${OUTPUT}/ae_adjacency
BUNDLE=${OUTPUT}/explorer_bundle

for path in "${FREE_CKPT}" "${AE_CKPT}" "${DATASET}" "${FREE_SPLIT}" "${AE_SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMPS}/VIS" ]] || { echo "Missing VIS stamps: ${STAMPS}/VIS" >&2; exit 2; }

mkdir -p "${FREE_EVAL}" "${AE_EVAL}"
cd "${REPO}"

evaluate_checkpoint() {
    local checkpoint=$1
    local split=$2
    local destination=$3

    if [[ -f "${destination}/validation_embeddings.npz" && -f "${destination}/per_object.csv" ]]; then
        echo "Reusing evaluation products in ${destination}"
        return
    fi
    python -u -m euclid.evaluate_zoobot_clip \
        --checkpoint "${checkpoint}" \
        --dataset "${DATASET}" \
        --stamp-root "${STAMPS}" \
        --split "${split}" \
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
    local checkpoint=$1
    local evaluation=$2
    local label=$3

    if [[ -f "${evaluation}/euclid_clip_umap_diagnostics.npz" ]]; then
        echo "Reusing UMAP archive in ${evaluation}"
        return
    fi
    python -u -m euclid.umap_zoobot_clip \
        --embeddings "${evaluation}/validation_embeddings.npz" \
        --dataset "${DATASET}" \
        --per-object "${evaluation}/per_object.csv" \
        --output "${evaluation}/euclid_clip_umap_diagnostics.pdf" \
        --run-label "${label}: $(basename "${checkpoint}" .ckpt)" \
        --seed 42
}

export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

evaluate_checkpoint "${FREE_CKPT}" "${FREE_SPLIT}" "${FREE_EVAL}"
evaluate_checkpoint "${AE_CKPT}" "${AE_SPLIT}" "${AE_EVAL}"

make_umap "${FREE_CKPT}" "${FREE_EVAL}" "unrestricted adapters"
make_umap "${AE_CKPT}" "${AE_EVAL}" "AE adjacency, post-warmup final"

if [[ -f "${BUNDLE}/manifest.json" ]]; then
    echo "Comparison bundle already exists: ${BUNDLE}"
else
    python -u -m euclid.export_explorer_bundle \
        --dataset "${DATASET}" \
        --stamp-root "${STAMPS}" \
        --umap "${FREE_EVAL}/euclid_clip_umap_diagnostics.npz" \
        --umap "${AE_EVAL}/euclid_clip_umap_diagnostics.npz" \
        --output-dir "${BUNDLE}" \
        --band VIS
fi

ARCHIVE=${OUTPUT}/explorer_bundle.tar
tar -C "${OUTPUT}" -cf "${ARCHIVE}" explorer_bundle

echo "Unrestricted checkpoint: ${FREE_CKPT}"
echo "AE-adjacency checkpoint: ${AE_CKPT}"
echo "Comparison bundle: ${BUNDLE}"
echo "Download archive: ${ARCHIVE}"
