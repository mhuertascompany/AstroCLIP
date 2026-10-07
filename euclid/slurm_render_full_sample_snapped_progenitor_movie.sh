#!/bin/bash
#SBATCH --job-name=progenitor_snap
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/progenitor_snap_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/progenitor_snap_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Recompute the selected descendant's track against the complete bright sample,
# then render dense track points with exact conditions from their nearest real
# full-cache galaxies. Optional positional arguments:
#   1 descendant ID
#   2 output directory
#   3 number of dense movie frames
#   4 base noise seed (frame i uses base+i)

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TRAINING=${BASE}/training_bright_frozen_mlp
FULL_BUNDLE=${TRAINING}/full_sample_explorer_bundle
FULL_ARCHIVE=${TRAINING}/full_sample_explorer/euclid_clip_full_umap_diagnostics.npz
DATASET=${BASE}/sfh_clip_150k.h5
CONDITION_CACHE=${BASE}/diffusion_conditions_ae_adjacency_no_edgeon.npz
DIFFUSION=${BASE}/pixel_diffusion_ae_adjacency_no_edgeon_conditioned_full

DESCENDANT_ID=${1:-2701130960681498535}
OUTPUT=${2:-${BASE}/progenitor_snapped_full_${DESCENDANT_ID}_${SLURM_JOB_ID}}
FRAMES=${3:-120}
NOISE_SEED=${4:-42000}
TRACK=${OUTPUT}/full_sample_track

for path in \
    "${FULL_BUNDLE}/euclid_explorer.h5" "${FULL_ARCHIVE}" "${DATASET}" \
    "${CONDITION_CACHE}" "${DIFFUSION}/runtime.json"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${FULL_BUNDLE}/VIS" ]] || { echo "Missing full-sample stamps: ${FULL_BUNDLE}/VIS" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "Output already exists: ${OUTPUT}" >&2; exit 2; }

PIXEL_CHECKPOINT=$(
python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' \
    "${DIFFUSION}/runtime.json"
)
[[ -f "${PIXEL_CHECKPOINT}" ]] || {
    echo "Best diffusion checkpoint is missing: ${PIXEL_CHECKPOINT}" >&2
    exit 2
}

export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT}"

# Keep extra ranked candidates because the diffusion cache deliberately omits
# the edge-on ablation objects. The renderer takes the best available candidate
# at each anchor, while dense movie points are snapped over the full cache.
python -u -m euclid.progenitor_analogues \
    --bundle "${FULL_BUNDLE}" \
    --archive "${FULL_ARCHIVE}" \
    --no-catalog \
    --descendant-id "${DESCENDANT_ID}" \
    --minimum-descendant-mass 0 \
    --minimum-progenitor-mass 9 \
    --mass-tolerance 0.15 \
    --history-gyr 2 \
    --global-shape-weight 0.5 \
    --n-analogues 20 \
    --output "${TRACK}" \
    --pdf "${OUTPUT}/full_sample_track.pdf"

python -u -m euclid.render_progenitor_morphology_movie \
    --candidates "${TRACK}/analogue_candidates.csv" \
    --descendant-id "${DESCENDANT_ID}" \
    --condition-cache "${CONDITION_CACHE}" \
    --pixel-checkpoint "${PIXEL_CHECKPOINT}" \
    --sfh-dataset "${DATASET}" \
    --output "${OUTPUT}/movie" \
    --n-analogues 1 \
    --frames "${FRAMES}" \
    --batch-size 8 \
    --steps 100 \
    --guidance 1 \
    --noise-seed "${NOISE_SEED}" \
    --snap-to-reference \
    --independent-noise \
    --fps 15 \
    --device cuda

tar -C "$(dirname "${OUTPUT}")" -czf "${OUTPUT}.tar.gz" "$(basename "${OUTPUT}")"
echo "Movie, SFH frames, track diagnostics: ${OUTPUT}"
echo "Download: ${OUTPUT}.tar.gz"
