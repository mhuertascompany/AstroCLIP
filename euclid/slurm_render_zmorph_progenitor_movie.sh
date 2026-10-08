#!/bin/bash
#SBATCH --job-name=prog_zmorph
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/prog_zmorph_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/prog_zmorph_%j.err
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

# Build a progenitor track in aligned SFH space, snap dense epochs to real SFH
# states, retrieve the nearest aligned image embedding for every state, and use
# the z_morph-conditioned diffusion model to render the movie.
# Optional positional arguments:
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
STAMPS=${BASE}/zoobot_stamps_rmax/VIS
SFH_CACHE=${BASE}/diffusion_conditions_ae_adjacency_no_edgeon.npz
MORPH_CACHE=${BASE}/diffusion_conditions_zmorph_ae_adjacency_no_edgeon.npz
REFERENCE_UMAP=${BASE}/explorer_ae_adjacency_edgeon_ablation/edgeon_filtered/euclid_clip_umap_diagnostics.npz
SFH_UMAP=${BASE}/diffusion_conditions_ae_adjacency_no_edgeon_explorer_anchored_sfh_umap.npz
DIFFUSION=${BASE}/pixel_diffusion_zmorph_ae_adjacency_no_edgeon_conditioned_full

DESCENDANT_ID=${1:-2701130960681498535}
OUTPUT=${2:-${BASE}/progenitor_zmorph_full_${DESCENDANT_ID}_${SLURM_JOB_ID}}
FRAMES=${3:-120}
NOISE_SEED=${4:-42000}
TRACK=${OUTPUT}/full_sample_track

for path in \
    "${FULL_BUNDLE}/euclid_explorer.h5" "${FULL_ARCHIVE}" "${DATASET}" \
    "${SFH_CACHE}" "${MORPH_CACHE}" "${REFERENCE_UMAP}" \
    "${DIFFUSION}/runtime.json"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMPS}" ]] || { echo "Missing canonical VIS stamps: ${STAMPS}" >&2; exit 2; }
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
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${OUTPUT}"

if [[ ! -f "${SFH_UMAP}" ]]; then
    python -u -m euclid.build_condition_umap \
        --conditions "${SFH_CACHE}" \
        --output "${SFH_UMAP}" \
        --reference-archive "${REFERENCE_UMAP}" \
        --neighbors 15 --min-dist 0.1 --seed 42
fi

python -u -m euclid.progenitor_analogues \
    --bundle "${FULL_BUNDLE}" \
    --archive "${FULL_ARCHIVE}" \
    --stamps "${STAMPS}" \
    --track-umap "${SFH_UMAP}" \
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
    --condition-cache "${SFH_CACHE}" \
    --generation-condition-cache "${MORPH_CACHE}" \
    --pixel-checkpoint "${PIXEL_CHECKPOINT}" \
    --sfh-dataset "${DATASET}" \
    --condition-umap "${SFH_UMAP}" \
    --real-stamps "${STAMPS}" \
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
echo "SFH-track → aligned-morphology movie: ${OUTPUT}"
echo "Download: ${OUTPUT}.tar.gz"
