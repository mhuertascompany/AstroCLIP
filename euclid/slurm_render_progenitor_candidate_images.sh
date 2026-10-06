#!/bin/bash
#SBATCH --job-name=progenitor_candidates
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/progenitor_candidates_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/progenitor_candidates_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
CANDIDATES=${1:-euclid/movie_tracks/ae_adjacency_descendant_2701130960681498535.csv}
DESCENDANT_ID=${2:-2701130960681498535}
OUTPUT=${3:-${BASE}/progenitor_candidate_images_${DESCENDANT_ID}_${SLURM_JOB_ID}}
NOISE_SEED=${4:-314159}
CONDITION_CACHE=${BASE}/diffusion_conditions_aligned_best.npz
PIXEL_CHECKPOINT=${BASE}/pixel_diffusion_aligned_full/checkpoints/last.ckpt

for path in "${CANDIDATES}" "${CONDITION_CACHE}" "${PIXEL_CHECKPOINT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Output already exists: ${OUTPUT}" >&2; exit 2; }
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"

python -u -m euclid.render_progenitor_candidate_images \
    --candidates "${CANDIDATES}" \
    --descendant-id "${DESCENDANT_ID}" \
    --condition-cache "${CONDITION_CACHE}" \
    --pixel-checkpoint "${PIXEL_CHECKPOINT}" \
    --output "${OUTPUT}" \
    --n-analogues 5 \
    --batch-size 8 \
    --steps 100 \
    --guidance 1 \
    --noise-seed "${NOISE_SEED}" \
    --device cuda

echo "Discrete candidate images and diagnostics: ${OUTPUT}"
