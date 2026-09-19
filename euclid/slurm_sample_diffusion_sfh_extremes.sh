#!/bin/bash
#SBATCH --job-name=diff_sfh_extremes
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/diff_extremes_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/diff_extremes_%j.err
set -euo pipefail
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
CHECKPOINT=${1:-${BASE}/pixel_diffusion_aligned_full/checkpoints/last.ckpt}
OUTPUT=${2:-${BASE}/pixel_diffusion_aligned_full/sfh_extremes_${SLURM_JOB_ID}}
# Snapshot a checkpoint before sampling if training is still updating last.ckpt.
SNAPSHOT=$(mktemp "${TMPDIR:-/tmp}/diffusion_extremes.XXXXXX.ckpt")
trap 'rm -f "${SNAPSHOT}"' EXIT
cp "${CHECKPOINT}" "${SNAPSHOT}"
python -u -m euclid.sample_diffusion_sfh_extremes \
  --checkpoint "${SNAPSHOT}" --conditions "${BASE}/diffusion_conditions_aligned_best.npz" \
  --dataset "${BASE}/sfh_clip_150k.h5" --stamps "${BASE}/zoobot_stamps_rmax" \
  --output "${OUTPUT}" --n-seeds 8 --steps 100
