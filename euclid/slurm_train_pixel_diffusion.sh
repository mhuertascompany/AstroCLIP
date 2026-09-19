#!/bin/bash
#SBATCH --job-name=sfh_pixel_diffusion
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/pixel_diffusion_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/pixel_diffusion_%j.err
set -euo pipefail
# Usage: sbatch ...sh [smoke|full] [batch_size] [resume_checkpoint]
MODE=${1:-smoke}
BATCH=${2:-8}
RESUME=${3:-}
case "${MODE}" in
  smoke) EXTRA=(--smoke);;
  full) EXTRA=();;
  *) echo 'Mode must be smoke or full' >&2; exit 2;;
esac
if [[ -n "${RESUME}" ]]; then EXTRA+=(--resume "${RESUME}"); fi
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
python -u -m euclid.train_pixel_diffusion \
  --conditions "${BASE}/diffusion_conditions_aligned_best.npz" \
  --stamps "${BASE}/zoobot_stamps_rmax" \
  --output "${BASE}/pixel_diffusion_aligned_${MODE}" \
  --batch-size "${BATCH}" --accumulate 4 --workers "${SLURM_CPUS_PER_TASK}" \
  --base 32 --epochs 100 "${EXTRA[@]}"
