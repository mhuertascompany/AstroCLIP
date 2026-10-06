#!/bin/bash
#SBATCH --job-name=analyze_sfh_res
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/analyze_sfh_res_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/analyze_sfh_res_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TRAINING=${BASE}/conditional_sfh_diffusion_sfh_sfr100
PREDICTIVE=${1:-${TRAINING}/validation_predictive_sfhs.h5}
OUTPUT=${2:-${TRAINING}/residual_analysis}
DATASET=${BASE}/sfh_clip_150k.h5

for path in "${PREDICTIVE}" "${DATASET}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done

cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.analyze_conditional_sfh_residuals \
    --predictive "${PREDICTIVE}" \
    --dataset "${DATASET}" \
    --output "${OUTPUT}" \
    --folds 5 \
    --seed 42
