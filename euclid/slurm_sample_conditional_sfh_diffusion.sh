#!/bin/bash
#SBATCH --job-name=sample_sfh_cond
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sample_sfh_cond_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sample_sfh_cond_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage: sbatch $0 [checkpoint] [output_h5] [draws]
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TRAINING=${BASE}/conditional_sfh_diffusion_phz
CHECKPOINT=${1:-${TRAINING}/checkpoints/last.ckpt}
OUTPUT=${2:-${TRAINING}/validation_predictive_sfhs.h5}
DRAWS=${3:-32}
DATASET=${BASE}/sfh_clip_150k.h5
SPLIT=${TRAINING}/conditional_sfh_split.npz

for path in "${CHECKPOINT}" "${DATASET}" "${SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done

cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.sample_conditional_sfh_diffusion \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}" \
    --partition val \
    --draws "${DRAWS}" \
    --batch-size 64 \
    --sample-steps 100 \
    --guidance 1 \
    --seed 42 \
    --device cuda
