#!/bin/bash
#SBATCH --job-name=sfh_iaaft_test
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sfh_iaaft_test_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sfh_iaaft_test_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
SOURCE=${1:-${BASE}/sfh_clip_150k.h5}
OUTPUT_DIR=${2:-${BASE}/sfh_autoencoder_iaaft_test}
IAAFT=${OUTPUT_DIR}/sfh_iaaft_test_2048.h5

[[ -f "${SOURCE}" ]] || { echo "Missing source dataset: ${SOURCE}" >&2; exit 2; }
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output is not empty: ${OUTPUT_DIR}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}"

python -u -m euclid.precompute_iaaft_sfhs \
    --source "${SOURCE}" \
    --output "${IAAFT}" \
    --max-rows 2048 \
    --recent-fraction 0.1 \
    --transition-bins 10 \
    --candidates 2 \
    --max-iterations 200 \
    --integral-tolerance 2e-6 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 128 \
    --seed 42

python -u -m euclid.pretrain_sfh_autoencoder \
    --dataset "${IAAFT}" \
    --output-dir "${OUTPUT_DIR}/training" \
    --run-name euclid_sfh_autoencoder_iaaft_test \
    --input-mode median \
    --max-objects 2048 \
    --batch-size 128 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --embed-dim 256 \
    --d-model 128 \
    --n-heads 4 \
    --encoder-layers 4 \
    --decoder-layers 2 \
    --mask-fraction 0.35 \
    --w1-weight 0.5 \
    --max-epochs 3 \
    --warmup-epochs 1 \
    --patience 3 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed
