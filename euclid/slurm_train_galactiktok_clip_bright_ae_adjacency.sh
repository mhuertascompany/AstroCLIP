#!/bin/bash
#SBATCH --job-name=gtt_ae_adjacency
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/train_gtt_ae_adjacency_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/train_gtt_ae_adjacency_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage: sbatch $0 TOKENIZER_DIR SFH_AUTOENCODER_CKPT [galactiktok_root] [resume.ckpt]
if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 TOKENIZER_DIR SFH_AUTOENCODER_CKPT [galactiktok_root] [resume.ckpt]" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TOKENIZER_DIR=$1
AUTOENCODER_CKPT=$2
GALACTIKTOK_ROOT=${3:-/n03data/huertas/python/galactiktok/galactiktok}
RESUME_FROM=${4:-}
OUTPUT_DIR=${BASE_DIR}/training_bright_galactiktok_ae_adjacency

[[ -f "${TOKENIZER_DIR}/model.safetensors" ]] || { echo "Missing tokenizer: ${TOKENIZER_DIR}" >&2; exit 2; }
[[ -f "${AUTOENCODER_CKPT}" ]] || { echo "Missing SFH autoencoder: ${AUTOENCODER_CKPT}" >&2; exit 2; }
[[ -d "${GALACTIKTOK_ROOT}/src/galactiktok" ]] || { echo "Missing GalaxyTikTok checkout: ${GALACTIKTOK_ROOT}" >&2; exit 2; }
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ -z "${RESUME_FROM}" ]]; then
    echo "Output is not empty; pass a resume checkpoint or move it: ${OUTPUT_DIR}" >&2
    exit 2
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    [[ -f "${RESUME_FROM}" ]] || { echo "Missing resume checkpoint: ${RESUME_FROM}" >&2; exit 2; }
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

export PYTHONPATH=${GALACTIKTOK_ROOT}/src:${PYTHONPATH:-}
mkdir -p "${OUTPUT_DIR}"
python -u -m euclid.train_zoobot_clip \
    --dataset "${BASE_DIR}/sfh_clip_150k.h5" \
    --fits-root "${BASE_DIR}/cutouts_run" \
    --eligibility-stamp-root "${BASE_DIR}/zoobot_stamps_rmax" \
    --image-stats "${TOKENIZER_DIR}/../image_stats.json" \
    --band VIS \
    --galactiktok-checkpoint "${TOKENIZER_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_bright_galactiktok_ae_adjacency_150k \
    --sfh-pretrained-checkpoint "${AUTOENCODER_CKPT}" \
    --freeze-sfh-encoder \
    --sfh-reconstruction-weight 0 \
    --token-pool-hidden-dim 256 \
    --token-pool-heads 4 \
    --token-pool-layers 2 \
    --sfh-projection mlp \
    --sfh-projection-hidden-dim 1024 \
    --sfh-projection-hidden-layers 2 \
    --no-sample-posterior \
    --batch-size 128 \
    --soft-positive-weight 0 \
    --ae-adjacency-weight 100 \
    --ae-adjacency-warmup-epochs 3 \
    --ae-adjacency-temperature 0.07 \
    --queue-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --image-size 96 \
    --embed-dim 256 \
    --sfh-encoder transformer \
    --sfh-d-model 128 \
    --sfh-n-heads 4 \
    --sfh-n-layers 4 \
    --max-epochs 60 \
    --warmup-epochs 3 \
    --patience 12 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    "${RESUME_ARGS[@]}"
