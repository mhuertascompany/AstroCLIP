#!/bin/bash
#SBATCH --job-name=bright_mlp_test
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/bright_mlp_test_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/bright_mlp_test_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 AUTOENCODER_CKPT" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE_DIR=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
AUTOENCODER_CKPT=$1
OUTPUT_DIR=${BASE_DIR}/training_bright_frozen_mlp_test

if [[ ! -f "${AUTOENCODER_CKPT}" ]]; then
    echo "Missing autoencoder checkpoint: ${AUTOENCODER_CKPT}" >&2
    exit 2
fi
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output is not empty: ${OUTPUT_DIR}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.train_zoobot_clip \
    --dataset "${BASE_DIR}/sfh_clip_150k.h5" \
    --stamp-root "${BASE_DIR}/zoobot_stamps_rmax" \
    --band VIS \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${OUTPUT_DIR}" \
    --run-name euclid_bright_frozen_mlp_test \
    --sfh-pretrained-checkpoint "${AUTOENCODER_CKPT}" \
    --sfh-reconstruction-weight 0 \
    --sfh-projection residual_mlp \
    --sfh-projection-hidden-dim 512 \
    --sfh-projection-residual-scale 0.1 \
    --freeze-sfh-encoder \
    --no-sample-posterior \
    --max-pairs 1024 \
    --batch-size 128 \
    --soft-positive-weight 0 \
    --queue-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --image-size 224 \
    --embed-dim 256 \
    --sfh-encoder transformer \
    --sfh-d-model 128 \
    --sfh-n-heads 4 \
    --sfh-n-layers 4 \
    --sfh-lr-scale 1 \
    --max-epochs 3 \
    --warmup-epochs 1 \
    --patience 3 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --unfreeze-blocks 0 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed
