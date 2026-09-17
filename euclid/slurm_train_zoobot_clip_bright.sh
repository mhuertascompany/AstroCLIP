#!/bin/bash
#SBATCH --job-name=euclid_bright
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_bright_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_100k/train_bright_%j.err
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

# Usage: sbatch euclid/slurm_train_zoobot_clip_bright.sh MAX_VIS_MAG [output_dir] [resume.ckpt]
if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 MAX_VIS_MAG [output_dir] [resume.ckpt]" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE=/n03data/huertas/euclid/sfh_clip/edfn_100k
MAX_VIS_MAG=$1
TAG=${MAX_VIS_MAG/./p}
OUTPUT=${2:-${BASE}/training_transformer_median_vislt${TAG}}
RESUME_FROM=${3:-}
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi
if [[ -d "${OUTPUT}" ]] && [[ -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ -z "${RESUME_FROM}" ]]; then
    echo "Output is not empty; choose a new directory or pass a checkpoint: ${OUTPUT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT}" "${HF_HOME}"
cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.train_zoobot_clip \
    --dataset "${BASE}/sfh_clip_100k.h5" \
    --selection-catalog "${BASE}/sfh_edfn100k/catalog_sfh_100k.fits" \
    --stamp-root "${BASE}/zoobot_stamps_rmax" \
    --band VIS \
    --max-vis-mag "${MAX_VIS_MAG}" \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${OUTPUT}" \
    --run-name "euclid_vis_sfh_transformer_median_vislt${TAG}" \
    --no-sample-posterior \
    --batch-size 128 \
    --queue-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --image-size 224 \
    --embed-dim 256 \
    --sfh-encoder transformer \
    --sfh-d-model 128 \
    --sfh-n-heads 4 \
    --sfh-n-layers 4 \
    --sfh-lr-scale 3 \
    --max-epochs 50 \
    --warmup-epochs 2 \
    --patience 10 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --unfreeze-blocks 0 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    "${RESUME_ARGS[@]}"
