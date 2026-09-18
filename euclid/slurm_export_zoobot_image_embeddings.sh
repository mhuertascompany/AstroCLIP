#!/bin/bash
#SBATCH --job-name=euclid_zoobot_embed
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/zoobot_embed_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/zoobot_embed_%j.err
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

# Usage:
#   sbatch euclid/slurm_export_zoobot_image_embeddings.sh \
#       [sfh_clip.h5] [stamp_root] [output_dir] [max_objects]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
DATASET=${1:-${BASE}/sfh_clip_150k.h5}
STAMP_ROOT=${2:-${BASE}/zoobot_stamps_rmax}
OUTPUT=${3:-${BASE}/zoobot_image_embedding_30k}
MAX_OBJECTS=${4:-30000}

mkdir -p "${HF_HOME}"
cd /n03data/huertas/python/AstroCLIP

python -u -m euclid.export_zoobot_image_embeddings \
    --dataset "${DATASET}" \
    --stamp-root "${STAMP_ROOT}" \
    --output-dir "${OUTPUT}" \
    --model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --band VIS \
    --image-size 224 \
    --batch-size 256 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --max-objects "${MAX_OBJECTS}" \
    --seed 42 \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --device cuda
