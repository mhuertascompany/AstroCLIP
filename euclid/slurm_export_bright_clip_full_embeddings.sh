#!/bin/bash
#SBATCH --job-name=bright_full_embed
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/full_embed_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/full_embed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=pscomp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TRAINING=${BASE}/training_bright_frozen_mlp
CHECKPOINT=${1:-${TRAINING}/checkpoints/euclid_bright_frozen_mlp_150k-epoch=020-val_loss=4.2759.ckpt}
OUTPUT=${2:-${TRAINING}/full_sample_explorer}
DATASET=${BASE}/sfh_clip_150k.h5
STAMPS=${BASE}/zoobot_stamps_rmax

[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATASET}" ]] || { echo "Missing dataset: ${DATASET}" >&2; exit 2; }
[[ -d "${STAMPS}/VIS" ]] || { echo "Missing stamps: ${STAMPS}/VIS" >&2; exit 2; }
mkdir -p "${OUTPUT}"

python -u -m euclid.export_zoobot_clip_full_embeddings \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --stamp-root "${STAMPS}" \
  --output "${OUTPUT}/full_embeddings.npz" \
  --batch-size 128 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --device cuda
