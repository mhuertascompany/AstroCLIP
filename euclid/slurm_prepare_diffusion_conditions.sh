#!/bin/bash
#SBATCH --job-name=sfh_diff_conditions
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/diff_conditions_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/diff_conditions_%j.err
set -euo pipefail
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
RUN=${BASE}/training_bright_frozen_mlp
python -u -m euclid.prepare_diffusion_conditions \
  --checkpoint "${RUN}/checkpoints/euclid_bright_frozen_mlp_150k-epoch=020-val_loss=4.2759.ckpt" \
  --dataset "${BASE}/sfh_clip_150k.h5" \
  --split "${RUN}/pair_split.npz" \
  --stamps "${BASE}/zoobot_stamps_rmax" \
  --output "${BASE}/diffusion_conditions_aligned_best.npz"
