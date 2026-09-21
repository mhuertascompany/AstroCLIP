#!/bin/bash
#SBATCH --job-name=diff_sfh_clusters
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/diff_clusters_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/diff_clusters_%j.err
set -euo pipefail
# Usage: sbatch euclid/slurm_sample_diffusion_clusters.sh selection.csv [checkpoint] [new_output_directory]
# One job / checkpoint snapshot; same seeds for every selected SFH in every cluster.
SELECTION=${1:?Provide the explorer selection CSV path}
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
CHECKPOINT=${2:-${BASE}/pixel_diffusion_aligned_full/checkpoints/last.ckpt}
OUTPUT=${3:-${BASE}/pixel_diffusion_aligned_full/cluster_selection_${SLURM_JOB_ID}}
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
python -u -m euclid.sample_diffusion_selection \
  --selection "${SELECTION}" --checkpoint "${CHECKPOINT}" \
  --conditions "${BASE}/diffusion_conditions_aligned_best.npz" \
  --dataset "${BASE}/sfh_clip_150k.h5" --stamps "${BASE}/zoobot_stamps_rmax" \
  --output "${OUTPUT}" --n-seeds 4 --guidance 1 2 --steps 100
