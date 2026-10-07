#!/bin/bash
#SBATCH --job-name=diff_zmorph_noedge
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/diff_zmorph_noedge_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/diff_zmorph_noedge_%j.err

set -euo pipefail

# Cache the aligned morphology embedding z_morph for every object in the exact
# edge-on-removed AE-adjacency split. Optional arguments: checkpoint, output.
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
RUN=${BASE}/training_bright_frozen_unrestricted_ae_no_edgeon
CHECKPOINT=${1:-${RUN}/checkpoints/last.ckpt}
OUTPUT=${2:-${BASE}/diffusion_conditions_zmorph_ae_adjacency_no_edgeon.npz}
DATASET=${BASE}/sfh_clip_150k.h5
SPLIT=${RUN}/pair_split.npz
STAMPS=${BASE}/zoobot_stamps_rmax

for path in "${CHECKPOINT}" "${DATASET}" "${SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMPS}/VIS" ]] || { echo "Missing VIS stamps: ${STAMPS}/VIS" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite: ${OUTPUT}" >&2; exit 2; }

python -u -m euclid.prepare_diffusion_conditions \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --stamps "${STAMPS}" \
    --output "${OUTPUT}" \
    --modality image \
    --batch-size 256 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --device cuda

echo "Aligned morphology conditions: ${OUTPUT}"
