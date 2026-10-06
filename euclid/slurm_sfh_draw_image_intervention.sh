#!/bin/bash
#SBATCH --job-name=sfh_image_intervention
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sfh_image_intervention_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sfh_image_intervention_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage: sbatch $0 [output_dir] [n_galaxies] [n_draws]
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
PREDICTIVE=${BASE}/conditional_sfh_diffusion_sfh_sfr100/validation_predictive_sfhs.h5
DATASET=${BASE}/sfh_clip_150k.h5
CONDITION_CACHE=${BASE}/diffusion_conditions_aligned_best.npz
PIXEL_CHECKPOINT=${BASE}/pixel_diffusion_aligned_full/checkpoints/last.ckpt
OUTPUT=${1:-${BASE}/sfh_draw_image_intervention_${SLURM_JOB_ID}}
N_GALAXIES=${2:-6}
N_DRAWS=${3:-5}

CLIP_CHECKPOINT=$(python - "${CONDITION_CACHE}" <<'PY'
import json
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as source:
    print(json.loads(str(source['metadata']))['checkpoint'])
PY
)

for path in \
    "${PREDICTIVE}" "${DATASET}" "${CONDITION_CACHE}" \
    "${PIXEL_CHECKPOINT}" "${CLIP_CHECKPOINT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Output already exists: ${OUTPUT}" >&2; exit 2; }

export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"

python -u -m euclid.sfh_draw_image_intervention \
    --predictive "${PREDICTIVE}" \
    --dataset "${DATASET}" \
    --condition-cache "${CONDITION_CACHE}" \
    --clip-checkpoint "${CLIP_CHECKPOINT}" \
    --pixel-checkpoint "${PIXEL_CHECKPOINT}" \
    --output "${OUTPUT}" \
    --n-galaxies "${N_GALAXIES}" \
    --n-draws "${N_DRAWS}" \
    --sfr-tolerance 0.15 \
    --image-seeds 42 43 \
    --steps 100 \
    --guidance 1 \
    --selection-seed 42 \
    --device cuda

echo "Results: ${OUTPUT}"
