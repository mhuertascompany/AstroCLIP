#!/bin/bash
#SBATCH --job-name=plot_sfh_draws
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/plot_sfh_draws_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/plot_sfh_draws_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
TRAINING=${BASE}/conditional_sfh_diffusion_sfh_sfr100
PREDICTIVE=${1:-${TRAINING}/validation_predictive_sfhs.h5}
OUTPUT=${2:-${TRAINING}/draw_examples}

[[ -f "${PREDICTIVE}" ]] || { echo "Missing predictive file: ${PREDICTIVE}" >&2; exit 2; }

export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"
cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.plot_conditional_sfh_draws \
    --predictive "${PREDICTIVE}" \
    --output "${OUTPUT}" \
    --n-examples 12 \
    --draws-to-show 10 \
    --seed 42
