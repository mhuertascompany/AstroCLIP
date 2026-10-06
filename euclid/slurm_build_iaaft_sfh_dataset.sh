#!/bin/bash
#SBATCH --job-name=build_iaaft_sfh
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/build_iaaft_sfh_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/build_iaaft_sfh_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Build the full compact IAAFT file. The job fails immediately if any stored
# SFH differs from unit integral by more than 2e-6.
# Usage: sbatch $0 [recent|past90] [source_h5] [output_h5]

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
MODE=${1:-recent}
if [[ "${MODE}" != "recent" && "${MODE}" != "past90" ]]; then
    echo "Usage: sbatch $0 [recent|past90] [source_h5] [output_h5]" >&2
    exit 2
fi
SOURCE=${2:-${BASE}/sfh_clip_150k.h5}
if [[ "${MODE}" == "recent" ]]; then
    OUTPUT=${3:-${BASE}/sfh_iaaft_recent10_150k.h5}
    PRESERVE=recent
    TRANSITION_BINS=10
else
    OUTPUT=${3:-${BASE}/sfh_iaaft_past90_150k.h5}
    PRESERVE=past
    # The randomized recent interval contains only about 25 bins. Three bins
    # suppress a boundary jump while leaving most of that interval perturbed.
    TRANSITION_BINS=3
fi

[[ -f "${SOURCE}" ]] || { echo "Missing source dataset: ${SOURCE}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" && ! -e "${OUTPUT}.partial" ]] || {
    echo "Output or partial output already exists: ${OUTPUT}" >&2
    exit 2
}

python -u -m euclid.precompute_iaaft_sfhs \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --preserve "${PRESERVE}" \
    --recent-fraction 0.1 \
    --transition-bins "${TRANSITION_BINS}" \
    --candidates 4 \
    --max-iterations 1000 \
    --integral-tolerance 2e-6 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 256 \
    --seed 42
