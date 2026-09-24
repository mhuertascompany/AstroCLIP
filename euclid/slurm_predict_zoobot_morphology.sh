#!/bin/bash
#SBATCH --job-name=euclid_zoobot_full
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/zoobot_full_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/zoobot_full_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage:
#   sbatch euclid/slurm_predict_zoobot_morphology.sh \
#       [sfh_clip.h5] [stamp_root] [output.fits] [max_objects]
# max_objects=0 runs every available stamp.

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
DATASET=${1:-${BASE}/sfh_clip_150k.h5}
STAMP_ROOT=${2:-${BASE}/zoobot_stamps_rmax}
OUTPUT=${3:-${BASE}/zoobot_full_predictions.fits}
MAX_OBJECTS=${4:-0}

mkdir -p "${HF_HOME}" "$(dirname "${OUTPUT}")"
cd /n03data/huertas/python/AstroCLIP

DATASET_ARGS=()
if [[ -f "${DATASET}" ]]; then
    DATASET_ARGS=(--dataset "${DATASET}")
else
    echo "HDF5 not found; classifying directly from stamp filenames: ${DATASET}" >&2
fi

python - <<'PY'
try:
    import zoobot
except ImportError as error:
    raise SystemExit(
        'Zoobot is missing from cosmos_visual. Install it once with:\n'
        'python -m pip install "zoobot[pytorch]"'
    ) from error
PY

python -u -m euclid.predict_zoobot_morphology \
    "${DATASET_ARGS[@]}" \
    --stamp-root "${STAMP_ROOT}" \
    --output "${OUTPUT}" \
    --repo-id mwalmsley/zoobot-finetuned-euclid \
    --filename FinetuneableZoobotTree.ckpt \
    --band VIS \
    --image-size 224 \
    --batch-size 256 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --max-objects "${MAX_OBJECTS}" \
    --device cuda
