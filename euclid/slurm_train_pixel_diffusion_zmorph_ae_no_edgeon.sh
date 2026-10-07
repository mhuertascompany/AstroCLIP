#!/bin/bash
#SBATCH --job-name=pixdiff_zmorph
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/pixdiff_zmorph_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/pixdiff_zmorph_%j.err

set -euo pipefail

# Same architecture and optimization as the condition-sensitive SFH run; only
# the conditioning modality changes to the aligned morphology embedding.
# Usage: sbatch $0 [smoke|full] [batch_size] [resume_checkpoint]
MODE=${1:-smoke}
BATCH=${2:-8}
RESUME=${3:-}
case "${MODE}" in
    smoke) EXTRA=(--smoke);;
    full) EXTRA=();;
    *) echo "Mode must be smoke or full" >&2; exit 2;;
esac
if [[ -n "${RESUME}" ]]; then EXTRA+=(--resume "${RESUME}"); fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
CONDITIONS=${BASE}/diffusion_conditions_zmorph_ae_adjacency_no_edgeon.npz
STAMPS=${BASE}/zoobot_stamps_rmax
OUTPUT=${BASE}/pixel_diffusion_zmorph_ae_adjacency_no_edgeon_conditioned_${MODE}

[[ -f "${CONDITIONS}" ]] || { echo "Missing condition cache: ${CONDITIONS}" >&2; exit 2; }
[[ -d "${STAMPS}/VIS" ]] || { echo "Missing VIS stamps: ${STAMPS}/VIS" >&2; exit 2; }

python - "${CONDITIONS}" <<'PY'
import json
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as source:
    metadata = json.loads(str(source['metadata']))
if metadata.get('modality') != 'image':
    raise SystemExit(f"Expected image conditions, found: {metadata}")
print(metadata['condition'])
PY

python -u -m euclid.train_pixel_diffusion \
    --conditions "${CONDITIONS}" \
    --stamps "${STAMPS}" \
    --output "${OUTPUT}" \
    --batch-size "${BATCH}" \
    --accumulate 4 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --base 32 \
    --epochs 100 \
    --condition-dropout 0.10 \
    --high-noise-fraction 0.50 \
    --high-noise-min 0.80 \
    "${EXTRA[@]}"

echo "Aligned-morphology-conditioned diffusion: ${OUTPUT}"
