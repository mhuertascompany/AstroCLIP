#!/bin/bash
#SBATCH --job-name=iaaft_window
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/iaaft_window_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/iaaft_window_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# End-to-end temporal-window ablation:
#   1. preserve exactly [WINDOW_START, WINDOW_START+0.1) in fractional time;
#   2. independently IAAFT-randomize both outside segments;
#   3. train a matched SFH autoencoder;
#   4. train the edge-on-ablated ZooBot/AE-adjacency alignment.
#
# Usage: sbatch $0 WINDOW_START
# Example: sbatch $0 0.1   # preserves [0.1, 0.2)

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 WINDOW_START" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
SOURCE=${BASE}/sfh_clip_150k.h5
STAMP_ROOT=${BASE}/zoobot_stamps_rmax
PAIR_SPLIT=${BASE}/training_bright_frozen_unrestricted_ae_no_edgeon/pair_split.npz
WINDOW_START=$1

read -r WINDOW_END LABEL <<EOF
$(python - "${WINDOW_START}" <<'PY'
import sys
start = float(sys.argv[1])
end = start + 0.1
if start < 0.1 or end > 1.0 + 1e-10:
    raise SystemExit("WINDOW_START must lie in [0.1, 0.9]")
print(f"{end:.10g} {round(start * 100):02d}_{round(end * 100):02d}")
PY
)
EOF

IAAFT=${BASE}/sfh_iaaft_window_${LABEL}_150k.h5
AE_OUTPUT=${BASE}/sfh_autoencoder_iaaft_window_${LABEL}_150k
CLIP_OUTPUT=${BASE}/training_bright_ae_iaaft_window_${LABEL}
AE_RUN=euclid_sfh_autoencoder_iaaft_window_${LABEL}_150k
CLIP_RUN=euclid_bright_ae_iaaft_window_${LABEL}

for path in "${SOURCE}" "${PAIR_SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMP_ROOT}/VIS" ]] || { echo "Missing VIS stamps: ${STAMP_ROOT}/VIS" >&2; exit 2; }
for path in "${IAAFT}" "${IAAFT}.partial" "${AE_OUTPUT}" "${CLIP_OUTPUT}"; do
    [[ ! -e "${path}" ]] || { echo "Refusing to overwrite: ${path}" >&2; exit 2; }
done

python -u -m euclid.precompute_iaaft_sfhs \
    --source "${SOURCE}" \
    --output "${IAAFT}" \
    --preserve window \
    --window-start "${WINDOW_START}" \
    --window-end "${WINDOW_END}" \
    --transition-bins 10 \
    --candidates 4 \
    --max-iterations 1000 \
    --integral-tolerance 2e-6 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 256 \
    --seed 42

mkdir -p "${AE_OUTPUT}"
python -u -m euclid.pretrain_sfh_autoencoder \
    --dataset "${IAAFT}" \
    --split "${PAIR_SPLIT}" \
    --output-dir "${AE_OUTPUT}" \
    --run-name "${AE_RUN}" \
    --input-mode median \
    --batch-size 128 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --embed-dim 256 --d-model 128 --n-heads 4 \
    --encoder-layers 4 --decoder-layers 2 \
    --mask-fraction 0.35 --w1-weight 0.5 \
    --max-epochs 50 --warmup-epochs 3 --patience 10 \
    --lr 1e-4 --weight-decay 0.01 \
    --accelerator gpu --devices 1 --precision 16-mixed

AE_CHECKPOINT=$(python - "${AE_OUTPUT}/checkpoints" <<'PY'
import re, sys
from pathlib import Path
paths = list(Path(sys.argv[1]).glob("*.ckpt"))
scored = []
for path in paths:
    match = re.search(r"val_loss=([0-9.]+)", path.name)
    if match:
        scored.append((float(match.group(1).rstrip('.')), path))
if not scored:
    raise SystemExit("No validation-scored SFH autoencoder checkpoint found")
print(min(scored)[1])
PY
)

mkdir -p "${CLIP_OUTPUT}"
cp "${PAIR_SPLIT}" "${CLIP_OUTPUT}/pair_split.npz"
python -u -m euclid.train_zoobot_clip \
    --dataset "${SOURCE}" \
    --sfh-override "${IAAFT}" \
    --stamp-root "${STAMP_ROOT}" --band VIS \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${CLIP_OUTPUT}" --run-name "${CLIP_RUN}" \
    --sfh-pretrained-checkpoint "${AE_CHECKPOINT}" \
    --freeze-sfh-encoder --sfh-reconstruction-weight 0 \
    --image-projection mlp --image-projection-hidden-dim 1024 \
    --image-projection-hidden-layers 2 \
    --sfh-projection mlp --sfh-projection-hidden-dim 1024 \
    --sfh-projection-hidden-layers 2 \
    --no-sample-posterior \
    --exclude-edge-on-axis-ratio-below 0.5 \
    --exclude-edge-on-probability-above 0.8 \
    --batch-size 128 --soft-positive-weight 0 \
    --ae-adjacency-weight 100 --ae-adjacency-warmup-epochs 3 \
    --ae-adjacency-temperature 0.07 --queue-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK}" --image-size 224 \
    --embed-dim 256 --sfh-encoder transformer --sfh-d-model 128 \
    --sfh-n-heads 4 --sfh-n-layers 4 \
    --max-epochs 60 --warmup-epochs 3 --patience 12 \
    --lr 1e-4 --weight-decay 0.01 --unfreeze-blocks 0 \
    --accelerator gpu --devices 1 --precision 16-mixed

echo "Preserved fractional-time window: [${WINDOW_START}, ${WINDOW_END})"
echo "IAAFT dataset: ${IAAFT}"
echo "SFH autoencoder: ${AE_OUTPUT}"
echo "CLIP alignment: ${CLIP_OUTPUT}"
