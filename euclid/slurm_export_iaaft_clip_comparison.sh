#!/bin/bash
#SBATCH --job-name=export_iaaft_cmp
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/export_iaaft_cmp_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/export_iaaft_cmp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Build a linked explorer for the existing edge-on-filtered AE-adjacency run
# and the recent-10%-preserved IAAFT ablation. Both models are evaluated on
# exactly the same saved validation IDs. Optional arguments:
#   1. reference checkpoint
#   2. IAAFT checkpoint
#   3. output directory

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

REPO=/n03data/huertas/python/AstroCLIP
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
REFERENCE_TRAINING=${BASE}/training_bright_frozen_unrestricted_ae_no_edgeon
IAAFT_TRAINING=${BASE}/training_bright_ae_iaaft_recent10

best_checkpoint() {
    python - "$1" <<'PY'
import re
import sys
from pathlib import Path

directory = Path(sys.argv[1]) / 'checkpoints'
candidates = []
for path in directory.glob('*.ckpt'):
    match = re.search(r'val_loss=([0-9]+(?:\.[0-9]+)?)', path.name)
    if match:
        candidates.append((float(match.group(1)), path))
if candidates:
    print(min(candidates, key=lambda item: item[0])[1])
elif (directory / 'last.ckpt').is_file():
    print(directory / 'last.ckpt')
else:
    raise SystemExit(f'No checkpoint found in {directory}')
PY
}

REFERENCE_CKPT=${1:-$(best_checkpoint "${REFERENCE_TRAINING}")}
IAAFT_CKPT=${2:-$(best_checkpoint "${IAAFT_TRAINING}")}
OUTPUT=${3:-${BASE}/explorer_ae_original_vs_iaaft_recent10}

DATASET=${BASE}/sfh_clip_150k.h5
SFH_OVERRIDE=${BASE}/sfh_iaaft_recent10_150k.h5
STAMPS=${BASE}/zoobot_stamps_rmax
REFERENCE_SPLIT=${REFERENCE_TRAINING}/pair_split.npz
IAAFT_SPLIT=${IAAFT_TRAINING}/pair_split.npz
REFERENCE_EVAL=${OUTPUT}/original_sfh_ae_adjacency
IAAFT_EVAL=${OUTPUT}/iaaft_recent10_ae_adjacency
BUNDLE=${OUTPUT}/explorer_bundle

for path in \
    "${REFERENCE_CKPT}" "${IAAFT_CKPT}" "${DATASET}" "${SFH_OVERRIDE}" \
    "${REFERENCE_SPLIT}" "${IAAFT_SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMPS}/VIS" ]] || { echo "Missing VIS stamps: ${STAMPS}/VIS" >&2; exit 2; }

mkdir -p "${REFERENCE_EVAL}" "${IAAFT_EVAL}"
cd "${REPO}"

python - "${REFERENCE_SPLIT}" "${IAAFT_SPLIT}" <<'PY'
import sys
import numpy as np

reference = np.load(sys.argv[1])
iaaft = np.load(sys.argv[2])
for key in ('train_rows', 'train_ids', 'val_rows', 'val_ids'):
    if not np.array_equal(reference[key], iaaft[key]):
        raise SystemExit(
            f'Saved splits differ for {key}: reference={reference[key].shape}, '
            f'IAAFT={iaaft[key].shape}'
        )
print(
    f"Verified identical splits: train={len(reference['train_ids']):,}, "
    f"validation={len(reference['val_ids']):,}"
)
PY

evaluate_checkpoint() {
    local checkpoint=$1
    local destination=$2
    local override=${3:-}
    if [[ -f "${destination}/validation_embeddings.npz" ]]; then
        echo "Reusing evaluation in ${destination}"
        return
    fi
    local override_args=()
    [[ -n "${override}" ]] && override_args=(--sfh-override "${override}")
    python -u -m euclid.evaluate_zoobot_clip \
        --checkpoint "${checkpoint}" \
        --dataset "${DATASET}" \
        "${override_args[@]}" \
        --stamp-root "${STAMPS}" \
        --split "${IAAFT_SPLIT}" \
        --output-dir "${destination}" \
        --band VIS \
        --batch-size 128 \
        --num-workers "${SLURM_CPUS_PER_TASK}" \
        --chunk-size 512 \
        --n-permutations 200 \
        --permutation-size 5000 \
        --shape-subset 2000 \
        --skip-posterior \
        --device cuda
}

make_umap() {
    local evaluation=$1
    local label=$2
    if [[ -f "${evaluation}/euclid_clip_umap_diagnostics.npz" ]]; then
        echo "Reusing UMAP in ${evaluation}"
        return
    fi
    python -u -m euclid.umap_zoobot_clip \
        --embeddings "${evaluation}/validation_embeddings.npz" \
        --dataset "${DATASET}" \
        --per-object "${evaluation}/per_object.csv" \
        --output "${evaluation}/euclid_clip_umap_diagnostics.pdf" \
        --run-label "${label}" \
        --seed 42
}

export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}
export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

evaluate_checkpoint "${REFERENCE_CKPT}" "${REFERENCE_EVAL}"
evaluate_checkpoint "${IAAFT_CKPT}" "${IAAFT_EVAL}" "${SFH_OVERRIDE}"
make_umap "${REFERENCE_EVAL}" "Original SFHs: AE adjacency"
make_umap "${IAAFT_EVAL}" "IAAFT past; original recent 10%: AE adjacency"

if [[ ! -f "${BUNDLE}/manifest.json" ]]; then
    python -u -m euclid.export_explorer_bundle \
        --dataset "${DATASET}" \
        --stamp-root "${STAMPS}" \
        --umap "${REFERENCE_EVAL}/euclid_clip_umap_diagnostics.npz" \
        --umap "${IAAFT_EVAL}/euclid_clip_umap_diagnostics.npz" \
        --output-dir "${BUNDLE}" \
        --band VIS
fi

tar -C "${OUTPUT}" -cf "${OUTPUT}/explorer_bundle.tar" explorer_bundle
echo "Reference checkpoint: ${REFERENCE_CKPT}"
echo "IAAFT checkpoint: ${IAAFT_CKPT}"
echo "Comparison bundle: ${OUTPUT}/explorer_bundle.tar"
