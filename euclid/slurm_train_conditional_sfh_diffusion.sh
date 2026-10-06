#!/bin/bash
#SBATCH --job-name=sfh_cond_diff
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sfh_cond_diff_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/sfh_cond_diff_%j.err
#SBATCH --partition=pscomp
#SBATCH --nodelist=n36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

set -euo pipefail

# Usage: sbatch $0 [full|smoke] [output_dir] [resume_checkpoint]
MODE=${1:-full}
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
DATASET=${BASE}/sfh_clip_150k.h5
SPLIT=${BASE}/training_bright_frozen_unrestricted_ae_no_edgeon/pair_split.npz
OUTPUT=${2:-${BASE}/conditional_sfh_diffusion_phz}
RESUME=${3:-}

[[ "${MODE}" == "full" || "${MODE}" == "smoke" ]] || {
    echo "First argument must be full or smoke." >&2; exit 2;
}
for path in "${DATASET}" "${SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

EXTRA=()
[[ "${MODE}" == "smoke" ]] && EXTRA+=(--smoke)
if [[ -n "${RESUME}" ]]; then
    [[ -f "${RESUME}" ]] || { echo "Missing resume checkpoint: ${RESUME}" >&2; exit 2; }
    EXTRA+=(--resume "${RESUME}")
fi

cd /n03data/huertas/python/AstroCLIP
python -u -m euclid.train_conditional_sfh_diffusion \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}" \
    --input-mode posterior \
    --batch-size 256 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --epochs 100 \
    --patience 15 \
    --d-model 128 \
    --n-heads 4 \
    --n-layers 4 \
    --diffusion-steps 1000 \
    --condition-dropout 0.15 \
    --lr 1e-4 \
    --accelerator gpu \
    "${EXTRA[@]}"

