#!/bin/bash
#SBATCH --job-name=sfh_ae_iaaft_full
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/sfh_ae_iaaft_full_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/sfh_ae_iaaft_full_%j.err

set -euo pipefail
module load python
VENV="${RUSTY_VENV:-${HOME}/.sfh_euclid}"
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
DATASET=${BASE}/runs/iaaft_full_curve/sfh_iaaft_full_curve_150k.h5
SPLIT=${BASE}/archives/edfn_vislt22p0_150000/reference_splits/pair_split_ae_no_edgeon.npz
OUTPUT=${BASE}/runs/iaaft_full_curve/sfh_autoencoder_full
RESUME_FROM="${1:-}"
cd "${REPO}"

for path in "${DATASET}" "${SPLIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    [[ -f "${RESUME_FROM}" ]] || { echo "Missing checkpoint: ${RESUME_FROM}" >&2; exit 2; }
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
elif [[ -f "${OUTPUT}/checkpoints/last.ckpt" ]]; then
    RESUME_ARGS=(--resume-from "${OUTPUT}/checkpoints/last.ckpt")
elif [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Nonempty output without last.ckpt: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT}"

python -u -m euclid.pretrain_sfh_autoencoder \
    --dataset "${DATASET}" --split "${SPLIT}" \
    --output-dir "${OUTPUT}" --run-name euclid_sfh_autoencoder_iaaft_full_curve_150k \
    --input-mode median --batch-size 128 --num-workers "${SLURM_CPUS_PER_TASK}" \
    --embed-dim 256 --d-model 128 --n-heads 4 \
    --encoder-layers 4 --decoder-layers 2 \
    --mask-fraction 0.35 --w1-weight 0.5 \
    --max-epochs 50 --warmup-epochs 3 --patience 10 \
    --lr 1e-4 --weight-decay 0.01 \
    --accelerator gpu --devices 1 --precision 16-mixed \
    "${RESUME_ARGS[@]}"
