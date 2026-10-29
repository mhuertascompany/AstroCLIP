#!/bin/bash
#SBATCH --job-name=euclid_counterflow
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --requeue
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/counterflow_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/counterflow_%j.err

set -euo pipefail

# Paper-inspired bidirectional raw VIS/SFH counterfactual flow.
# Usage:
#   sbatch euclid/rusty/slurm_counterfactual_flow.sh smoke
#   sbatch euclid/rusty/slurm_counterfactual_flow.sh full

MODE=${1:-smoke}
if [[ "${MODE}" != smoke && "${MODE}" != full ]]; then
    echo "MODE must be smoke or full, received ${MODE}" >&2
    exit 2
fi

# Rusty's system PyTorch 2.6.0 is built against the CUDA 12/cuDNN 8 stack.
# Load both explicitly; otherwise convolution backward fails while searching
# for libcudnn_cnn_train.so.8.
module load python cuda/12.5.1 cudnn/8.9.7.29-12
source "${RUSTY_VENV:-${HOME}/.sfh_euclid}/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE=${BASE}/archives/edfn_vislt22p0_150000
DATASET=${ARCHIVE}/sfh_clip_150k.h5
STAMPS=${ARCHIVE}/zoobot_stamps_rmax
OUTPUT=${BASE}/runs/counterfactual_raw_vis_sfh_${MODE}

cd "${REPO}"
[[ -f "${DATASET}" ]] || { echo "Missing ${DATASET}" >&2; exit 2; }
[[ -d "${STAMPS}/VIS" ]] || { echo "Missing ${STAMPS}/VIS" >&2; exit 2; }

EXTRA=()
if [[ "${MODE}" == smoke ]]; then
    EXTRA=(--smoke --max-pairs 2048 --epochs 2 --batch-size 8 --donors 4 --pool-size 24)
else
    EXTRA=(--epochs 100 --batch-size 16 --donors 8 --pool-size 64)
fi
if [[ -f "${OUTPUT}/checkpoints/last.ckpt" ]]; then
    EXTRA+=(--resume "${OUTPUT}/checkpoints/last.ckpt")
elif [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Nonempty output without last.ckpt: ${OUTPUT}" >&2
    exit 2
fi

echo "Repository commit: $(git rev-parse HEAD)"
echo "Mode: ${MODE}"
echo "Output: ${OUTPUT}"
python -u -m euclid.train_counterfactual_flow \
    --dataset "${DATASET}" \
    --stamps "${STAMPS}" \
    --output "${OUTPUT}" \
    --image-size 64 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --common-dim 64 --prior-dim 64 --condition-dim 128 \
    --image-encoder-base 16 --image-flow-base 16 \
    --sfh-d-model 96 --sfh-layers 3 \
    --common-loss-weight 0.05 --common-dropout 0.1 \
    --max-vis-mag 22 \
    --exclude-edge-on-axis-ratio-below 0.5 \
    --exclude-edge-on-probability-above 0.8 \
    --accelerator gpu "${EXTRA[@]}"
