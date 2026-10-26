#!/bin/bash
#SBATCH --job-name=clip_iaaft_smoke
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/clip_iaaft_smoke_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/clip_iaaft_smoke_%j.err

set -euo pipefail

module load python
VENV="${RUSTY_VENV:-${HOME}/.sfh_euclid}"
source "${VENV}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
ARCHIVE="${BASE}/archives/edfn_vislt22p0_150000"
DATASET="${ARCHIVE}/sfh_clip_150k.h5"
SFH_OVERRIDE="${BASE}/runs/iaaft_random_window/sfh_iaaft_window_random_window_past90_150k.h5"
STAMP_ROOT="${ARCHIVE}/zoobot_stamps_rmax"
AE_CKPT="${BASE}/runs/iaaft_random_window/sfh_autoencoder_full/checkpoints/euclid_sfh_autoencoder_iaaft_random_window_150k-epoch=019-val_loss=0.01495.ckpt"
OUTPUT="${BASE}/runs/iaaft_random_window/clip_ae_adjacency_no_edgeon_smoke"
RESUME_FROM="${1:-}"

export HF_HOME="${BASE}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${BASE}/cache/torch"

cd "${REPO}"
git merge-base --is-ancestor 6c3d142 HEAD || {
    echo "Checkout lacks required random-window boundary fix 6c3d142" >&2
    exit 2
}

for path in "${DATASET}" "${SFH_OVERRIDE}" "${AE_CKPT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -d "${STAMP_ROOT}/VIS" ]] || {
    echo "Missing ZooBot stamps: ${STAMP_ROOT}/VIS" >&2
    exit 2
}

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    [[ -f "${RESUME_FROM}" ]] || { echo "Missing resume checkpoint: ${RESUME_FROM}" >&2; exit 2; }
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
elif [[ -d "${OUTPUT}" ]] && [[ -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output is not empty: ${OUTPUT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT}" "${HF_HUB_CACHE}" "${TORCH_HOME}"

python - <<'PY'
import lightning
import timm
import torch
print("Torch:", torch.__version__, torch.__file__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Lightning:", lightning.__version__)
print("timm:", timm.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("GPU:", torch.cuda.get_device_name(0))
PY

STAMP_COUNT=$(find "${STAMP_ROOT}/VIS" -maxdepth 1 -type f -name 'VIS_*.jpg' | wc -l)
[[ "${STAMP_COUNT}" -ge 2048 ]] || { echo "Only ${STAMP_COUNT} ZooBot stamps found" >&2; exit 2; }

echo "Repository branch: $(git branch --show-current)"
echo "Repository commit: $(git rev-parse HEAD)"
echo "Source dataset: ${DATASET}"
echo "SFH override: ${SFH_OVERRIDE}"
echo "SFH autoencoder: ${AE_CKPT}"
echo "ZooBot stamps: ${STAMP_COUNT}"
echo "Output: ${OUTPUT}"

python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" \
    --sfh-override "${SFH_OVERRIDE}" \
    --stamp-root "${STAMP_ROOT}" \
    --band VIS \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${OUTPUT}" \
    --run-name euclid_clip_iaaft_random_window_smoke \
    --sfh-pretrained-checkpoint "${AE_CKPT}" \
    --freeze-sfh-encoder \
    --sfh-reconstruction-weight 0 \
    --image-projection mlp \
    --image-projection-hidden-dim 1024 \
    --image-projection-hidden-layers 2 \
    --sfh-projection mlp \
    --sfh-projection-hidden-dim 1024 \
    --sfh-projection-hidden-layers 2 \
    --no-sample-posterior \
    --exclude-edge-on-axis-ratio-below 0.5 \
    --exclude-edge-on-probability-above 0.8 \
    --max-pairs 2048 \
    --batch-size 128 \
    --soft-positive-weight 0 \
    --ae-adjacency-weight 100 \
    --ae-adjacency-warmup-epochs 3 \
    --ae-adjacency-temperature 0.07 \
    --queue-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --image-size 224 \
    --embed-dim 256 \
    --sfh-encoder transformer \
    --sfh-d-model 128 \
    --sfh-n-heads 4 \
    --sfh-n-layers 4 \
    --max-epochs 5 \
    --warmup-epochs 1 \
    --patience 5 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --unfreeze-blocks 0 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    "${RESUME_ARGS[@]}"

echo "Random-window CLIP smoke test: OK"
