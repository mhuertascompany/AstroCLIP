#!/bin/bash
#SBATCH --job-name=clip_iaaft_ablation
#SBATCH --output=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/clip_iaaft_ablation_%j.out
#SBATCH --error=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000/clip_iaaft_ablation_%j.err
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

# Matched ZooBot/AE-adjacency alignment for original or IAAFT SFHs.
# Usage: sbatch $0 control|iaaft AUTOENCODER_CKPT \
#        [source_dataset] [stamp_root] [output_dir] [sfh_override] [resume.ckpt]

if [[ $# -lt 2 || ( "$1" != "control" && "$1" != "iaaft" ) ]]; then
    echo "Usage: sbatch $0 control|iaaft AUTOENCODER_CKPT [source_dataset] [stamp_root] [output_dir] [sfh_override] [resume.ckpt]" >&2
    exit 2
fi

source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/
export HF_HOME=${HF_HOME:-/n03data/huertas/.cache/huggingface}

MODE=$1
AUTOENCODER_CKPT=$2
BASE=/n03data/huertas/euclid/sfh_clip/edfn_vislt22p0_150000
DATASET=${3:-${BASE}/sfh_clip_150k.h5}
STAMP_ROOT=${4:-${BASE}/zoobot_stamps_rmax}
if [[ "${MODE}" == "control" ]]; then
    DEFAULT_OUTPUT=${BASE}/training_bright_ae_recent10_control
    RUN_NAME=euclid_bright_ae_recent10_control
    DEFAULT_OVERRIDE=
else
    DEFAULT_OUTPUT=${BASE}/training_bright_ae_iaaft_recent10
    RUN_NAME=euclid_bright_ae_iaaft_recent10
    DEFAULT_OVERRIDE=${BASE}/sfh_iaaft_recent10_150k.h5
fi
OUTPUT_DIR=${5:-${DEFAULT_OUTPUT}}
SFH_OVERRIDE=${6:-${DEFAULT_OVERRIDE}}
RESUME_FROM=${7:-}

[[ -f "${AUTOENCODER_CKPT}" ]] || { echo "Missing autoencoder checkpoint: ${AUTOENCODER_CKPT}" >&2; exit 2; }
[[ -f "${DATASET}" ]] || { echo "Missing source dataset: ${DATASET}" >&2; exit 2; }
[[ -d "${STAMP_ROOT}/VIS" ]] || { echo "Missing VIS stamps: ${STAMP_ROOT}/VIS" >&2; exit 2; }
if [[ "${MODE}" == "iaaft" ]]; then
    [[ -f "${SFH_OVERRIDE}" ]] || { echo "Missing IAAFT SFH override: ${SFH_OVERRIDE}" >&2; exit 2; }
elif [[ -n "${SFH_OVERRIDE}" ]]; then
    echo "The control run must not use an SFH override." >&2
    exit 2
fi
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ -z "${RESUME_FROM}" ]]; then
    echo "Output is not empty; choose a new directory or pass a resume checkpoint: ${OUTPUT_DIR}" >&2
    exit 2
fi

OVERRIDE_ARGS=()
[[ -n "${SFH_OVERRIDE}" ]] && OVERRIDE_ARGS=(--sfh-override "${SFH_OVERRIDE}")
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    [[ -f "${RESUME_FROM}" ]] || { echo "Missing resume checkpoint: ${RESUME_FROM}" >&2; exit 2; }
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"
python -u -m euclid.train_zoobot_clip \
    --dataset "${DATASET}" \
    "${OVERRIDE_ARGS[@]}" \
    --stamp-root "${STAMP_ROOT}" \
    --band VIS \
    --zoobot-model-name hf_hub:mwalmsley/zoobot-encoder-euclid \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --sfh-pretrained-checkpoint "${AUTOENCODER_CKPT}" \
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
    --max-epochs 60 \
    --warmup-epochs 3 \
    --patience 12 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --unfreeze-blocks 0 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    "${RESUME_ARGS[@]}"
