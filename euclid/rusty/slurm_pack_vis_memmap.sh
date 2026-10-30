#!/bin/bash
#SBATCH --job-name=pack_vis64
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/pack_vis64_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/pack_vis64_%j.err

set -euo pipefail

module load python
source "${RUSTY_VENV:-${HOME}/.sfh_euclid}/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
ARCHIVE=/mnt/home/mhuertas/ceph/euclid/sfh_clip/archives/edfn_vislt22p0_150000
DATASET=${ARCHIVE}/sfh_clip_150k.h5
FITS_ROOT=${ARCHIVE}/cutouts_run
OUTPUT=${ARCHIVE}/packed_vis_64

cd "${REPO}"
[[ -f "${DATASET}" ]] || { echo "Missing ${DATASET}" >&2; exit 2; }
[[ -d "${FITS_ROOT}/cutouts/VIS" ]] || { echo "Missing native VIS FITS" >&2; exit 2; }

echo "Repository commit: $(git rev-parse HEAD)"
echo "Packing native VIS FITS into ${OUTPUT}"
python -u -m euclid.pack_vis_memmap \
    --dataset "${DATASET}" \
    --fits-root "${FITS_ROOT}" \
    --output "${OUTPUT}" \
    --image-size 64 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 16
