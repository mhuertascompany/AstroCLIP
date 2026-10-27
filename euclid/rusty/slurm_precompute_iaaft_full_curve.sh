#!/bin/bash
#SBATCH --job-name=iaaft_full
#SBATCH --partition=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/iaaft_full_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/iaaft_full_%j.err

set -euo pipefail
module load python
VENV="${RUSTY_VENV:-${HOME}/.sfh_euclid}"
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
SOURCE=/mnt/home/mhuertas/ceph/euclid/sfh_clip/archives/edfn_vislt22p0_150000/sfh_clip_150k.h5
RUN_ROOT=/mnt/home/mhuertas/ceph/euclid/sfh_clip/runs/iaaft_full_curve
OUTPUT=${RUN_ROOT}/sfh_iaaft_full_curve_150k.h5

cd "${REPO}"
git merge-base --is-ancestor cfad781 HEAD || {
    echo "Checkout lacks full-curve IAAFT implementation cfad781" >&2
    exit 2
}
[[ -f "${SOURCE}" ]] || { echo "Missing source: ${SOURCE}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" && ! -e "${OUTPUT}.partial" ]] || {
    echo "Refusing to overwrite ${OUTPUT} or its partial file" >&2
    exit 2
}
mkdir -p "${RUN_ROOT}"

echo "Repository commit: $(git rev-parse HEAD)"
echo "Full-curve IAAFT: no preserved bins or regional integrals"
python -u -m euclid.precompute_iaaft_sfhs \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --preserve none \
    --candidates 4 \
    --max-iterations 1000 \
    --integral-tolerance 2e-6 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 256 \
    --seed 42

python - "${OUTPUT}" <<'PY'
import sys
import h5py
path = sys.argv[1]
with h5py.File(path, "r") as data:
    assert data["sfh"].shape == (150000, 250)
    assert data.attrs["preserved_segment"] == "none"
    assert data.attrs["maximum_absolute_stored_integral_error"] <= 2e-6
    assert "preserved_window_start" not in data
    print("Transformation:", data.attrs["transformation"])
    print("Original-surrogate mean-lookback correlation:",
          data.attrs["population_correlation_original_surrogate_mean_lookback"])
    print("Original-surrogate recent-10% correlation:",
          data.attrs["population_correlation_original_surrogate_recent_10"])
    print("Median temporal correlation:", data.attrs["median_full_correlation"])
    print("Median spectral error:", data.attrs["median_spectral_error"])
print("Full-curve IAAFT dataset validation: OK")
PY
