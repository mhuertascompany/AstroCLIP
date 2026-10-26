#!/bin/bash
#SBATCH --job-name=iaaft_random
#SBATCH --partition=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/iaaft_random_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/iaaft_random_%j.err

set -euo pipefail

module load python

VENV="${RUSTY_VENV:-${HOME}/.sfh_euclid}"
[[ -f "${VENV}/bin/activate" ]] || {
    echo "Missing Rusty environment: ${VENV}/bin/activate" >&2
    exit 2
}
source "${VENV}/bin/activate"

# Prevent each worker process from launching its own BLAS thread pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
SOURCE_BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip/archives/edfn_vislt22p0_150000
RUN_ROOT=/mnt/home/mhuertas/ceph/euclid/sfh_clip/runs/iaaft_random_window

SOURCE="${SOURCE_BASE}/sfh_clip_150k.h5"
OUTPUT="${RUN_ROOT}/sfh_iaaft_window_random_window_past90_150k.h5"
PARTIAL="${OUTPUT}.partial"

cd "${REPO}"

REQUIRED_FIX=6c3d142
if ! git merge-base --is-ancestor "${REQUIRED_FIX}" HEAD; then
    echo "Current checkout does not contain required fix ${REQUIRED_FIX}" >&2
    echo "Current commit: $(git rev-parse HEAD)" >&2
    exit 2
fi

[[ -f "${SOURCE}" ]] || {
    echo "Missing source SFH dataset: ${SOURCE}" >&2
    exit 2
}
[[ ! -e "${OUTPUT}" ]] || {
    echo "Refusing to overwrite completed output: ${OUTPUT}" >&2
    exit 2
}
[[ ! -e "${PARTIAL}" ]] || {
    echo "Refusing to overwrite partial output: ${PARTIAL}" >&2
    exit 2
}

mkdir -p "${RUN_ROOT}"

echo "Repository commit: $(git rev-parse HEAD)"
echo "Source: ${SOURCE}"
echo "Output: ${OUTPUT}"
echo "Workers: ${SLURM_CPUS_PER_TASK}"

python -u -m euclid.precompute_iaaft_sfhs \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --preserve random-window \
    --transition-bins 10 \
    --candidates 4 \
    --max-iterations 1000 \
    --integral-tolerance 2e-6 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --chunk-size 256 \
    --seed 42

python - "${OUTPUT}" <<'PY'
import sys

import h5py
import numpy as np

path = sys.argv[1]

with h5py.File(path, "r") as data:
    ids = np.asarray(data["galaxy_id"])
    sfh = data["sfh"]
    starts = np.asarray(data["preserved_window_start"])
    ends = np.asarray(data["preserved_window_end"])
    time = np.asarray(data["sfh_time_grid"])

    old_starts = np.searchsorted(time, ends, side="left")

    print("Galaxies:", len(ids))
    print("SFH shape:", sfh.shape)
    print("Window start range:", float(starts.min()), float(starts.max()))
    print("Window end range:", float(ends.min()), float(ends.max()))
    print(
        "Maximum stored integral error:",
        float(data.attrs["maximum_absolute_stored_integral_error"]),
    )

    assert len(ids) == 150000
    assert sfh.shape == (150000, 250)
    assert np.all(starts >= 0.1)
    assert np.allclose(ends - starts, 0.1, atol=1e-6)
    assert np.all(len(time) - old_starts >= 4)
    assert float(data.attrs["maximum_absolute_stored_integral_error"]) <= 2e-6

print("Full randomized-SFH dataset validation: OK")
PY
