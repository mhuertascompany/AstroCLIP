#!/bin/bash
#SBATCH --job-name=euclid_stamps
#SBATCH --partition=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/prepare_stamps_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/prepare_stamps_%j.err

set -euo pipefail

module load python

VENV="${RUSTY_VENV:-${HOME}/.sfh_euclid}"
[[ -f "${VENV}/bin/activate" ]] || {
    echo "Missing Rusty environment: ${VENV}/bin/activate" >&2
    exit 2
}
source "${VENV}/bin/activate"

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip/archives/edfn_vislt22p0_150000

CUTOUT_ROOT="${BASE}/cutouts_run"
CATALOG="${CUTOUT_ROOT}/morphology_catalog.fits"

# Pass a positive integer for a smoke test:
#   sbatch this_script.sh 32
#
# Call without arguments for the complete sample:
#   sbatch this_script.sh

LIMIT="${1:-0}"
EXTRA_ARGS=()

if [[ "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    OUTPUT="${BASE}/zoobot_stamps_rmax_test${LIMIT}"
    EXTRA_ARGS=(--limit "${LIMIT}")
elif [[ "${LIMIT}" == "0" ]]; then
    OUTPUT="${BASE}/zoobot_stamps_rmax"
else
    echo "Usage: sbatch $0 [positive_test_limit]" >&2
    exit 2
fi

cd "${REPO}"

[[ -f "${CUTOUT_ROOT}/manifest.csv" ]] || {
    echo "Missing manifest: ${CUTOUT_ROOT}/manifest.csv" >&2
    exit 2
}
[[ -d "${CUTOUT_ROOT}/cutouts/VIS" ]] || {
    echo "Missing VIS FITS directory: ${CUTOUT_ROOT}/cutouts/VIS" >&2
    exit 2
}
[[ -f "${CATALOG}" ]] || {
    echo "Missing sizing catalog: ${CATALOG}" >&2
    exit 2
}

python - <<'PY'
import astropy
import numpy
import skimage
from PIL import Image

print("NumPy:", numpy.__version__)
print("Astropy:", astropy.__version__)
print("scikit-image:", skimage.__version__)
print("Pillow:", Image.__version__ if hasattr(Image, "__version__") else "available")
PY

N_INPUT=$(find "${CUTOUT_ROOT}/cutouts/VIS" \
    -maxdepth 1 -type f \
    \( -name '*.fits' -o -name '*.fit' -o -name '*.fits.gz' -o -name '*.fz' \) \
    | wc -l)

echo "Input VIS FITS files: ${N_INPUT}"
echo "Sizing catalog: ${CATALOG}"
echo "Output directory: ${OUTPUT}"
echo "Limit: ${LIMIT}"

if [[ "${N_INPUT}" -ne 150000 ]]; then
    echo "Expected 150000 VIS FITS files, found ${N_INPUT}" >&2
    exit 2
fi

python -u -m euclid.prepare_zoobot_cutouts \
    --cutout-root "${CUTOUT_ROOT}" \
    --catalog "${CATALOG}" \
    --output "${OUTPUT}" \
    --band VIS \
    --image-size 224 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --resume \
    "${EXTRA_ARGS[@]}"

python - "${OUTPUT}" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

output = Path(sys.argv[1])
manifest = output / "manifest.csv"

with manifest.open(newline="") as stream:
    rows = list(csv.DictReader(stream))

counts = Counter(row["status"] for row in rows)
paths = sorted((output / "VIS").glob("*.jpg"))

print("Manifest rows:", len(rows))
print("Status counts:", dict(counts))
print("VIS JPEG files:", len(paths))

if not paths:
    raise SystemExit("No JPEG stamps were generated")
if counts.get("failed", 0):
    raise SystemExit(f"{counts['failed']} conversions failed")

for path in paths[:3]:
    with Image.open(path) as image:
        print(path.name, image.mode, image.size)
        if image.mode != "L" or image.size != (224, 224):
            raise SystemExit(f"Unexpected image format: {path}")
PY
