#!/bin/bash
#SBATCH --job-name=sfh_keep_recent
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/sfh_keep_recent_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/sfh_keep_recent_%j.err

set -euo pipefail
module load python
source "${RUSTY_VENV:-${HOME}/.sfh_euclid}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
SOURCE=${BASE}/archives/edfn_vislt22p0_150000/sfh_clip_150k.h5
RUN_ROOT=${BASE}/runs/past_shape_ablation
OUTPUT=${RUN_ROOT}/sfh_recent_preserved_independent_past_150k.h5

cd "${REPO}"
[[ -f "${SOURCE}" ]] || { echo "Missing source SFHs: ${SOURCE}" >&2; exit 2; }
if [[ -e "${OUTPUT}" || -e "${OUTPUT}.partial" ]]; then
    echo "Refusing to overwrite existing output or partial output: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}" "${BASE}/logs"

echo "Repository commit: $(git rev-parse HEAD)"
python -u -m euclid.precompute_past_shape_ablation_sfhs \
    --source "${SOURCE}" --output "${OUTPUT}" \
    --seed 42 --recent-fraction 0.1 --transition-bins 8 \
    --integral-tolerance 2e-6 --chunk-size 512

python - "${OUTPUT%.h5}.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['all_past_donors_distinct_from_target']
assert report['maximum_absolute_stored_integral_error'] <= report['integral_tolerance']
assert report['maximum_recent_segment_error'] <= report['integral_tolerance']
assert report['recent_fraction_maximum_error'] <= report['integral_tolerance']
assert report['target_ablated_recent_fraction_correlation'] > 0.999999
print(json.dumps(report, indent=2))
print('Past-shape ablation validation: OK')
PY
