#!/bin/bash
#SBATCH --job-name=sfh_factor_null
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --account=other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/sfh_factor_null_%j.out
#SBATCH --error=/mnt/home/mhuertas/ceph/euclid/sfh_clip/logs/sfh_factor_null_%j.err

set -euo pipefail
module load python
VENV="${RUSTY_VENV:-${HOME}/.sfh_euclid}"
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/mnt/home/mhuertas/python/AstroCLIP
BASE=/mnt/home/mhuertas/ceph/euclid/sfh_clip
SOURCE=${BASE}/archives/edfn_vislt22p0_150000/sfh_clip_150k.h5
RUN_ROOT=${BASE}/runs/factorized_null
OUTPUT=${RUN_ROOT}/sfh_factorized_null_150k.h5

cd "${REPO}"
[[ -f "${SOURCE}" ]] || { echo "Missing source SFHs: ${SOURCE}" >&2; exit 2; }
if [[ -e "${OUTPUT}" || -e "${OUTPUT}.partial" ]]; then
    echo "Refusing to overwrite existing output or partial output: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}" "${BASE}/logs"

echo "Repository commit: $(git rev-parse HEAD)"
echo "Source: ${SOURCE}"
echo "Output: ${OUTPUT}"
python -u -m euclid.precompute_factorized_null_sfhs \
    --source "${SOURCE}" --output "${OUTPUT}" \
    --seed 42 --recent-fraction 0.1 --transition-bins 8 \
    --integral-tolerance 2e-6 --chunk-size 512

python - "${OUTPUT%.h5}.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['all_donors_distinct_from_target_and_each_other']
for key in (
    'target_null_correlation_recent_fraction',
    'target_null_correlation_mean_lookback',
    'target_null_correlation_t50',
):
    value = abs(float(report[key]))
    print(f'{key}: {value:.6f}')
    if value > 0.03:
        raise SystemExit(f'Null validation failed: |{key}|={value:.6f} > 0.03')
print('Factorized-null validation: OK')
PY
