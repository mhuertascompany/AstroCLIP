#!/bin/bash
#SBATCH --job-name=pack_rusty
#SBATCH --partition=pscomp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/n03data/huertas/euclid/pack_rusty_%j.out
#SBATCH --error=/n03data/huertas/euclid/pack_rusty_%j.err

set -euo pipefail

# Package the complete Euclid training workspace for transfer to Rusty.
# Optional arguments:
#   1 source directory containing edfn_vislt22p0_150000 and edfn_100k
#   2 output directory for the archives

SOURCE=${1:-/n03data/huertas/euclid/sfh_clip}
MIGRATION=${2:-/n03data/huertas/euclid/rusty_migration}

BRIGHT=edfn_vislt22p0_150000
LEGACY=edfn_100k
BRIGHT_SOURCE=${SOURCE}/${BRIGHT}
LEGACY_SOURCE=${SOURCE}/${LEGACY}
BRIGHT_TAR=${MIGRATION}/${BRIGHT}.tar
LEGACY_TAR=${MIGRATION}/${LEGACY}.tar
CHECKSUMS=${MIGRATION}/SHA256SUMS

for path in "${BRIGHT_SOURCE}" "${LEGACY_SOURCE}"; do
    [[ -d "${path}" ]] || { echo "Missing source directory: ${path}" >&2; exit 2; }
done
mkdir -p "${MIGRATION}"

if command -v sha256sum >/dev/null 2>&1; then
    SHA256=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
    SHA256=(shasum -a 256)
else
    echo "Neither sha256sum nor shasum is available." >&2
    exit 2
fi

archive_is_reusable() {
    local archive=$1
    if [[ ! -e "${archive}" ]]; then
        return 1
    fi
    echo "Validating existing archive: ${archive}"
    if ! tar -tf "${archive}" >/dev/null; then
        echo "Existing archive is incomplete or invalid: ${archive}" >&2
        echo "Move it aside or remove it before resubmitting." >&2
        exit 2
    fi
    echo "Reusing complete archive: ${archive}"
    return 0
}

REUSE_BRIGHT=false
REUSE_LEGACY=false
if archive_is_reusable "${BRIGHT_TAR}"; then REUSE_BRIGHT=true; fi
if archive_is_reusable "${LEGACY_TAR}"; then REUSE_LEGACY=true; fi

BRIGHT_TMP=${BRIGHT_TAR}.partial.${SLURM_JOB_ID}
LEGACY_TMP=${LEGACY_TAR}.partial.${SLURM_JOB_ID}
CHECKSUMS_TMP=${CHECKSUMS}.partial.${SLURM_JOB_ID}
cleanup() {
    rm -f "${BRIGHT_TMP}" "${LEGACY_TMP}" "${CHECKSUMS_TMP}"
}
trap cleanup EXIT INT TERM

# Uncompressed tar files need approximately the apparent source size plus tar
# headers. Count only archives that still need to be built, then require a 5%
# margin and an additional GiB before doing any work.
SOURCE_BYTES=0
if [[ "${REUSE_BRIGHT}" == false ]]; then
    BYTES=$(du -sk "${BRIGHT_SOURCE}" | awk '{print $1 * 1024}')
    SOURCE_BYTES=$((SOURCE_BYTES + BYTES))
fi
if [[ "${REUSE_LEGACY}" == false ]]; then
    BYTES=$(du -sk "${LEGACY_SOURCE}" | awk '{print $1 * 1024}')
    SOURCE_BYTES=$((SOURCE_BYTES + BYTES))
fi
AVAILABLE_BYTES=$(df -Pk "${MIGRATION}" | awk 'NR == 2 {print $4 * 1024}')
REQUIRED_BYTES=$((SOURCE_BYTES + SOURCE_BYTES / 20 + 1073741824))
printf 'Source apparent size: %s bytes\n' "${SOURCE_BYTES}"
printf 'Required with margin: %s bytes\n' "${REQUIRED_BYTES}"
printf 'Available at destination: %s bytes\n' "${AVAILABLE_BYTES}"
if (( AVAILABLE_BYTES < REQUIRED_BYTES )); then
    echo "Insufficient free space in ${MIGRATION}." >&2
    exit 2
fi

if [[ "${REUSE_BRIGHT}" == false ]]; then
    echo "Packaging ${BRIGHT_SOURCE}"
    tar -cf "${BRIGHT_TMP}" -C "${SOURCE}" "${BRIGHT}"
fi

if [[ "${REUSE_LEGACY}" == false ]]; then
    echo "Packaging ${LEGACY_SOURCE}"
    tar -cf "${LEGACY_TMP}" -C "${SOURCE}" "${LEGACY}"
fi

if [[ "${REUSE_BRIGHT}" == false ]]; then
    mv "${BRIGHT_TMP}" "${BRIGHT_TAR}"
fi
if [[ "${REUSE_LEGACY}" == false ]]; then
    mv "${LEGACY_TMP}" "${LEGACY_TAR}"
fi

# Store relative filenames so the checksum manifest works after transfer.
(
    cd "${MIGRATION}"
    "${SHA256[@]}" "${BRIGHT}.tar" "${LEGACY}.tar"
) > "${CHECKSUMS_TMP}"
mv "${CHECKSUMS_TMP}" "${CHECKSUMS}"

trap - EXIT INT TERM
echo "Rusty migration archives are ready:"
ls -lh "${BRIGHT_TAR}" "${LEGACY_TAR}" "${CHECKSUMS}"
echo
cat "${CHECKSUMS}"
