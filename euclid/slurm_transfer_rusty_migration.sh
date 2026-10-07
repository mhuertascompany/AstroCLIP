#!/bin/bash
#SBATCH --job-name=send_rusty
#SBATCH --partition=pscomp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/n03data/huertas/euclid/send_rusty_%j.out
#SBATCH --error=/n03data/huertas/euclid/send_rusty_%j.err

set -euo pipefail

# Transfer the packaged Euclid workspace to Rusty and verify it remotely.
# Noninteractive `ssh rusty true` must work from the allocated Candide node.
# Optional arguments:
#   1 local archive directory
#   2 SSH host or alias
#   3 remote archive directory (default is the remote user's Ceph directory)

SOURCE=${1:-/n03data/huertas/euclid/rusty_migration}
REMOTE_HOST=${2:-rusty}

BRIGHT=edfn_vislt22p0_150000.tar
LEGACY=edfn_100k.tar
CHECKSUMS=SHA256SUMS

for name in "${BRIGHT}" "${LEGACY}" "${CHECKSUMS}"; do
    [[ -f "${SOURCE}/${name}" ]] || {
        echo "Missing migration file: ${SOURCE}/${name}" >&2
        exit 2
    }
done

SSH_OPTIONS=(
    -o BatchMode=yes
    -o ConnectTimeout=30
    -o ServerAliveInterval=60
    -o ServerAliveCountMax=5
)

echo "Testing noninteractive access to ${REMOTE_HOST} from $(hostname)"
if ! REMOTE_USER=$(ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" 'id -un'); then
    cat >&2 <<EOF
Noninteractive SSH to ${REMOTE_HOST} failed from this compute node.
Confirm that the SSH alias and key work without a password or verification
prompt in a small Slurm allocation before resubmitting.
EOF
    exit 2
fi

REMOTE_ROOT=${3:-/mnt/ceph/users/${REMOTE_USER}/euclid/archives}
echo "Remote destination: ${REMOTE_HOST}:${REMOTE_ROOT}"

echo "Verifying local archives"
(
    cd "${SOURCE}"
    sha256sum -c "${CHECKSUMS}"
)

ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}'"

# --append-verify resumes interrupted large-file transfers and verifies the
# already transferred prefix before appending. The final SHA256 check protects
# the complete archive contents independently of rsync's transport checks.
RSYNC_SSH='ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=60 -o ServerAliveCountMax=5'
rsync -a \
    --human-readable \
    --info=progress2,stats2 \
    --partial \
    --append-verify \
    --timeout=600 \
    -e "${RSYNC_SSH}" \
    "${SOURCE}/${BRIGHT}" \
    "${SOURCE}/${LEGACY}" \
    "${SOURCE}/${CHECKSUMS}" \
    "${REMOTE_HOST}:${REMOTE_ROOT}/"

echo "Verifying transferred archives on Rusty"
ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "cd '${REMOTE_ROOT}' && sha256sum -c '${CHECKSUMS}'"

echo "Rusty migration transfer completed and verified."
