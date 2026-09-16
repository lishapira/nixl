#!/bin/bash
# Root-only peer-host capture around injection on the victim host.
set -euo pipefail

run_id=${1:?usage: root_peer_capture.sh RUN_ID RUNTIME_TOKEN RESULTS_DIR}
runtime_token=${2:?runtime token is required}
results=${3:?results directory is required}
peer_host=${NIXL_FAULT_PEER_HOST:-gb-wlake-17}
started="${results}/injection.started"
log="${results}/root_capture_${peer_host}.log"
runtime_dir="/run/nixl_ep_nvlink_fault/${run_id}_${runtime_token}"

if [[ $(id -u) -ne 0 || $(hostname -s) != "${peer_host}" ]]; then
    echo "FATAL: run as real root on ${peer_host}" >&2
    exit 1
fi
if [[ ! "${run_id}" =~ ^[0-9]+$ || ! "${runtime_token}" =~ ^[0-9a-f]{16}$ ]]; then
    echo "FATAL: invalid run id or runtime token" >&2
    exit 1
fi
if [[ ! -d "${results}" || ! -w "${results}" ]]; then
    echo "FATAL: results directory must already be writable: ${results}" >&2
    exit 1
fi

install -d -o root -g root -m 0755 "${runtime_dir}"
exec > >(tee -a "${log}") 2>&1
echo "PEER_CAPTURE_START run_id=${run_id} ns=$(date +%s%N)"
dmesg -T > "${results}/dmesg_pre_${peer_host}.txt"
nvidia-smi -q > "${results}/smi_q_pre_${peer_host}.txt"
touch "${runtime_dir}/peer_capture.armed"
chmod 0444 "${runtime_dir}/peer_capture.armed"

python3 - "${started}" "${NIXL_FAULT_ARM_TIMEOUT_SEC:-10800}" <<'PY'
import os
import sys
import time

marker, timeout = sys.argv[1], int(sys.argv[2])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    if os.path.isfile(marker):
        raise SystemExit(0)
    time.sleep(0.01)
raise SystemExit("timed out waiting for injection.started")
PY

echo "PEER_CAPTURE_INJECTION_SEEN ns=$(date +%s%N)"
sleep 20
dmesg -T > "${results}/dmesg_post_${peer_host}.txt"
nvidia-smi -q > "${results}/smi_q_post_${peer_host}.txt" 2>&1 || true
for gpu in 0 1 2 3; do
    nvidia-smi nvlink -s -i "${gpu}" \
        > "${results}/nvlink_post_gpu${gpu}_${peer_host}.txt" 2>&1 || true
    nvidia-smi nvlink -e -i "${gpu}" \
        > "${results}/nvlink_errors_gpu${gpu}_${peer_host}.txt" 2>&1 || true
done
sync
touch "${results}/peer_capture.done"
touch "${runtime_dir}/peer_capture.done"
chmod 0444 "${runtime_dir}/peer_capture.done"
echo "PEER_CAPTURE_DONE ns=$(date +%s%N)"
