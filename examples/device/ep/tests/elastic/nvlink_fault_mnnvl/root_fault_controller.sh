#!/bin/bash
# Root-only victim-host controller. Never power-cycles automatically.
set -euo pipefail

run_id=${1:?usage: root_fault_controller.sh RUN_ID RUNTIME_TOKEN RESULTS_DIR REQUEST_ROOT [DELAY_MS]}
runtime_token=${2:?runtime token is required}
results=${3:?results directory is required}
request_root=${4:?node-local request root is required}
delay_ms=${5:-20}

dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_injector=${NIXL_FAULT_INJECTOR:-${dir}/nvlink_hwinject}
victim_host=${NIXL_FAULT_VICTIM_HOST:-gb-wlake-18}
gpu=${NIXL_FAULT_GPU:-2}
link=${NIXL_FAULT_LINK:-0}
request_dir="${request_root}/${run_id}_${runtime_token}"
request="${request_dir}/request"
log="${results}/root_controller_${victim_host}.log"
runtime_dir="/run/nixl_ep_nvlink_fault/${run_id}_${runtime_token}"
injector="${runtime_dir}/nvlink_hwinject"

if [[ $(id -u) -ne 0 ]]; then
    echo "FATAL: run as real host root" >&2
    exit 1
fi
if [[ $(hostname -s) != "${victim_host}" ]]; then
    echo "FATAL: expected ${victim_host}, got $(hostname -s)" >&2
    exit 1
fi
if [[ ! "${run_id}" =~ ^[0-9]+$ || ! "${runtime_token}" =~ ^[0-9a-f]{16}$ ]]; then
    echo "FATAL: invalid run id or runtime token" >&2
    exit 1
fi
if [[ ! -x "${source_injector}" ]]; then
    echo "FATAL: injector is not executable: ${source_injector}" >&2
    exit 1
fi
if [[ ! -d "${results}" || ! -w "${results}" ]]; then
    echo "FATAL: results directory must already be writable: ${results}" >&2
    exit 1
fi

mkdir -p "${request_dir}"
chmod 0777 "${request_dir}"
rm -f "${request}"
install -d -o root -g root -m 0755 "${runtime_dir}"
# Prevent replacement between the harmless probe and destructive ioctl.
install -o root -g root -m 0700 "${source_injector}" "${injector}"
exec > >(tee -a "${log}") 2>&1

echo "ROOT_CONTROLLER_START run_id=${run_id} gpu=${gpu} link=${link} ns=$(date +%s%N)"
sha256sum "${injector}"
file "${injector}"
nvidia-smi --query-gpu=index,gpu_uuid,name --format=csv
nvidia-smi -q -i "${gpu}" > "${results}/smi_q_pre_${victim_host}.txt"
nvidia-smi nvlink -s -i "${gpu}" > "${results}/nvlink_pre_gpu${gpu}_${victim_host}.txt"
dmesg -T > "${results}/dmesg_pre_${victim_host}.txt"

# Harmless ABI/permission probe: linkMask=0 touches no link.
"${injector}" "${gpu}"
touch "${runtime_dir}/controller.armed"
chmod 0444 "${runtime_dir}/controller.armed"
echo "ROOT_CONTROLLER_ARMED request=${request} ns=$(date +%s%N)"

python3 - "${request}" "${NIXL_FAULT_ARM_TIMEOUT_SEC:-10800}" <<'PY'
import os
import sys
import time

request, timeout = sys.argv[1], int(sys.argv[2])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    if os.path.isfile(request):
        raise SystemExit(0)
    time.sleep(0.001)
raise SystemExit("timed out waiting for fault request")
PY

cat "${request}"
grep -qx "run_id=${run_id}" "${request}"
grep -qx "runtime_token=${runtime_token}" "${request}"
grep -qx "gpu=${gpu}" "${request}"
grep -qx 'mode=down' "${request}"
grep -qx "link=${link}" "${request}"
grep -qx "host=${victim_host}" "${request}"

echo "FAULT_REQUEST_SEEN ns=$(date +%s%N)"
python3 - "${delay_ms}" <<'PY'
import sys
import time
time.sleep(int(sys.argv[1]) / 1000)
PY

echo "INJECTION_BEGIN gpu=${gpu} link=${link} ns=$(date +%s%N)"
touch "${results}/injection.started"
set +e
"${injector}" "${gpu}" down "${link}"
inject_rc=$?
set -e
echo "INJECTION_END rc=${inject_rc} ns=$(date +%s%N)"
printf '%s\n' "${inject_rc}" > "${results}/injector.rc"

sleep 15
dmesg -T > "${results}/dmesg_post_${victim_host}.txt" || true
nvidia-smi -q > "${results}/smi_q_post_${victim_host}.txt" 2>&1 || true
for index in 0 1 2 3; do
    nvidia-smi nvlink -s -i "${index}" \
        > "${results}/nvlink_post_gpu${index}_${victim_host}.txt" 2>&1 || true
    nvidia-smi nvlink -e -i "${index}" \
        > "${results}/nvlink_errors_gpu${index}_${victim_host}.txt" 2>&1 || true
done
sync
touch "${results}/controller.done"
touch "${runtime_dir}/controller.done"
chmod 0444 "${runtime_dir}/controller.done"
echo "ROOT_CONTROLLER_DONE ns=$(date +%s%N)"
exit "${inject_rc}"
