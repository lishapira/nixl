#!/bin/bash
# Unprivileged elastic.py callback. It requests, but cannot perform, injection.
set -euo pipefail

gpu=${1:?gpu minor is required}
mode=${2:?mode is required}
link=${3:?link is required}
run_id=${NIXL_FAULT_RUN_ID:?NIXL_FAULT_RUN_ID is required}
runtime_token=${NIXL_FAULT_RUNTIME_TOKEN:?NIXL_FAULT_RUNTIME_TOKEN is required}
request_root=${NIXL_FAULT_REQUEST_ROOT:?NIXL_FAULT_REQUEST_ROOT is required}
expected_gpu=${NIXL_FAULT_GPU:-2}
expected_link=${NIXL_FAULT_LINK:-0}

if [[ "${gpu}" != "${expected_gpu}" || "${mode}" != down ||
      "${link}" != "${expected_link}" ]]; then
    echo "Refusing unexpected request: gpu=${gpu} mode=${mode} link=${link}" >&2
    exit 2
fi

if [[ ! "${runtime_token}" =~ ^[0-9a-f]{16}$ ]]; then
    echo "Refusing invalid runtime token" >&2
    exit 2
fi

request_dir="${request_root}/${run_id}_${runtime_token}"
request="${request_dir}/request"
tmp="${request}.tmp.$$"
mkdir -p "${request_dir}"
printf 'run_id=%s\nruntime_token=%s\ngpu=%s\nmode=%s\nlink=%s\npid=%s\nhost=%s\nrequested_ns=%s\n' \
    "${run_id}" "${runtime_token}" "${gpu}" "${mode}" "${link}" "$$" "$(hostname)" "$(date +%s%N)" \
    > "${tmp}"
mv -f "${tmp}" "${request}"
echo "FAULT_REQUEST_WRITTEN path=${request} gpu=${gpu} link=${link} ns=$(date +%s%N)"
