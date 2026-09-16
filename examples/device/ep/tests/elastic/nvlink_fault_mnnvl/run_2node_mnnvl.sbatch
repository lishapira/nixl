#!/bin/bash
#SBATCH --job-name=nixl-ep-mnnvl-fault
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=mnnvl_fault_%j.log
#
# Destructive experiment. The workload cannot inject until separately started
# root helpers have captured pre-state and armed on both hosts.
set -euo pipefail

: "${NIXL_REPO_ROOT:?export the shared checkout path before sbatch}"
: "${NIXL_INSTALL:?export the shared NIXL install prefix before sbatch}"
: "${NIXL_EP_IMAGE:?export the node-local Pyxis/Enroot image path before sbatch}"

run_id=${SLURM_JOB_ID}
runtime_token=$(python3 -c 'import secrets; print(secrets.token_hex(8))')
experiment_dir="${NIXL_REPO_ROOT}/examples/device/ep/tests/elastic/nvlink_fault_mnnvl"
results_root=${NIXL_FAULT_RESULTS_ROOT:-${experiment_dir}/results}
results="${results_root}/run_${run_id}_${runtime_token}"
local_root=${NIXL_LOCAL_ROOT:-/raid/users/${SLURM_JOB_USER}}
request_root=${NIXL_FAULT_REQUEST_ROOT:-${local_root}/nixl_nvlink_fault}
plan="${experiment_dir}/two_node_fault_rank6.json"
request_tool="${experiment_dir}/request_nvlink_fault.sh"
injector="${experiment_dir}/nvlink_hwinject"
build_dir=${NIXL_BUILD_DIR:-nixl_build}
num_tokens=${NUM_TOKENS:-2048}
join_delay=${JOIN_DELAY:-30}

mapfile -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#hosts[@]} -ne 2 ]]; then
    echo "FATAL: expected exactly two nodes" >&2
    exit 1
fi
peer_host=${NIXL_FAULT_PEER_HOST:-${hosts[0]}}
victim_host=${NIXL_FAULT_VICTIM_HOST:-${hosts[1]}}
if [[ "${peer_host}" != "${hosts[0]}" || "${victim_host}" != "${hosts[1]}" ]]; then
    echo "FATAL: rank 6 is on node index 1; expected peer=${hosts[0]} victim=${hosts[1]}" >&2
    exit 1
fi

mkdir -p "${results}"
# Shared NFS is root-squashed on the tested cluster. Root helpers therefore
# create evidence as nobody and need this run directory writable.
chmod 0777 "${results}"
rm -f "${results}"/{controller.armed,peer_capture.armed,controller.done,peer_capture.done,injection.started,injector.rc}

echo "RUN_ID=${run_id}"
echo "RUNTIME_TOKEN=${runtime_token}"
echo "PEER_HOST=${peer_host} VICTIM_HOST=${victim_host}"
echo "RESULTS=${results}"

# Both the image and node-local mount root must exist on both hosts.
srun --overlap --ntasks=2 --ntasks-per-node=1 bash -c '
set -e
test -r "'"${NIXL_EP_IMAGE}"'"
test -d "'"${local_root}"'"
'
srun --overlap -w "${victim_host}" --ntasks=1 bash -c \
    "mkdir -p '${request_root}/${run_id}_${runtime_token}' &&
     chmod 0777 '${request_root}/${run_id}_${runtime_token}' &&
     rm -f '${request_root}/${run_id}_${runtime_token}/request'"

# Immutable preflight evidence.
srun --overlap --ntasks=2 --ntasks-per-node=1 --gres=gpu:4 bash -c '
out="'"${results}"'/pre_$(hostname).log"
{
  echo "host=$(hostname) ns=$(date +%s%N)"
  cat /proc/driver/nvidia/version
  sha256sum "'"${injector}"'"
  nvidia-smi --query-gpu=index,gpu_uuid,name,driver_version --format=csv
  for gpu in 0 1 2 3; do
    links=$(nvidia-smi nvlink -s -i "${gpu}" | grep -c "GB/s")
    recovery=$(nvidia-smi -q -i "${gpu}" |
      grep "GPU Recovery Action" | awk -F": " "{print \$2}" | tr -d " ")
    printf "GPU%s_links=%s recovery=%s\n" "${gpu}" "${links}" "${recovery}"
    [[ "${links}" == 18 && "${recovery}" == None ]] || {
      echo "FATAL: GPU${gpu} is not healthy before injection"
      exit 1
    }
  done
  nvidia-smi -q -i 0 | grep -iE "Recovery Action|State *:|CliqueId"
} > "${out}" 2>&1
'

cat <<EOF

Arm these in root shells inside this allocation:

  ${victim_host}:
    NIXL_FAULT_VICTIM_HOST=${victim_host} ${experiment_dir}/root_fault_controller.sh ${run_id} ${runtime_token} ${results} ${request_root} 20

  ${peer_host}:
    NIXL_FAULT_PEER_HOST=${peer_host} ${experiment_dir}/root_peer_capture.sh ${run_id} ${runtime_token} ${results}

No injection occurs before both root-owned node-local markers exist.
EOF

touch "${results}/workload.waiting_for_root"
export run_id runtime_token peer_host victim_host
export arm_timeout=${ROOT_ARM_TIMEOUT_SEC:-10800}
srun --overlap --ntasks=2 --ntasks-per-node=1 bash -c '
host=$(hostname -s)
case "${host}" in
  "${victim_host}") role=controller ;;
  "${peer_host}") role=peer_capture ;;
  *) echo "FATAL: unexpected host ${host}"; exit 1 ;;
esac
marker="/run/nixl_ep_nvlink_fault/${run_id}_${runtime_token}/${role}.armed"
deadline=$(( $(date +%s) + arm_timeout ))
until [[ -f "${marker}" ]]; do
  [[ $(date +%s) -lt ${deadline} ]] || {
    echo "FATAL: timed out waiting for ${marker}"
    exit 1
  }
  sleep 1
done
echo "${host}: root helper armed"
'

# Capture process ownership and utilization around the injection.
srun --overlap --ntasks=2 --ntasks-per-node=1 --gres=gpu:4 bash -c '
out="'"${results}"'/sampler_$(hostname).log"
for _ in $(seq 1 180); do
  echo "=== ns=$(date +%s%N) ==="
  nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader
  nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv,noheader
  sleep 1
done > "${out}" 2>&1
' &
sampler_step=$!

master_ip=$(getent hosts "${peer_host}" | awk '{print $1}' | head -1)
export run_id results request_root plan request_tool num_tokens join_delay master_ip
export NIXL_REPO_ROOT NIXL_INSTALL NIXL_EP_IMAGE build_dir local_root
export UCX_TLS=${UCX_TLS:-^rc_gda}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-^mlx5_16:1}
export NIXL_FAULT_GPU=2
export NIXL_FAULT_LINK=0

set +e
srun --overlap --ntasks=2 --ntasks-per-node=1 --gres=gpu:4 \
  --container-image="${NIXL_EP_IMAGE}" \
  --container-mounts="${NIXL_REPO_ROOT}:${NIXL_REPO_ROOT},${NIXL_INSTALL}:${NIXL_INSTALL},${local_root}:${local_root},/dev/infiniband:/dev/infiniband" \
  bash -c '
set -o pipefail
arch=$(uname -m)
ep=examples/device/ep
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${NIXL_INSTALL}/lib/${arch}-linux-gnu:${NIXL_INSTALL}/lib/${arch}-linux-gnu/plugins:/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH}"
export NIXL_PLUGIN_DIR="${NIXL_INSTALL}/lib/${arch}-linux-gnu/plugins"
export NIXL_PREFIX="${NIXL_INSTALL}"
export PYTHONPATH="${NIXL_REPO_ROOT}/src/bindings/python/nixl-meta:${NIXL_REPO_ROOT}/${build_dir}/${ep}:${NIXL_REPO_ROOT}/${ep}/tests:${NIXL_REPO_ROOT}/${ep}/tests/elastic"
export NIXL_FAULT_RUN_ID="${run_id}"
export NIXL_FAULT_RUNTIME_TOKEN="${runtime_token}"
export NIXL_FAULT_REQUEST_ROOT="${request_root}"
node_log="${results}/app_$(hostname).log"
common="--plan ${plan} --num-processes 4 --num-experts-per-rank 32 --num-topk 8 --num-tokens ${num_tokens} --timeout-ms 10000 --validate-phase-failures --fault-nvlink --fault-nvlink-link 0 --fault-nvlink-tool ${request_tool}"
cd "${NIXL_REPO_ROOT}"
echo "APP_START host=$(hostname) nodeid=${SLURM_NODEID} ns=$(date +%s%N)" | tee "${node_log}"
if [[ "${SLURM_NODEID}" == 0 ]]; then
    python3 "${ep}/tests/elastic/elastic.py" ${common} 2>&1 | tee -a "${node_log}"
else
    python3 - "${master_ip}" <<'"'"'PY'"'"' || exit 1
import socket
import sys
import time
deadline = time.time() + 180
while time.time() < deadline:
    try:
        socket.create_connection((sys.argv[1], 9999), 3).close()
        raise SystemExit(0)
    except OSError:
        time.sleep(2)
raise SystemExit(1)
PY
    sleep "${join_delay}"
    python3 "${ep}/tests/elastic/elastic.py" ${common} --tcp-server "${master_ip}" 2>&1 | tee -a "${node_log}"
fi
rc=${PIPESTATUS[0]}
echo "APP_END host=$(hostname) rc=${rc} ns=$(date +%s%N)" | tee -a "${node_log}"
exit "${rc}"
'
workload_rc=$?
set -e

printf '%s\n' "${workload_rc}" > "${results}/workload.rc"
echo "WORKLOAD_RC=${workload_rc} (nonzero is expected after the real fault)"
kill "${sampler_step}" 2>/dev/null || true
wait "${sampler_step}" 2>/dev/null || true

export capture_timeout=${ROOT_CAPTURE_TIMEOUT_SEC:-90}
srun --overlap --ntasks=2 --ntasks-per-node=1 bash -c '
host=$(hostname -s)
case "${host}" in
  "${victim_host}") role=controller ;;
  "${peer_host}") role=peer_capture ;;
  *) echo "FATAL: unexpected host ${host}"; exit 1 ;;
esac
marker="/run/nixl_ep_nvlink_fault/${run_id}_${runtime_token}/${role}.done"
deadline=$(( $(date +%s) + capture_timeout ))
until [[ -f "${marker}" ]]; do
  [[ $(date +%s) -lt ${deadline} ]] || {
    echo "FATAL: timed out waiting for ${marker}"
    exit 1
  }
  sleep 1
done
echo "${host}: root capture complete"
'

injector_rc=$(<"${results}/injector.rc")
if [[ "${injector_rc}" != 0 ]]; then
    echo "FATAL: injection failed with rc=${injector_rc}" >&2
    exit 1
fi

echo "Artifacts: ${results}"
echo "Review evidence, then cold-power-cycle only the victim node."
exit 0
