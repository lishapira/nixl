#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two-node NVLink fault-tolerance (elastic contraction) test launcher.
#
# Runs elastic.py on two nodes (4 GPUs each, 8 ranks total). Node 0 hosts
# the rank server + TCPStore; node 1 connects via --tcp-server. Intra-node
# peers use NVLink (cuda_ipc); cross-node peers use UCX.
#
# Requires an active 2-node SLURM allocation (or explicit MASTER_NODE /
# WORKER_NODE). Inside the GB300 container, source setup_node.sh first or
# let this script source it on each node.
#
# Example (inside salloc with 2 nodes):
#   bash run_nvlink_fault_tolerance_2node_test.sh
#   bash run_nvlink_fault_tolerance_2node_test.sh \
#       --fault-kill-timing before-dispatch
#
# Quick smoke (one CPU-level timing):
#   TIMINGS=before-dispatch bash run_nvlink_fault_tolerance_2node_sweep.sh
#
# NOTE: rebuild NIXL from the nvlink_fault_tolerance branch before running;
# the in-kernel marker APIs must be present in the installed nixl_ep module.

set -euo pipefail

TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "${TEST_DIR}"
# Path seen inside the GB300 container (lustre is mounted at /workspace/lishapira).
REMOTE_TEST_DIR=${REMOTE_TEST_DIR:-/workspace/lishapira/nixl/examples/device/ep/tests/elastic}

# ---------------------------------------------------------------------------
# Node selection (scontrol may be absent inside the container)
# ---------------------------------------------------------------------------
parse_nodelist() {
    local list="$1"
    if [[ -z "${list}" ]]; then
        return 1
    fi
    if command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "${list}"
        return 0
    fi
    # Fallback: expand theia[0058,0061] style lists without scontrol.
    python3 - "${list}" <<'PY'
import sys, re
spec = sys.argv[1]
out = []
for part in spec.split(","):
    m = re.match(r"^(.+)\[(.+)\]$", part)
    if m:
        prefix, inner = m.group(1), m.group(2)
        for item in inner.split(","):
            item = item.strip()
            if "-" in item:
                a, b = item.split("-", 1)
                width = max(len(a), len(b))
                for i in range(int(a), int(b) + 1):
                    out.append(f"{prefix}{i:0{width}d}")
            else:
                out.append(f"{prefix}{item}")
    else:
        out.append(part)
print("\n".join(out))
PY
}

if [[ -n "${MASTER_NODE:-}" && -n "${WORKER_NODE:-}" ]]; then
    NODES=("${MASTER_NODE}" "${WORKER_NODE}")
elif [[ -n "${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}" ]]; then
    mapfile -t NODES < <(parse_nodelist "${SLURM_NODELIST:-${SLURM_JOB_NODELIST}}")
else
    echo "error: need SLURM_NODELIST, SLURM_JOB_NODELIST, or MASTER_NODE+WORKER_NODE" >&2
    exit 2
fi

if (( ${#NODES[@]} < 2 )); then
    echo "error: need at least 2 nodes, got ${#NODES[@]} (${NODES[*]:-none})" >&2
    exit 2
fi

MASTER="${NODES[0]}"
WORKER="${NODES[1]}"
NUM_PROCESSES=${NUM_PROCESSES:-4}
PLAN_FILE=${PLAN_FILE:-nvlink_fault_tolerance_2node.json}
FAULT_KILL_SIGNAL=${FAULT_KILL_SIGNAL:-sigkill}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}
SETTLE_SECONDS=${SETTLE_SECONDS:-5}
MEM_LEAK_MIB=${MEM_LEAK_MIB:-64}
STRICT_CLEANUP=${STRICT_CLEANUP:-0}

# srun wrapper: reuse the caller's container/mount settings when present.
# When srun is unavailable (e.g. already inside the container shell), fall back
# to ssh for cross-node launch.
SRUN=(srun)
USE_SRUN=1
if ! command -v srun >/dev/null 2>&1; then
    USE_SRUN=0
fi
if (( USE_SRUN == 1 )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SRUN+=(--overlap)
fi
if (( USE_SRUN == 1 )) && [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    SRUN+=(--container-image="${CONTAINER_IMAGE}")
fi
if (( USE_SRUN == 1 )) && [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
    SRUN+=(--container-mounts="${CONTAINER_MOUNTS}")
    SRUN+=(--container-workdir=/workspace/lishapira)
fi

launch_on_node() {
    local node="$1"
    local cmd="$2"
    if (( USE_SRUN == 1 )); then
        timeout "${RUN_TIMEOUT}" "${SRUN[@]}" --nodes=1 --nodelist="${node}" \
            bash -lc "${cmd}"
    else
        timeout "${RUN_TIMEOUT}" ssh -o BatchMode=yes -o StrictHostKeyChecking=no \
            "${node}" bash -lc "${cmd}"
    fi
}

COMMON_ARGS=(
    --plan "${PLAN_FILE}"
    --num-processes "${NUM_PROCESSES}"
    --fault-kill-signal "${FAULT_KILL_SIGNAL}"
)
# Translate the host-side lustre path used by the launcher into the path the
# container sees (lustre is bind-mounted at /workspace/lishapira). The python
# helper that writes in-kernel evidence files runs inside the container, so
# without translation os.makedirs() fails on the lustre absolute path that
# doesn't exist inside the container mount namespace.
LUSTRE_HOST_PREFIX="${LUSTRE_HOST_PREFIX:-/lustre/fsw/network_research_advdev/lishapira}"
LUSTRE_CONTAINER_PREFIX="${LUSTRE_CONTAINER_PREFIX:-/workspace/lishapira}"
if [[ -n "${FAULT_EVIDENCE_DIR:-}" ]]; then
    REMOTE_EVIDENCE_DIR="${FAULT_EVIDENCE_DIR/#${LUSTRE_HOST_PREFIX}/${LUSTRE_CONTAINER_PREFIX}}"
    COMMON_ARGS+=(--fault-evidence-dir "${REMOTE_EVIDENCE_DIR}")
fi

# Remote command executed on each node. Sources setup_node.sh when available
# so PYTHONPATH/LD_LIBRARY_PATH point at the lustre-built NIXL tree.
read -r -d '' REMOTE_BODY <<'EOS' || true
set +e
if [[ -f /workspace/lishapira/setup_node.sh ]]; then
    # shellcheck disable=SC1091
    source /workspace/lishapira/setup_node.sh >/dev/null 2>&1
elif [[ -f /workspace/lishapira/build_nixl_aarch64.sh ]]; then
    # shellcheck disable=SC1091
    source /workspace/lishapira/build_nixl_aarch64.sh env >/dev/null 2>&1
    cd /workspace/lishapira/nixl 2>/dev/null || true
fi
unset UCX_TLS
cd TESTDIR_PLACEHOLDER
exec python3 -u elastic.py ELASTIC_ARGS_PLACEHOLDER
EOS

build_remote_cmd() {
    local role="$1"
    local tcp_arg="$2"
    shift 2
    local extra=("$@")
    local args_str
    printf -v args_str '%q ' "${COMMON_ARGS[@]}" "${extra[@]}"
    [[ -n "${tcp_arg}" ]] && args_str+=" ${tcp_arg}"
    args_str=${args_str%% }
    local body="${REMOTE_BODY//TESTDIR_PLACEHOLDER/${REMOTE_TEST_DIR}}"
    body="${body//ELASTIC_ARGS_PLACEHOLDER/${args_str}}"
    printf '%s\n' "${body}"
}

log_line() {
    local role="$1"
    local node="$2"
    local msg="$3"
    printf '%s [%s:%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%S)" "${role}" "${node}" "${msg}"
}

# ---------------------------------------------------------------------------
# Cleanup helpers (run locally on the launcher host after both nodes finish)
# ---------------------------------------------------------------------------
gpu_mem_used_mib() {
    { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true; } \
        | awk 'BEGIN{s=0} {s+=$1} END{print s+0}'
}
leftover_proc_count() {
    { pgrep -af 'elastic\.py|rank_server|spawn_main|torch.multiprocessing' 2>/dev/null || true; } \
        | wc -l
}
shm_leak_count() {
    { ls /dev/shm/torch_* /dev/shm/cuda.shm.* 2>/dev/null || true; } | wc -l
}

echo "[$(hostname -s)] 2-node run: master=${MASTER} worker=${WORKER} plan=${PLAN_FILE}"

# Snapshot pre-state on master (best-effort; launcher may not have GPUs).
PRE_GPU_MIB=$(gpu_mem_used_mib)
PRE_PROCS=$(leftover_proc_count)
PRE_SHM=$(shm_leak_count)

MASTER_LOG=$(mktemp)
WORKER_LOG=$(mktemp)
trap 'rm -f "${MASTER_LOG}" "${WORKER_LOG}"' EXIT

log_line master "${MASTER}" "starting elastic.py (rank server + 4 workers)"
MASTER_CMD=$(build_remote_cmd master "" "$@")
{
    log_line master "${MASTER}" "===== command ====="
    log_line master "${MASTER}" "python3 elastic.py ${COMMON_ARGS[*]} $*"
    log_line master "${MASTER}" "===== output ====="
    launch_on_node "${MASTER}" "${MASTER_CMD}" 2>&1 | while IFS= read -r line; do
            log_line master "${MASTER}" "${line}"
        done
} > "${MASTER_LOG}" 2>&1 &
MASTER_PID=$!

# Give the rank server a moment to bind; both nodes should start phase 0
# concurrently so all 8 ranks are present from the first dispatch.
sleep 2

log_line worker "${WORKER}" "starting elastic.py (--tcp-server ${MASTER})"
WORKER_CMD=$(build_remote_cmd worker "--tcp-server ${MASTER}" "$@")
{
    log_line worker "${WORKER}" "===== command ====="
    log_line worker "${WORKER}" "python3 elastic.py ${COMMON_ARGS[*]} --tcp-server ${MASTER} $*"
    log_line worker "${WORKER}" "===== output ====="
    launch_on_node "${WORKER}" "${WORKER_CMD}" 2>&1 | while IFS= read -r line; do
            log_line worker "${WORKER}" "${line}"
        done
} > "${WORKER_LOG}" 2>&1 &
WORKER_PID=$!

set +e
wait "${MASTER_PID}"
MASTER_RC=$?
wait "${WORKER_PID}"
WORKER_RC=$?
set -e

cat "${MASTER_LOG}"
cat "${WORKER_LOG}"

# Either side may exit 0 when the other handles the fault; fail only if BOTH
# sides report a non-zero exit (excluding expected SIGKILL on the victim).
TEST_RC=0
if (( MASTER_RC != 0 && WORKER_RC != 0 )); then
    TEST_RC=1
elif (( MASTER_RC != 0 && MASTER_RC != 137 && MASTER_RC != 143 )); then
    TEST_RC="${MASTER_RC}"
elif (( WORKER_RC != 0 && WORKER_RC != 137 && WORKER_RC != 143 )); then
    TEST_RC="${WORKER_RC}"
fi

sleep "${SETTLE_SECONDS}"

POST_GPU_MIB=$(gpu_mem_used_mib)
POST_PROCS=$(leftover_proc_count)
POST_SHM=$(shm_leak_count)
GPU_DELTA=$(( POST_GPU_MIB - PRE_GPU_MIB ))

issues=()
if (( POST_PROCS > PRE_PROCS )); then
    issues+=("leftover_procs:${PRE_PROCS}->${POST_PROCS}")
fi
if (( POST_SHM > PRE_SHM )); then
    issues+=("shm_leak:${PRE_SHM}->${POST_SHM}")
fi
if (( GPU_DELTA > MEM_LEAK_MIB )); then
    issues+=("gpu_mem_delta:${GPU_DELTA}MiB>threshold:${MEM_LEAK_MIB}MiB")
fi

printf '\n===================== CLEANUP REPORT =====================\n'
printf 'master=%s worker=%s master_rc=%s worker_rc=%s test_rc=%s\n' \
    "${MASTER}" "${WORKER}" "${MASTER_RC}" "${WORKER_RC}" "${TEST_RC}"
printf 'leftover_procs:        pre=%s post=%s\n' "${PRE_PROCS}" "${POST_PROCS}"
printf 'shm_torch_files:       pre=%s post=%s\n' "${PRE_SHM}" "${POST_SHM}"
printf 'gpu_mem_used_mib:      pre=%s post=%s delta=%s threshold=%s\n' \
    "${PRE_GPU_MIB}" "${POST_GPU_MIB}" "${GPU_DELTA}" "${MEM_LEAK_MIB}"
if (( ${#issues[@]} == 0 )); then
    printf 'cleanup_result:        CLEAN\n'
else
    printf 'cleanup_result:        DIRTY (%s)\n' "${issues[*]}"
fi
printf '==========================================================\n'

if (( STRICT_CLEANUP == 1 && ${#issues[@]} > 0 && TEST_RC == 0 )); then
    echo "STRICT_CLEANUP=1: escalating dirty cleanup to exit 2" >&2
    exit 2
fi
exit "${TEST_RC}"
