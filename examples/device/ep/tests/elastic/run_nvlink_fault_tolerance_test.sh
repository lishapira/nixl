#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launcher for the NVLink fault-tolerance (contraction) test on a SINGLE node
# with 4 GPUs. It runs the elastic EP test (elastic.py) with the 4-rank
# nvlink_fault_tolerance.json plan and ground-truth mask validation.
#
# Assumptions:
#   - 4 ranks total (1 node x 4 GPUs)
#   - All GPU peers are on the same node, so NVLink (CUDA IPC) is the natural
#     intra-node path; we leave UCX free to pick cuda_ipc for every peer.
#   - A killed rank must be detected by every surviving rank through the
#     runtime mask buffer; elastic.py builds an independent ground-truth mask
#     from the plan and asserts the runtime mask matches it.
#
# Everything runs locally in a single process tree: this script starts the rank
# server / TCPStore locally and spawns 4 python workers (one per GPU) via
# `--num-processes 4`. No multi-node / srun coordination is required.
#
# Example invocation (directly on a 4-GPU box, with a built+rebuilt NIXL EP
# install at /workspace/nixl/install):
#
#   bash /workspace/nixl/examples/device/ep/tests/elastic/run_nvlink_fault_tolerance_test.sh
#
# It also works under `srun --nodes=1 --ntasks-per-node=1 --gres=gpu:4 ...`,
# but SLURM is not required.
#
# NOTE: this tree (Lior's in-kernel SIGKILL marker commit) modifies the CUDA
# kernels and buffer.py, so the install MUST be rebuilt from this source before
# running, otherwise enable_in_kernel_fault_marker / the in-kernel timings will
# be missing.
#
# Any extra arguments after the script name are forwarded verbatim to
# elastic.py, e.g. to exercise an in-kernel kill window:
#   bash run_nvlink_fault_tolerance_test.sh \
#        --fault-kill-timing dispatch-send-during-kernel-cold
#   bash run_nvlink_fault_tolerance_test.sh \
#        --fault-kill-timing dispatch-send-during-kernel

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. NIXL EP install paths (override via env if your tree lives elsewhere)
# ---------------------------------------------------------------------------
export NIXL_INSTALL=${NIXL_INSTALL:-/workspace/nixl/install}

NIXL_EP_CPP=$(ls "${NIXL_INSTALL}"/lib/python3/dist-packages/nixl_ep/nixl_ep_cpp.cpython-*.so 2>/dev/null | head -n 1 || true)
if [[ -z "${NIXL_EP_CPP}" ]]; then
    echo "error: ${NIXL_INSTALL} does not contain a built nixl_ep module." >&2
    echo "       Build (and REBUILD for this commit's kernel changes) first:" >&2
    echo "         meson setup nixl_build --prefix=${NIXL_INSTALL} \\" >&2
    echo "           -Ducx_path=/opt/hpcx/ucx -Dbuild_docs=false -Drust=false \\" >&2
    echo "           -Dbuild_nixl_ep=true -Dlibfabric_path=/opt/amazon/efa \\" >&2
    echo "           --buildtype=release" >&2
    echo "         ninja -C nixl_build install" >&2
    exit 1
fi

export PYTHONPATH=${NIXL_INSTALL}/lib/python3/dist-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=${NIXL_INSTALL}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export NIXL_PLUGIN_DIR=${NIXL_INSTALL}/lib/x86_64-linux-gnu/plugins

# ---------------------------------------------------------------------------
# 2. NVLink (intra-node)
# ---------------------------------------------------------------------------
# Leave UCX free to pick cuda_ipc for every peer so the GPUs talk over NVLink.
# This is the OPPOSITE of run_test_ht_2x2.sh, which forces UCX onto a NIC via
# UCX_TLS=^cuda_ipc. We clear any inherited UCX_TLS restriction, and the python
# harness leaves --disable-ll-nvlink off by default, so the low-latency kernels
# keep NVLink.
unset UCX_TLS

# ---------------------------------------------------------------------------
# 3. Locate the test (this script lives in the same dir as elastic.py)
# ---------------------------------------------------------------------------
TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "${TEST_DIR}"

NUM_PROCESSES=${NUM_PROCESSES:-4}   # GPUs on this node
PLAN_FILE=${PLAN_FILE:-nvlink_fault_tolerance.json}
FAULT_KILL_SIGNAL=${FAULT_KILL_SIGNAL:-sigkill}

# Common args to elastic.py. The fault kill signal defaults to SIGKILL to
# simulate an unrecoverable GPU loss; override via FAULT_KILL_SIGNAL=sigterm
# or by passing --fault-kill-signal on the command line (last one wins).
COMMON_ARGS=(
    --plan "${PLAN_FILE}"
    --num-processes "${NUM_PROCESSES}"
    --fault-kill-signal "${FAULT_KILL_SIGNAL}"
)
# Optional: durable in-kernel evidence dir (only meaningful for in-kernel
# timings). Set FAULT_EVIDENCE_DIR to enable.
if [[ -n "${FAULT_EVIDENCE_DIR:-}" ]]; then
    COMMON_ARGS+=(--fault-evidence-dir "${FAULT_EVIDENCE_DIR}")
fi

echo "[$(hostname)] single-node run: plan=${PLAN_FILE}, " \
     "num_processes=${NUM_PROCESSES}, fault_kill_signal=${FAULT_KILL_SIGNAL}"

# ---------------------------------------------------------------------------
# 4. Cleanup-verification helpers
# ---------------------------------------------------------------------------
# Each manual run captures pre-test state, runs the test, settles for a few
# seconds so async cleanup can complete, then re-queries state and prints a
# CLEANUP REPORT block. Default semantics are informational only: the script's
# exit code is the test's exit code. Set STRICT_CLEANUP=1 to escalate a dirty
# cleanup into a non-zero exit (useful for CI).
#
# NO pkill / no forced cleanup is performed -- the whole point is to measure
# whether the test's *own* shutdown path cleans up correctly after a SIGKILL.

# All of these run under `set -euo pipefail`, so each pipe MUST exit zero.
# `pgrep` returns 1 when no processes match, `ls /dev/shm/foo*` returns 2 when
# nothing matches, and `nvidia-smi` / `ss` can also fail under weird driver
# states. Wrap each fallible upstream command in `{ ... || true; }` so an
# empty result is counted as 0, not propagated as a script-killing failure.
gpu_mem_used_mib() {
    { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true; } \
        | awk 'BEGIN{s=0} {s+=$1} END{print s+0}'
}
leftover_proc_count() {
    { pgrep -af 'elastic\.py|rank_server|spawn_main|torch.multiprocessing' 2>/dev/null || true; } \
        | wc -l
}
shm_leak_count() {
    # shellcheck disable=SC2012
    { ls /dev/shm/torch_* /dev/shm/cuda.shm.* 2>/dev/null || true; } | wc -l
}
ports_in_use() {
    local count=0
    if command -v ss >/dev/null 2>&1; then
        count=$({ ss -tlnH 2>/dev/null || true; } | awk '$4 ~ /:(9999|10000)$/' | wc -l)
    elif command -v netstat >/dev/null 2>&1; then
        count=$({ netstat -tln 2>/dev/null || true; } | awk '$4 ~ /:(9999|10000)$/' | wc -l)
    fi
    echo "${count:-0}"
}
nvsmi_compute_app_count() {
    { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true; } \
        | awk 'NF>0' | wc -l
}

PRE_GPU_MIB=$(gpu_mem_used_mib)
PRE_PROCS=$(leftover_proc_count)
PRE_SHM=$(shm_leak_count)
PRE_PORTS=$(ports_in_use)
PRE_APPS=$(nvsmi_compute_app_count)

# ---------------------------------------------------------------------------
# 5. Run (rank server + TCPStore + all workers are started locally)
# ---------------------------------------------------------------------------
set +e
python3 elastic.py "${COMMON_ARGS[@]}" "$@"
TEST_RC=$?
set -e

# ---------------------------------------------------------------------------
# 6. Post-test cleanup verification
# ---------------------------------------------------------------------------
SETTLE_SECONDS=${SETTLE_SECONDS:-5}
MEM_LEAK_MIB=${MEM_LEAK_MIB:-64}
STRICT_CLEANUP=${STRICT_CLEANUP:-0}

sleep "${SETTLE_SECONDS}"

POST_GPU_MIB=$(gpu_mem_used_mib)
POST_PROCS=$(leftover_proc_count)
POST_SHM=$(shm_leak_count)
POST_PORTS=$(ports_in_use)
POST_APPS=$(nvsmi_compute_app_count)
GPU_DELTA=$(( POST_GPU_MIB - PRE_GPU_MIB ))

issues=()
if (( POST_PROCS > PRE_PROCS )); then
    issues+=("leftover_procs:${PRE_PROCS}->${POST_PROCS}")
fi
if (( POST_SHM > PRE_SHM )); then
    issues+=("shm_leak:${PRE_SHM}->${POST_SHM}")
fi
if (( POST_PORTS > 0 )); then
    issues+=("ports_in_use:${POST_PORTS}")
fi
if (( POST_APPS > PRE_APPS )); then
    issues+=("gpu_compute_apps:${PRE_APPS}->${POST_APPS}")
fi
if (( GPU_DELTA > MEM_LEAK_MIB )); then
    issues+=("gpu_mem_delta:${GPU_DELTA}MiB>threshold:${MEM_LEAK_MIB}MiB")
fi

printf '\n===================== CLEANUP REPORT =====================\n'
printf 'settle_seconds=%s test_exit_code=%s\n' "${SETTLE_SECONDS}" "${TEST_RC}"
printf 'leftover_procs:        pre=%s post=%s\n' "${PRE_PROCS}" "${POST_PROCS}"
printf 'shm_torch_files:       pre=%s post=%s\n' "${PRE_SHM}" "${POST_SHM}"
printf 'ports(9999,10000):     post=%s\n' "${POST_PORTS}"
printf 'nvsmi_compute_apps:    pre=%s post=%s\n' "${PRE_APPS}" "${POST_APPS}"
printf 'gpu_mem_used_mib:      pre=%s post=%s delta=%s threshold=%s\n' \
    "${PRE_GPU_MIB}" "${POST_GPU_MIB}" "${GPU_DELTA}" "${MEM_LEAK_MIB}"
if (( ${#issues[@]} == 0 )); then
    printf 'cleanup_result:        CLEAN\n'
else
    printf 'cleanup_result:        DIRTY (%s)\n' "${issues[*]}"
fi
printf '==========================================================\n'

if (( ${#issues[@]} > 0 )); then
    printf '\n--- detail: leftover processes ---\n'
    pgrep -af 'elastic\.py|rank_server|spawn_main|torch.multiprocessing' \
        || printf '(none)\n'
    printf '\n--- detail: /dev/shm leaks ---\n'
    ls -la /dev/shm/torch_* /dev/shm/cuda.shm.* 2>/dev/null \
        || printf '(none)\n'
    printf '\n--- detail: nvidia-smi compute apps ---\n'
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null \
        || printf '(none)\n'
fi

if (( STRICT_CLEANUP == 1 && ${#issues[@]} > 0 && TEST_RC == 0 )); then
    echo "STRICT_CLEANUP=1: escalating dirty cleanup to exit 2" >&2
    exit 2
fi
exit "${TEST_RC}"
