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
# 4. Run (rank server + TCPStore + all workers are started locally)
# ---------------------------------------------------------------------------
exec python3 elastic.py "${COMMON_ARGS[@]}" "$@"
