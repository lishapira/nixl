#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NIXL EP V2 CI: run the runtime/ep correctness tests on one node with 4 GPUs.

# shellcheck disable=SC1091
. "$(dirname "$0")/../.ci/scripts/common.sh"

set -e
set -x
set -o pipefail

INSTALL_DIR=$1

if [ -z "$INSTALL_DIR" ]; then
    echo "Usage: $0 <install_dir>"
    exit 1
fi

# NIXL EP V2 is built against the vLLM venv's Python and Torch (NIXL_PYTHON in test-dl-ep-matrix.yaml).
: "${VLLM_ELASTIC_TEST_DIR:?vLLM Elastic EP test environment is not installed}"
EP_PYTHON="${VLLM_ELASTIC_TEST_DIR}/.venv/bin/python"

if [ ! -x "${EP_PYTHON}" ]; then
    echo "ERROR: Python environment is missing: ${EP_PYTHON}" >&2
    exit 1
fi

ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] && ARCH="aarch64"

export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${INSTALL_DIR}/lib/$ARCH-linux-gnu:${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins:/usr/local/lib:$LD_LIBRARY_PATH
export CPATH=${INSTALL_DIR}/include:$CPATH
export PATH=${INSTALL_DIR}/bin:$PATH
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH
export NIXL_PLUGIN_DIR=${INSTALL_DIR}/lib/$ARCH-linux-gnu/plugins
export NIXL_PREFIX=${INSTALL_DIR}

# Load the source dispatcher and the backend installed in the shared venv.
export PYTHONPATH="${PWD}/src/bindings/python/nixl-meta${PYTHONPATH:+:$PYTHONPATH}"

# Avoid SPCx loading HPC-X UCX; NIXL EP V2 fails with it ("UCP API version is incompatible").
unset NCCL_NET_PLUGIN
unset UCX_NET_DEVICES
unset NIXL_ETCD_ENDPOINTS NIXL_ETCD_PEER_URLS NIXL_ETCD_NAMESPACE

# A fresh JIT cache per run, shared by the tests so each kernel compiles once.
export NIXL_EP_JIT_CACHE_DIR="${PWD}/nixl_ep_v2_jit_cache"
rm -rf "${NIXL_EP_JIT_CACHE_DIR}"
mkdir -p "${NIXL_EP_JIT_CACHE_DIR}"

NUM_GPUS=4

echo "==== Show system info ===="
env
nvidia-smi topo -m || true
nvidia-smi nvlink -s || true
ibv_devinfo || true
uname -a || true
cat /sys/devices/virtual/dmi/id/product_name || true

echo "==== Check that NIXL EP V2 is installed ===="
# meson only warns and skips V2 when a build requirement is missing, so fail here instead.
"${EP_PYTHON}" -c "from nixl.runtime import ep; print('NIXL EP V2', ep.__version__, ep.__file__)"

echo "==== Running NIXL EP V2 tests ===="
EP_V2_DIR="runtime/ep"
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29600}
num_runs=0

# run_v2_test <timeout_secs> <test_script> [args...]: one single-node run from runtime/ep,
# on its own rendezvous port so a lingering store of a previous run cannot collide.
run_v2_test() {
    local timeout_secs=$1
    shift
    local port=$((MASTER_PORT_BASE + num_runs))
    num_runs=$((num_runs + 1))
    echo "---- v2: $* [NIXL_EP_ADAPTIVE_MODE=${NIXL_EP_ADAPTIVE_MODE:-auto}] ----"
    (
        cd "${EP_V2_DIR}"
        WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=${port} \
        timeout "${timeout_secs}" "${EP_PYTHON}" "$@"
    )
}

# Fixed world: the full dispatch/combine option matrix against the torch.distributed reference.
run_v2_test 600 tests/elastic/test_ep.py --num-processes ${NUM_GPUS} --skip-perf-test

# Direct and hybrid kernels on one domain, CUDA graph replay across mode and world switches.
run_v2_test 600 tests/elastic/test_adaptive_modes.py --num-processes ${NUM_GPUS}

# Elastic plans: the default plan in the derived (direct) mode, forced hybrid with deterministic
# output, and leave/rejoin through disconnect_ranks/connect_ranks.
run_v2_test 480 tests/elastic/test_adaptive_elastic.py --num-processes ${NUM_GPUS} --skip-perf-test
NIXL_EP_ADAPTIVE_MODE=hybrid \
    run_v2_test 480 tests/elastic/test_adaptive_elastic.py --num-processes ${NUM_GPUS} --skip-perf-test --deterministic
run_v2_test 480 tests/elastic/test_adaptive_elastic.py --num-processes ${NUM_GPUS} --skip-perf-test --reconnect-on-leave

# Multi-step 1 -> 2 -> 4 -> 2 -> 1 scaling with another kernel shape: top-k 2 below the live rank count
# takes the combine path without the rank layout, which the runs above (top-k >= ranks) never reach.
run_v2_test 480 tests/elastic/test_adaptive_elastic.py --num-processes ${NUM_GPUS} --skip-perf-test \
    --plan '[[0],[0,1],[0,1,2,3],[0,1],[0]]' --hidden 4096 --num-topk 2

echo "==== NIXL EP V2 tests done ===="
