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

# Build UCX PR #12161 from source and run its CUDA Cluster Fabric Transport
# (CFT) gtests on the allocated DL-cluster GPU node.

set -e
set -o pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
UCX_CFT_REF="${UCX_CFT_REF:-222714e14b3ed35a2c4c3dfc4ce9648d2eac9db6}"
UCX_CFT_SOURCE="${UCX_CFT_SOURCE:-${PWD}/.ci/ucx-cft-src}"
CFT_JOBS="${CFT_JOBS:-32}"
WORK_DIR=$(mktemp -d /tmp/ucx-cft.XXXXXX)

cleanup()
{
    rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

section()
{
    echo
    echo "==== $* ===="
}

section "CFT environment"
hostname
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
"${CUDA_HOME}/bin/nvcc" --version
echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [ -d /dev/nvidia-caps-imex-channels ]; then
    ls -la /dev/nvidia-caps-imex-channels
else
    echo "IMEX_CHANNELS=absent"
fi

test -x "${CUDA_HOME}/bin/nvcc"
test -d "${UCX_CFT_SOURCE}"
test "$(cat "${UCX_CFT_SOURCE}/.ucx-cft-commit")" = "${UCX_CFT_REF}"

cat >"${WORK_DIR}/cft_driver_probe.cu" <<'EOF'
#include <cuda.h>

#include <cstdio>

static const char *const cft_ops[] = {
    "cuLogicalEndpointIdReserve",
    "cuLogicalEndpointIdRelease",
    "cuLogicalEndpointCreate",
    "cuLogicalEndpointDestroy",
    "cuLogicalEndpointQuery",
    "cuLogicalEndpointGetLimits",
    "cuLogicalEndpointBindAddr",
    "cuLogicalEndpointUnbind",
    "cuLogicalEndpointExport",
    "cuLogicalEndpointImport"
};

int main()
{
    int driver_version = 0;
    int device_count   = 0;
    bool supported     = true;

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::printf("cuInit=%d\n", static_cast<int>(result));
        return 1;
    }

    cuDriverGetVersion(&driver_version);
    result = cuDeviceGetCount(&device_count);
    if (result != CUDA_SUCCESS) {
        std::printf("cuDeviceGetCount=%d\n", static_cast<int>(result));
        return 1;
    }

    std::printf("cudaDriverGetVersion=%d device_count=%d\n", driver_version,
                device_count);
    if (device_count < 2) {
        std::printf("CFT_PROBE=UNAVAILABLE reason=requires-two-visible-gpus\n");
        return 2;
    }

    for (int index = 0; index < device_count; ++index) {
        CUdevice device;
        int logical_endpoint_supported = 0;

        result = cuDeviceGet(&device, index);
        if (result == CUDA_SUCCESS) {
            result = cuDeviceGetAttribute(
                    &logical_endpoint_supported,
                    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
                    device);
        }

        std::printf("device=%d logical_endpoint_unicast=%d query_result=%d\n",
                    index, logical_endpoint_supported,
                    static_cast<int>(result));
        supported = supported && (result == CUDA_SUCCESS) &&
                    (logical_endpoint_supported != 0);
    }

    for (const char *name : cft_ops) {
        void *function = nullptr;
        CUdriverProcAddressQueryResult symbol_status = {};

        result = cuGetProcAddress(name, &function, 13030,
                                  CU_GET_PROC_ADDRESS_DEFAULT, &symbol_status);
        std::printf("symbol=%s result=%d status=%d address=%p\n", name,
                    static_cast<int>(result),
                    static_cast<int>(symbol_status), function);
        supported = supported && (result == CUDA_SUCCESS) &&
                    (symbol_status == CU_GET_PROC_ADDRESS_SUCCESS) &&
                    (function != nullptr);
    }

    std::printf("CFT_PROBE=%s\n",
                supported ? "AVAILABLE" : "UNAVAILABLE");
    return supported ? 0 : 2;
}
EOF

section "CUDA CFT driver API probe"
probe_rc=125
if "${CUDA_HOME}/bin/nvcc" -std=c++14 -O2 \
        "${WORK_DIR}/cft_driver_probe.cu" \
        -L"${CUDA_HOME}/lib64/stubs" -lcuda \
        -o "${WORK_DIR}/cft_driver_probe"; then
    set +e
    "${WORK_DIR}/cft_driver_probe" 2>&1 |
        tee "${WORK_DIR}/driver-probe.log"
    probe_rc=${PIPESTATUS[0]}
    set -e
else
    echo "CFT_PROBE=UNAVAILABLE reason=CUDA-logical-endpoint-headers-missing"
fi

# Always build and run the upstream tests, even if the small probe says CFT is
# unavailable. This preserves UCX's own skip/error diagnostics as the primary
# evidence.
section "Prepare UCX PR #12161 source"
cp -a "${UCX_CFT_SOURCE}" "${WORK_DIR}/ucx"
echo "UCX_CFT_COMMIT=$(cat "${WORK_DIR}/ucx/.ucx-cft-commit")"
rm -f "${WORK_DIR}/ucx/.ucx-cft-commit"
(
    cd "${WORK_DIR}/ucx"
    ./autogen.sh
)

section "Configure UCX CFT gtests"
mkdir -p "${WORK_DIR}/build"
(
    cd "${WORK_DIR}/build"
    "${WORK_DIR}/ucx/contrib/configure-release-mt" \
        --prefix="${WORK_DIR}/install" \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-devel-headers \
        --enable-experimental-api \
        --enable-gtest \
        --without-avx \
        --enable-cma \
        --with-cuda="${CUDA_HOME}" \
        --with-verbs \
        --with-dm \
        --without-gdrcopy
)

grep -q '^#define HAVE_CUDA_LOGICAL_ENDPOINT 1' \
    "${WORK_DIR}/build/config.h"
echo "UCX_CONFIGURE_CFT_HEADERS=AVAILABLE"

section "Build UCX PR #12161"
make -C "${WORK_DIR}/build" -j"${CFT_JOBS}"
GTEST="${WORK_DIR}/build/test/gtest/gtest"
test -x "${GTEST}"

section "List UCX CFT tests"
"${GTEST}" --gtest_list_tests |
    tee "${WORK_DIR}/gtest-list.log" |
    grep -i -A8 -B2 cuda_ipc_cft || true
grep -qi 'cuda_ipc_cft' "${WORK_DIR}/gtest-list.log"

section "Run UCX CFT gtests"
set +e
UCX_HANDLE_ERRORS=bt UCX_LOG_LEVEL=info \
    timeout 900 "${GTEST}" \
        --gtest_filter='*cuda_ipc_cft*' \
        --gtest_color=no 2>&1 |
    tee "${WORK_DIR}/gtest-cft.log"
gtest_rc=${PIPESTATUS[0]}
set -e

passed=false
if grep -Eq '^\[  PASSED  \] [1-9][0-9]* test' \
        "${WORK_DIR}/gtest-cft.log"; then
    passed=true
fi

if [ "${probe_rc}" -eq 0 ] && [ "${gtest_rc}" -eq 0 ] &&
        [ "${passed}" = true ]; then
    echo "CFT_RESULT=PASS"
    exit 0
fi

echo "CFT_RESULT=FAIL probe_rc=${probe_rc} gtest_rc=${gtest_rc} passed=${passed}" >&2
exit 1
