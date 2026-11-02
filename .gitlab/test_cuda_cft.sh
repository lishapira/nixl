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
# (CFT) gtests on the allocated DL-cluster GPU node. UCX reports a skipped test
# to googletest as passed, so the result is PASS only if each device put and
# atomic test ran at least one variant without skipping.

set -e
set -o pipefail

NIXL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
UCX_CFT_REF="${UCX_CFT_REF:-222714e14b3ed35a2c4c3dfc4ce9648d2eac9db6}"
UCX_CFT_SOURCE="${UCX_CFT_SOURCE:-${NIXL_ROOT}/.ci/ucx-cft-src}"
CFT_JOBS="${CFT_JOBS:-$(( $(nproc) < 64 ? $(nproc) : 64 ))}"
CFT_REQUIRED_TESTS="put_device put_device_large atomic_add_device"
WORK_DIR=$(mktemp -d /tmp/ucx-cft.XXXXXX)
UCX_BUILD="${WORK_DIR}/build"
current_stage="setup"

on_exit()
{
    local rc=$?

    rm -rf "${WORK_DIR}"
    if [ "${rc}" -ne 0 ] && [ "${current_stage}" != "verdict" ]; then
        echo "CFT_RESULT=FAIL reason=${current_stage}-failed exit_code=${rc}" >&2
    fi
}
trap on_exit EXIT

fail()
{
    echo "ERROR: $*" >&2
    exit 1
}

section()
{
    current_stage="$1"
    shift
    echo
    echo "==== $* ===="
}

section environment "CFT environment"
hostname
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
nvidia-smi -q -i 0 | grep -E -A6 '^ +Fabric *$' || true
"${CUDA_HOME}/bin/nvcc" --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
grep -w nvidia-caps-imex-channels /proc/devices ||
    echo "IMEX: nvidia-caps-imex-channels is not registered in /proc/devices"
if [ -d /dev/nvidia-caps-imex-channels ]; then
    ls -la /dev/nvidia-caps-imex-channels
else
    echo "IMEX: /dev/nvidia-caps-imex-channels is absent"
fi

section source "UCX PR #12161 source"
[ -d "${UCX_CFT_SOURCE}/.git" ] ||
    fail "${UCX_CFT_SOURCE} is missing: the image build step could not fetch UCX ${UCX_CFT_REF}"
ucx_commit=$(git -c safe.directory='*' -C "${UCX_CFT_SOURCE}" rev-parse HEAD)
echo "UCX_CFT_COMMIT=${ucx_commit}"
[ "${ucx_commit}" = "${UCX_CFT_REF}" ] ||
    fail "UCX source is at ${ucx_commit}, expected ${UCX_CFT_REF}"
[ -f "${UCX_CFT_SOURCE}/src/uct/ib/mlx5/gdaki/gpunetio/common/doca_gpunetio_verbs_def.h" ] ||
    fail "UCX source has no GPUNetIO submodule (external/gpunetio), which the gdaki module needs"

section probe "CUDA logical endpoint and fabric memory probe"
cat >"${WORK_DIR}/cft_probe.cu" <<'EOF'
#include <cuda.h>

#include <cstdio>
#include <fstream>
#include <string>

/* Driver API version which UCX PR #12161 requests for these functions */
static const int cft_api_version = 13030;

static const char *const cft_functions[] = {
    "cuLogicalEndpointIdReserve", "cuLogicalEndpointIdRelease",
    "cuLogicalEndpointCreate",    "cuLogicalEndpointDestroy",
    "cuLogicalEndpointQuery",     "cuLogicalEndpointGetLimits",
    "cuLogicalEndpointBindAddr",  "cuLogicalEndpointUnbind",
    "cuLogicalEndpointExport",    "cuLogicalEndpointImport"
};

static const char *error_name(CUresult result)
{
    const char *name = nullptr;

    if ((cuGetErrorName(result, &name) != CUDA_SUCCESS) || (name == nullptr)) {
        return "unknown error";
    }

    return name;
}

static void print_libcuda()
{
    std::ifstream maps("/proc/self/maps");
    std::string line;

    while (std::getline(maps, line)) {
        if (line.find("libcuda.so") != std::string::npos) {
            std::printf("libcuda: %s\n", line.substr(line.find('/')).c_str());
            return;
        }
    }
}

/* Fabric-exportable VMM memory, like the remote buffer of the CFT gtests */
static CUresult fabric_alloc(CUdevice device)
{
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle handle;
    CUmemFabricHandle fabric_handle;
    size_t granularity = 0;
    CUresult result;

    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    result = cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (result != CUDA_SUCCESS) {
        return result;
    }

    result = cuMemCreate(&handle, granularity, &prop, 0);
    if (result != CUDA_SUCCESS) {
        return result;
    }

    result = cuMemExportToShareableHandle(&fabric_handle, handle,
                                          CU_MEM_HANDLE_TYPE_FABRIC, 0);
    cuMemRelease(handle);
    return result;
}

int main()
{
    int driver_version = 0;
    int device_count   = 0;
    bool cft           = true;
    bool fabric        = true;
    CUresult result;

    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::printf("cuInit: %s\n", error_name(result));
        return 1;
    }

    print_libcuda();
    cuDriverGetVersion(&driver_version);
    cuDeviceGetCount(&device_count);
    std::printf("cuDriverGetVersion: %d, devices: %d\n", driver_version,
                device_count);
    if (device_count < 2) {
        std::printf("the CFT gtests need two visible devices\n");
        cft = false;
    }

    for (const char *name : cft_functions) {
        void *function                        = nullptr;
        CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;

        result = cuGetProcAddress(name, &function, cft_api_version,
                                  CU_GET_PROC_ADDRESS_DEFAULT, &status);
        std::printf("%s: %s, symbol status %d\n", name, error_name(result),
                    static_cast<int>(status));
        cft = cft && (result == CUDA_SUCCESS) &&
              (status == CU_GET_PROC_ADDRESS_SUCCESS) && (function != nullptr);
    }

    for (int index = 0; index < device_count; ++index) {
        CUdevice device;
        CUcontext context;
        int le_supported     = 0;
        int fabric_supported = 0;
        CUresult le_result, fabric_result, alloc_result;

        result = cuDeviceGet(&device, index);
        if (result != CUDA_SUCCESS) {
            std::printf("device %d: cuDeviceGet: %s\n", index,
                        error_name(result));
            cft = fabric = false;
            continue;
        }

        le_result     = cuDeviceGetAttribute(
                &le_supported,
                CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED, device);
        fabric_result = cuDeviceGetAttribute(
                &fabric_supported,
                CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
        alloc_result  = cuDevicePrimaryCtxRetain(&context, device);
        if (alloc_result == CUDA_SUCCESS) {
            cuCtxSetCurrent(context);
            alloc_result = fabric_alloc(device);
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device);
        }

        std::printf("device %d: logical endpoint unicast %d (%s), fabric handle "
                    "%d (%s), fabric allocation %s\n",
                    index, le_supported, error_name(le_result),
                    fabric_supported, error_name(fabric_result),
                    error_name(alloc_result));
        cft    = cft && (le_result == CUDA_SUCCESS) && (le_supported != 0);
        fabric = fabric && (fabric_result == CUDA_SUCCESS) &&
                 (fabric_supported != 0) && (alloc_result == CUDA_SUCCESS);
    }

    std::printf("CFT_PROBE=%s\n", cft ? "AVAILABLE" : "UNAVAILABLE");
    std::printf("FABRIC_MEMORY_PROBE=%s\n", fabric ? "AVAILABLE" : "UNAVAILABLE");
    return (cft && fabric) ? 0 : 2;
}
EOF
probe_rc=0
if "${CUDA_HOME}/bin/nvcc" -std=c++14 -O2 "${WORK_DIR}/cft_probe.cu" \
        -L"${CUDA_HOME}/lib64/stubs" -lcuda -o "${WORK_DIR}/cft_probe"; then
    timeout 120 "${WORK_DIR}/cft_probe" || probe_rc=$?
else
    probe_rc=125
    echo "CFT_PROBE=UNAVAILABLE (the probe does not compile with ${CUDA_HOME}/include/cuda.h)"
fi

# Build and run the upstream tests even if the probe reports CFT as
# unavailable, so that UCX's own skip reasons stay the primary evidence.
section autogen "Generate UCX ${UCX_CFT_REF} build system"
cp -a "${UCX_CFT_SOURCE}" "${WORK_DIR}/ucx"
(cd "${WORK_DIR}/ucx" && ./autogen.sh)

section configure "Configure UCX with contrib/configure-devel"
mkdir "${UCX_BUILD}"
(cd "${UCX_BUILD}" &&
    "${WORK_DIR}/ucx/contrib/configure-devel" --prefix="${WORK_DIR}/install" \
        --with-cuda="${CUDA_HOME}" --without-java --without-go)
grep -q '^#define HAVE_CUDA_LOGICAL_ENDPOINT 1' "${UCX_BUILD}/config.h" ||
    fail "UCX did not find the CUDA logical endpoint API, so the CFT gtests are compiled out"

section make "Compile UCX and its gtest with ${CFT_JOBS} jobs"
make -C "${UCX_BUILD}" -j"${CFT_JOBS}"
"${UCX_BUILD}/src/tools/info/ucx_info" -v

section ucx-info "UCX cuda_ipc CFT detection"
UCX_CUDA_IPC_ENABLE_CFT=try UCX_LOG_LEVEL=debug \
    timeout 120 "${UCX_BUILD}/src/tools/info/ucx_info" -d -t cuda_ipc 2>&1 |
    grep -e 'fabric transport' -e 'not provided by the CUDA driver' \
         -e 'LOGICAL_ENDPOINT' -e 'Transport: cuda_ipc' || true

section gtest-list "List UCX CFT gtests"
# Same environment as "make -C test/gtest test", except that a fatal error
# prints a backtrace instead of freezing the process for a debugger
inherited_ucx_vars=$(compgen -e | grep '^UCX_' | tr '\n' ' ' || true)
echo "Unset inherited UCX variables: ${inherited_ucx_vars:-none}"
for var in ${inherited_ucx_vars}; do
    unset "${var}"
done
export UCX_HANDLE_ERRORS=bt UCX_LOG_LEVEL=warn UCX_LOG_PRINT_ENABLE=y
cd "${UCX_BUILD}/test/gtest"
./gtest --gtest_list_tests --gtest_filter='*cuda_ipc_cft*' \
    >"${WORK_DIR}/gtest-list.log"
grep -q 'test_cuda_ipc_cft' "${WORK_DIR}/gtest-list.log" ||
    fail "no test_cuda_ipc_cft test was instantiated for cuda_ipc"
echo "CFT test variants: $(grep -c '^  ' "${WORK_DIR}/gtest-list.log")"

section gtest-run "Run UCX CFT gtests"
gtest_rc=0
timeout 1200 stdbuf -e0 -o0 ./gtest --gtest_filter='*cuda_ipc_cft*' \
        --gtest_color=no 2>&1 |
    tee "${WORK_DIR}/gtest.log" || gtest_rc=$?

section verdict "CFT verdict"
summary_rc=0
awk -v required="${CFT_REQUIRED_TESTS}" '
    function method(name) {
        sub(/^[^.]*\./, "", name)
        sub(/\/.*$/, "", name)
        return name
    }
    /^\[ RUN      \] / {
        test = $0
        sub(/^\[ RUN      \] /, "", test)
        running = 1
        skipped = 0
        next
    }
    /^\[ +SKIP \] / {
        skipped = 1
        reason = $0
        sub(/^\[ +SKIP \] /, "", reason)
        reasons[reason]++
        next
    }
    /^\[       OK \] / {
        running = 0
        if (skipped) {
            nskipped++
        } else {
            nran++
            ran[method(test)]++
        }
        next
    }
    /^\[  (FAILED|SKIPPED) \] .*\([0-9]+ ms\)$/ {
        running = 0
        if ($0 ~ /FAILED/) {
            nfailed++
        } else {
            nskipped++
        }
    }
    END {
        nfailed += running
        printf("CFT gtests: %d ran, %d skipped, %d failed\n", nran, nskipped,
               nfailed)
        for (reason in reasons) {
            printf("  skipped %d times: %s\n", reasons[reason], reason)
        }
        ok = (nfailed == 0)
        n  = split(required, names, " ")
        for (i = 1; i <= n; i++) {
            printf("  %s: %d variants ran without skipping\n", names[i],
                   ran[names[i]])
            if (ran[names[i]] == 0) {
                ok = 0
            }
        }
        exit(ok ? 0 : 1)
    }' "${WORK_DIR}/gtest.log" || summary_rc=$?

if [ "${gtest_rc}" -eq 0 ] && [ "${summary_rc}" -eq 0 ]; then
    echo "CFT_RESULT=PASS probe_exit_code=${probe_rc}"
    exit 0
fi

echo "CFT_RESULT=FAIL gtest_exit_code=${gtest_rc} probe_exit_code=${probe_rc}" >&2
exit 1
