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

# NIXL EP CI: report the CUDA toolkit, driver and PyTorch CUDA versions seen
# inside the test container. Diagnostic only; never fails the step.

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
VLLM_PYTHON="${VLLM_ELASTIC_TEST_DIR:-${HOME}/vllm}/.venv/bin/python"

section() {
    echo
    echo "==== $* ===="
}

section "CUDA version report on $(hostname)"

section "Image environment"
env | grep -E '^(CUDA_VERSION|CUDA_DRIVER_VERSION|NVIDIA_PYTORCH_VERSION|PYTORCH_VERSION|NCCL_VERSION|TRT_VERSION|CUDNN_VERSION)=' | sort

section "CUDA toolkit: ${CUDA_HOME}"
readlink -f "${CUDA_HOME}"
"${CUDA_HOME}/bin/nvcc" --version
grep -h '#define CUDA_VERSION ' "${CUDA_HOME}/include/cuda.h"
grep -h '#define CUDART_VERSION ' "${CUDA_HOME}/include/cuda_runtime_api.h"

section "Driver"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
nvidia-smi | grep -m1 'CUDA Version'

section "Runtime and driver API versions of a program built with this nvcc"
tmp_dir=$(mktemp -d)
cat > "${tmp_dir}/cuda_version.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    printf("cudaRuntimeGetVersion=%d cudaDriverGetVersion=%d\n", runtime, driver);
    return 0;
}
EOF
"${CUDA_HOME}/bin/nvcc" -o "${tmp_dir}/cuda_version" "${tmp_dir}/cuda_version.cu" && "${tmp_dir}/cuda_version"
rm -rf "${tmp_dir}"

for py in python3 "${VLLM_PYTHON}"; do
    section "PyTorch via ${py}"
    "${py}" - <<'EOF'
import torch

print("torch", torch.__version__, "| built with CUDA", torch.version.cuda, "| NCCL", torch.cuda.nccl.version())
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
    with open("/proc/self/maps") as maps:
        libs = sorted({line.split()[-1] for line in maps if "libcudart" in line or "libcuda.so" in line})
    print("loaded:\n  " + "\n  ".join(libs))
else:
    print("CUDA is not available to torch")
EOF
done

section "End of CUDA version report"
exit 0
