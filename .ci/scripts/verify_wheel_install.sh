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

set -euo pipefail
set -x

WHEELHOUSE=${1:-/wheelhouse}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
TORCH_VERSIONS=${TORCH_VERSIONS:-"2.11 2.12 2.13"}
VENV_PATH=${VENV_PATH:-/tmp/nixl-wheel-smoke}

if [ ! -d "${WHEELHOUSE}" ]; then
    echo "ERROR: wheelhouse does not exist: ${WHEELHOUSE}" >&2
    exit 1
fi

CUDA_RELEASE=$(nvcc --version | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)
if [ -z "${CUDA_RELEASE}" ]; then
    echo "ERROR: unable to determine CUDA release from nvcc" >&2
    exit 1
fi

CUDA_MAJOR=${CUDA_RELEASE%%.*}
if [ "${CUDA_MAJOR}" != "12" ] && [ "${CUDA_MAJOR}" != "13" ]; then
    echo "ERROR: unsupported CUDA major version: ${CUDA_MAJOR}" >&2
    exit 1
fi

CUDA_TAG="cu${CUDA_RELEASE//./}"
TORCH_STABLE_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
TORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/${CUDA_TAG}"

rm -rf "${VENV_PATH}"
uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}"

install_torch() {
    local version=$1
    local major=${version%%.*}
    local minor=${version##*.}

    if uv pip install \
        --python "${VENV_PATH}/bin/python" \
        --index-url "${TORCH_STABLE_INDEX}" \
        "torch==${version}.*"; then
        return 0
    fi

    uv pip install \
        --python "${VENV_PATH}/bin/python" \
        --index-url "${TORCH_NIGHTLY_INDEX}" \
        --pre \
        "torch>=${major}.${minor}.0.dev0,<${major}.$((minor + 1))"
}

TORCH_INSTALLED=false
for torch_version in ${TORCH_VERSIONS}; do
    if install_torch "${torch_version}"; then
        TORCH_INSTALLED=true
        break
    fi
done

if [ "${TORCH_INSTALLED}" != "true" ]; then
    echo "ERROR: no requested torch version is installable for Python ${PYTHON_VERSION} + ${CUDA_TAG}" >&2
    exit 1
fi

PYTHONPATH= uv pip install \
    --python "${VENV_PATH}/bin/python" \
    --no-index \
    --find-links "${WHEELHOUSE}" \
    nixl

PYTHONPATH= "${VENV_PATH}/bin/python" -c '
import sys

import torch

import nixl_ep  # noqa: F401

cuda = torch.version.cuda
assert cuda, "torch.version.cuda is empty"

expected = "nixl_ep_cu" + cuda.split(".")[0]
loaded = [name for name in sys.modules if name.startswith("nixl_ep")]
assert expected in sys.modules, f"expected {expected}, loaded nixl_ep modules: {loaded}"

print(f"OK: import nixl_ep selected {expected}")
'
