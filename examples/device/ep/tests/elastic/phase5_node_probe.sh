#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Combined per-node probe used by run_phase5_2node.sh for the fast path.
# Runs inside a single srun -N2 --ntasks-per-node=1 step (once per node,
# in parallel) so we pay ONE container-start cost instead of 6 (two
# probe steps + two nvlink snapshot steps + two health probes).
#
# STAGE=pre:
#   - MNNVL probe (phase5_mnnvl_probe.py) -> $RUN_DIR/pre_<host>.json
#   - NVLink counter snapshot -> $RUN_DIR/pre_<host>_nvlink.csv
#
# STAGE=post:
#   - same two probes -> post_<host>{.json,_nvlink.csv}
#   - GPU health probe (nvidia-smi + torch.cuda smoke) -> health_<host>.log
#
# Required env (passed via srun --export):
#   RUN_DIR    absolute container-side path
#   STAGE      "pre" or "post"
#   TEST_DIR   absolute container-side path to nixl/.../elastic
#
# SLURM_NODEID picks the role (0=master, 1=worker) but this script does
# NOT care about role -- it just probes the local node.

set -uo pipefail

: "${RUN_DIR:?RUN_DIR not set}"
: "${STAGE:?STAGE not set (pre|post)}"
: "${TEST_DIR:?TEST_DIR not set}"

HOST=$(hostname -s)
echo "[probe ${HOST}] stage=${STAGE} run_dir=${RUN_DIR}"

# setup_node.sh is needed for torch (health probe) to see the correct
# CUDA lib path; harmless for the nvidia-smi-only paths.
source /workspace/lishapira/setup_node.sh >/dev/null 2>&1 || true

# --- MNNVL fabric ------------------------------------------------------
python3 "${TEST_DIR}/phase5_mnnvl_probe.py" \
    > "${RUN_DIR}/${STAGE}_${HOST}.json" 2>/dev/null || {
    echo "[probe ${HOST}] mnnvl probe FAILED"
    printf '{"host":"%s","gpus":[]}\n' "${HOST}" > "${RUN_DIR}/${STAGE}_${HOST}.json"
}

# --- NVLink counters ---------------------------------------------------
# Reuse fault_artifacts._parse_nvsmi_errorcounters so the CSV columns
# exactly match what phase5_pass_gate.py expects.
python3 - "${HOST}" > "${RUN_DIR}/${STAGE}_${HOST}_nvlink.csv" 2>/dev/null <<'PY'
import csv, subprocess, sys, time
sys.path.insert(0, "/workspace/lishapira/nixl/examples/device/ep/tests/elastic")
import fault_artifacts as fa

host = sys.argv[1]
try:
    raw = subprocess.check_output(
        ["nvidia-smi", "nvlink", "--errorcounters"],
        stderr=subprocess.STDOUT, text=True, timeout=15,
    )
except Exception:
    raw = ""

rows = fa._parse_nvsmi_errorcounters(raw, ts_ns=time.time_ns(), fallback_gpu="?") if raw else []
writer = csv.writer(sys.stdout)
writer.writerow(["host", "gpu", "link"] + list(fa._NVLINK_FIELDS))
# _parse_nvsmi_errorcounters returns rows of [ts_ns, gpu, link, <fields...>].
# We drop the ts column and prepend host for the phase5_pass_gate schema.
for row in rows:
    _ts, gpu, link, *values = row
    writer.writerow([host, gpu, link] + values)
PY
echo "[probe ${HOST}] wrote ${STAGE}_${HOST}.json + ${STAGE}_${HOST}_nvlink.csv"

# --- Post-stage only: GPU health probe --------------------------------
if [[ "${STAGE}" == "post" ]]; then
    {
        echo "=== HEALTH ${HOST} ==="
        nvidia-smi | head -30 2>&1 || echo "nvidia-smi failed"
        python3 - <<'PY' 2>&1
try:
    import torch
    ok = torch.cuda.is_available()
    n = torch.cuda.device_count() if ok else 0
    print(f"torch.cuda.is_available={ok} device_count={n}")
    if ok and n > 0:
        t = torch.zeros(4, device="cuda:0")
        t += 1.0
        torch.cuda.synchronize()
        print(f"GPU-0 CTX OK, tensor sum = {float(t.sum())}")
    else:
        print("CTX FAILED: no CUDA devices available in this process")
except Exception as ex:
    print(f"CTX FAILED: {ex!r}")
PY
    } > "${RUN_DIR}/health_${HOST}.log" 2>&1
    echo "[probe ${HOST}] wrote health_${HOST}.log"
fi

exit 0
