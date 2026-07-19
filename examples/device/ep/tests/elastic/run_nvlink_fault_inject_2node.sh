#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 2-node MNNVL fault-injection runner (sigkill OR cuMemUnmap on a live
# peer-exposed RDMA buffer). Selectable via FAULT_INJECT_MODE.
#
# 4-srun fast path (each srun container-start on Lyris costs ~30 s):
#   1. srun -N2 --ntasks-per-node=1  =>  parallel PRE probe on both nodes
#      (MNNVL fabric snapshot + NVLink error-counter snapshot).
#   2. srun -N1 on master             =>  elastic.py (rank server + 4 ranks).
#   3. srun -N1 on worker (bg)        =>  elastic.py --tcp-server <master>.
#   4. srun -N2 --ntasks-per-node=1  =>  parallel POST probe (same as PRE
#      + torch.cuda health probe).
#
# Pass gates (see fault_inject_pass_gate.py):
#   * MNNVL clique intact PRE and POST (fabric not corrupted).
#   * All 7 peer summary.json present, role=peer, survived=true.
#   * Victim summary handling is MODE-DEPENDENT:
#       - unmap-mid-flight: rank 2 summary.json MUST be present
#         (victim stays alive at Python level after the unmap).
#       - sigkill: rank 2 summary.json is EXPECTED to be missing
#         (victim self-SIGKILLs before it can flush); downgraded to WARN.
#   * At least one fault-observability signal (NIXL-EP mask-detection
#     timeout, victim illegal_address, IMEX error, NVLink counter delta,
#     or per-message NIXL-EP timeout from the victim). Both modes must
#     produce at least one of these to PASS unless ALLOW_NO_OBSERVABLE_FAULT=1.
#
# Required env (from an active salloc):
#   SLURM_JOB_ID
#   SLURM_JOB_NODELIST   (>= 2 nodes, MNNVL-coupled)
#
# Optional env:
#   FAULT_INJECT_MODE  sigkill | unmap-mid-flight    (default: unmap-mid-flight)
#   TIMING             elastic.py --fault-kill-timing (default: before-dispatch)
#   NPROCS_PN          workers per node               (default: 4)
#   PLAN_FILE          default: nvlink_fault_tolerance_2node_unmap.json
#   IN_KERNEL_SPIN_CYCLES  GPU spin cycles inside the in-kernel fault marker
#                          (default 1_000_000 ~ 500 us on GB200). Only used
#                          for '*-during-kernel-*' timings. Small values may
#                          collapse to LATE_UNMAP because cuMemUnmap teardown
#                          on the cold path can be ~26 ms; 200_000_000 (~100 ms)
#                          reliably lands as TRUE_IN_KERNEL_UNMAP.
#   P2P_PROBE_TARGET   dst_rank whose p2p_ptr_get() null/non-null count peers'
#                      send warps track; -1 disables. Set to the victim rank
#                      (e.g. 2) to record "does rank 2 stay a valid NVLink
#                      destination from every peer's GPU perspective across
#                      the fault window?" See WHY_NO_NVLINK_TRANSPORT_FAULT.md.
#   ALLOW_NO_OBSERVABLE_FAULT=1  downgrade observability gate to WARN

set -uo pipefail

LISHAPIRA_DIR=/lustre/fsw/network_research_advdev/lishapira
SQSH=${SQSH:-${LISHAPIRA_DIR}/nixl-hybrid-ep-cuda2.sqsh}
TEST_DIR_HOST="${LISHAPIRA_DIR}/nixl/examples/device/ep/tests/elastic"
TEST_DIR_CONT="/workspace/lishapira/nixl/examples/device/ep/tests/elastic"

: "${SLURM_JOB_ID:?run from active salloc}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST not set}"
[[ -f "${SQSH}" ]] || { echo "missing container image: ${SQSH}" >&2; exit 2; }

TIMING=${TIMING:-before-dispatch}
NPROCS_PN=${NPROCS_PN:-4}
PLAN_FILE=${PLAN_FILE:-nvlink_fault_tolerance_2node_unmap.json}
IN_KERNEL_SPIN_CYCLES=${IN_KERNEL_SPIN_CYCLES:-1000000}
P2P_PROBE_TARGET=${P2P_PROBE_TARGET:--1}
# FAULT_INJECT_MODE: sigkill (victim self-SIGKILLs; process gone) or
# unmap-mid-flight (victim calls cuMemUnmap+cuMemAddressFree+cuMemRelease
# on its own peer-exposed RDMA buffer and stays alive).
FAULT_INJECT_MODE=${FAULT_INJECT_MODE:-unmap-mid-flight}
TOTAL_PROCS=$(( NPROCS_PN * 2 ))
CONT_MOUNTS="${LISHAPIRA_DIR}:/workspace/lishapira,/var/log:/host/var/log:ro"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}" 2>/dev/null)
if (( ${#NODES[@]} < 2 )); then
    echo "error: need 2 nodes, got ${#NODES[@]}: ${NODES[*]:-none}" >&2
    exit 2
fi
MASTER_HOST="${NODES[0]}"
WORKER_HOST="${NODES[1]}"

UTC=$(date -u +%Y%m%d_%H%M%S)
RUN_DIR_HOST="${TEST_DIR_HOST}/results/nvlink_fault_inject_2node_${UTC}_${MASTER_HOST}_${WORKER_HOST}_${TIMING}"
RUN_DIR_CONT="${TEST_DIR_CONT}/results/nvlink_fault_inject_2node_${UTC}_${MASTER_HOST}_${WORKER_HOST}_${TIMING}"
mkdir -p "${RUN_DIR_HOST}"
chmod 2777 "${RUN_DIR_HOST}" 2>/dev/null || true

echo "=========================================================================="
echo " NVLink fault-inject: 2 nodes over MNNVL (fast-path)"
echo "   job        = ${SLURM_JOB_ID}"
echo "   master     = ${MASTER_HOST}"
echo "   worker     = ${WORKER_HOST}"
echo "   plan       = ${PLAN_FILE}  (victim rank = 2, on master)"
echo "   timing     = ${TIMING}"
echo "   spin_cycles= ${IN_KERNEL_SPIN_CYCLES} (in-kernel timings only)"
echo "   p2p_probe_target        = ${P2P_PROBE_TARGET}"\
"$( [[ ${P2P_PROBE_TARGET} -ge 0 ]] && echo ' (P2P PROBE ENABLED)' || echo ' (P2P probe disabled)' )"
echo "   fault_inject_mode       = ${FAULT_INJECT_MODE}"
echo "   nprocs/nod = ${NPROCS_PN}   total procs = ${TOTAL_PROCS}"
echo "   RUN_DIR    = ${RUN_DIR_HOST}"
echo "=========================================================================="

PROBE_SCRIPT="${TEST_DIR_CONT}/node_probe.sh"

# ---------------------------------------------------------------------------
# STEP 1: PRE probe (parallel on both nodes in one srun step)
# ---------------------------------------------------------------------------
echo "[fault-inject] STEP 1/4: PRE probe (MNNVL + NVLink counters, parallel on both nodes)..."
srun --jobid="${SLURM_JOB_ID}" --overlap \
    --nodes=2 --ntasks-per-node=1 --nodelist="${MASTER_HOST},${WORKER_HOST}" \
    --container-image="${SQSH}" \
    --container-mounts="${CONT_MOUNTS}" \
    --container-workdir=/workspace/lishapira \
    --export=ALL,RUN_DIR="${RUN_DIR_CONT}",STAGE=pre,TEST_DIR="${TEST_DIR_CONT}" \
    bash "${PROBE_SCRIPT}" 2>&1 | sed 's/^/  /' \
    || { echo "[fault-inject] PRE probe failed" >&2; exit 3; }

# Aggregate PRE artifacts on the login side into the shapes fault_inject_pass_gate expects.
python3 - "${RUN_DIR_HOST}" "${MASTER_HOST}" "${WORKER_HOST}" <<'PY'
import json, sys, os, csv
run_dir, m_host, w_host = sys.argv[1], sys.argv[2], sys.argv[3]

# ---- MNNVL: pre_<host>.json -> mnnvl_pre.json (list of 2 node payloads)
per_node = []
for h in (m_host, w_host):
    p = os.path.join(run_dir, f"pre_{h}.json")
    per_node.append(json.load(open(p)) if os.path.isfile(p) else {"host": h, "gpus": []})
json.dump(per_node, open(os.path.join(run_dir, "mnnvl_pre.json"), "w"), indent=2)

# ---- NVLink: pre_<host>_nvlink.csv (x2) -> nvlink_pre.csv (concat, one header)
out_path = os.path.join(run_dir, "nvlink_pre.csv")
header_written = False
with open(out_path, "w") as outfh:
    for h in (m_host, w_host):
        src = os.path.join(run_dir, f"pre_{h}_nvlink.csv")
        if not os.path.isfile(src):
            continue
        with open(src) as srcfh:
            lines = srcfh.readlines()
        if not lines:
            continue
        if not header_written:
            outfh.write(lines[0])
            header_written = True
        outfh.writelines(lines[1:])

# Print a quick MNNVL PRE verdict so a mismatch fails fast.
clusters, cliques = set(), set()
for node in per_node:
    for g in node.get("gpus", []):
        cu = g.get("cluster_uuid")
        ci = g.get("clique_id")
        if cu and cu != "unknown":
            clusters.add(cu)
        if ci and ci != "unknown":
            cliques.add(ci)
print(f"[fault-inject] MNNVL PRE: clusters={sorted(clusters)} cliques={sorted(cliques)}")
if len(clusters) != 1 or len(cliques) != 1:
    print("[fault-inject] FAIL: master and worker are NOT MNNVL-coupled (different Cluster/Clique)")
    print("[fault-inject] Refusing to run -- fault injection on non-MNNVL nodes tells us nothing.")
    sys.exit(2)
print("[fault-inject] MNNVL PRE OK")
PY
mnnvl_rc=$?
if (( mnnvl_rc != 0 )); then
    echo "[fault-inject] aborting: MNNVL pre-check failed (rc=${mnnvl_rc})"
    exit "${mnnvl_rc}"
fi
pre_lines=$(wc -l < "${RUN_DIR_HOST}/nvlink_pre.csv" 2>/dev/null || echo 0)
echo "[fault-inject] PRE artifacts: nvlink_pre.csv=${pre_lines} lines, mnnvl_pre.json written"

# ---------------------------------------------------------------------------
# STEP 2 + 3: launch master, poll for rank registrations, launch worker.
# ---------------------------------------------------------------------------
MASTER_LOG="${RUN_DIR_HOST}/master.log"
WORKER_LOG="${RUN_DIR_HOST}/worker.log"

launch_elastic() {
    local host="$1"
    local tcp_server="$2"
    local log_file="$3"

    local tcp_arg=""
    if [[ -n "${tcp_server}" ]]; then
        tcp_arg="--tcp-server ${tcp_server}"
    fi

    srun --jobid="${SLURM_JOB_ID}" --overlap \
        --nodes=1 --ntasks=1 --nodelist="${host}" \
        --container-image="${SQSH}" \
        --container-mounts="${CONT_MOUNTS}" \
        --container-workdir=/workspace/lishapira \
        --export=ALL,RUN_DIR_CONT="${RUN_DIR_CONT}" \
        bash -c "
set -uo pipefail
source /workspace/lishapira/setup_node.sh >/dev/null 2>&1
unset UCX_TLS
cd ${TEST_DIR_CONT}
export NIXL_FAULT_CAPTURE=1
export FAULT_EVIDENCE_DIR=${RUN_DIR_CONT}
mkdir -p \"\${FAULT_EVIDENCE_DIR}\"
echo \"[fault-inject-\$(hostname -s)] launching elastic.py ${tcp_arg}\"
python3 -u elastic.py \\
    --plan ${PLAN_FILE} \\
    --num-processes ${NPROCS_PN} \\
    --fault-kill-signal sigkill \\
    --fault-kill-timing ${TIMING} \\
    --fault-inject-mode ${FAULT_INJECT_MODE} \\
    --in-kernel-fault-spin-cycles ${IN_KERNEL_SPIN_CYCLES} \\
    --p2p-probe-target ${P2P_PROBE_TARGET} \\
    --fault-evidence-dir \"\${FAULT_EVIDENCE_DIR}\" \\
    ${tcp_arg}
rc=\$?
echo \"[fault-inject-\$(hostname -s)] elastic rc=\${rc}\"
exit \${rc}
" > "${log_file}" 2>&1 &
    LAUNCH_BG_PID=$!
}

cleanup_bg() {
    # Kill ONLY our own srun clients, not the parent allocation.
    for p in "${MASTER_PID:-}" "${WORKER_PID:-}"; do
        [[ -z "$p" ]] && continue
        if kill -0 "$p" 2>/dev/null; then
            echo "[fault-inject] cleanup: killing bg srun client pid=$p"
            kill -TERM "$p" 2>/dev/null || true
        fi
    done
}
# EXIT trap only -- do NOT trap INT/TERM so ^C on the orchestrator does
# not propagate SIGTERM to the srun clients we launched (that could
# cascade back to the allocation shell if slurm relays it).
trap 'cleanup_bg' EXIT

echo "[fault-inject] STEP 2/4: launching MASTER on ${MASTER_HOST}..."
launch_elastic "${MASTER_HOST}" "" "${MASTER_LOG}"
MASTER_PID="${LAUNCH_BG_PID}"

MASTER_READY_TIMEOUT=90
deadline=$(( $(date +%s) + MASTER_READY_TIMEOUT ))
master_ready=0
n_reg=0
while (( $(date +%s) < deadline )); do
    if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
        echo "[fault-inject] master died during head-start; log tail:" >&2
        tail -n 40 "${MASTER_LOG}" | sed 's/^/  /' >&2
        wait "${MASTER_PID}" 2>/dev/null || true
        MASTER_PID=""
        exit 4
    fi
    if [[ -f "${MASTER_LOG}" ]]; then
        # `grep -c` PRINTS the count (0) AND EXITS 1 when there is no match.
        # `|| echo 0` therefore concatenates "0\n0" into n_reg and (( ... ))
        # errors out. Use `; true` to reset the exit status without appending
        # a second 0 to the variable.
        n_reg=$(grep -c '^Process [0-9]\+ -> global_rank=' "${MASTER_LOG}" 2>/dev/null; true)
        n_reg=${n_reg:-0}
        if (( n_reg >= NPROCS_PN )); then
            master_ready=1
            break
        fi
    fi
    sleep 1
done

if (( master_ready == 0 )); then
    echo "[fault-inject] master not ready (${n_reg}/${NPROCS_PN}) after ${MASTER_READY_TIMEOUT}s" >&2
    tail -n 60 "${MASTER_LOG}" | sed 's/^/  /' >&2
    kill -TERM "${MASTER_PID}" 2>/dev/null || true
    wait "${MASTER_PID}" 2>/dev/null || true
    MASTER_PID=""
    exit 5
fi

echo "[fault-inject] master ready (${n_reg}/${NPROCS_PN})"
echo "[fault-inject] STEP 3/4: launching WORKER on ${WORKER_HOST}..."
launch_elastic "${WORKER_HOST}" "${MASTER_HOST}" "${WORKER_LOG}"
WORKER_PID="${LAUNCH_BG_PID}"

wait "${WORKER_PID}"; worker_rc=$?
WORKER_PID=""
wait "${MASTER_PID}"; master_rc=$?
MASTER_PID=""

echo "[fault-inject] MASTER rc=${master_rc}   WORKER rc=${worker_rc}"

# Best-effort settle for driver state.
sleep 5

# ---------------------------------------------------------------------------
# STEP 4: POST probe (parallel on both nodes in one srun step)
# ---------------------------------------------------------------------------
echo "[fault-inject] STEP 4/4: POST probe (MNNVL + NVLink + health, parallel)..."
srun --jobid="${SLURM_JOB_ID}" --overlap \
    --nodes=2 --ntasks-per-node=1 --nodelist="${MASTER_HOST},${WORKER_HOST}" \
    --container-image="${SQSH}" \
    --container-mounts="${CONT_MOUNTS}" \
    --container-workdir=/workspace/lishapira \
    --export=ALL,RUN_DIR="${RUN_DIR_CONT}",STAGE=post,TEST_DIR="${TEST_DIR_CONT}" \
    bash "${PROBE_SCRIPT}" 2>&1 | sed 's/^/  /' \
    || echo "[fault-inject] POST probe partial failure (continuing to gate)"

# Aggregate POST artifacts (same shapes as PRE).
python3 - "${RUN_DIR_HOST}" "${MASTER_HOST}" "${WORKER_HOST}" <<'PY'
import json, sys, os
run_dir, m_host, w_host = sys.argv[1], sys.argv[2], sys.argv[3]
per_node = []
for h in (m_host, w_host):
    p = os.path.join(run_dir, f"post_{h}.json")
    per_node.append(json.load(open(p)) if os.path.isfile(p) else {"host": h, "gpus": []})
json.dump(per_node, open(os.path.join(run_dir, "mnnvl_post.json"), "w"), indent=2)
out_path = os.path.join(run_dir, "nvlink_post.csv")
header_written = False
with open(out_path, "w") as outfh:
    for h in (m_host, w_host):
        src = os.path.join(run_dir, f"post_{h}_nvlink.csv")
        if not os.path.isfile(src):
            continue
        with open(src) as srcfh:
            lines = srcfh.readlines()
        if not lines:
            continue
        if not header_written:
            outfh.write(lines[0]); header_written = True
        outfh.writelines(lines[1:])
clusters, cliques = set(), set()
for node in per_node:
    for g in node.get("gpus", []):
        cu = g.get("cluster_uuid"); ci = g.get("clique_id")
        if cu and cu != "unknown": clusters.add(cu)
        if ci and ci != "unknown": cliques.add(ci)
print(f"[fault-inject] MNNVL POST: clusters={sorted(clusters)} cliques={sorted(cliques)}")
PY

# Per-node health summary line so the log makes the SUSPECT case obvious.
for h in "${MASTER_HOST}" "${WORKER_HOST}"; do
    log="${RUN_DIR_HOST}/health_${h}.log"
    if [[ -f "${log}" ]] && grep -q "CTX OK" "${log}"; then
        echo "[fault-inject] health ${h}: OK"
    else
        echo "[fault-inject] health ${h}: SUSPECT (see ${log})"
    fi
done

# ---------------------------------------------------------------------------
# Pass-gate assertions on the login side.
# ---------------------------------------------------------------------------
echo ""
echo "=== FAULT-INJECT PASS GATES ==="
python3 "${TEST_DIR_HOST}/fault_inject_pass_gate.py" \
    --run-dir "${RUN_DIR_HOST}" \
    --num-procs "${TOTAL_PROCS}" \
    --fault-inject-mode "${FAULT_INJECT_MODE}" \
    ${ALLOW_NO_OBSERVABLE_FAULT:+--allow-no-observable-fault}
gate_rc=$?

echo ""
echo "=========================================================================="
echo " NVLink fault-inject result: master_rc=${master_rc} worker_rc=${worker_rc} gate_rc=${gate_rc}"
echo " Artifacts: ${RUN_DIR_HOST}"
echo "=========================================================================="
exit $(( master_rc | worker_rc | gate_rc ))
