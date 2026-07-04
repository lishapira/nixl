#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Phase 5: real unmap injection across TWO nodes over MNNVL.
#
# Fast-path design (4 srun steps total, not 12):
#   1. srun -N2 --ntasks-per-node=1  =>  parallel PRE probe on both nodes
#      (MNNVL fabric + NVLink counter snapshot)
#   2. srun -N1 on master              =>  elastic.py (rank server + master ranks)
#   3. srun -N1 on worker (bg)         =>  elastic.py --tcp-server master
#   4. srun -N2 --ntasks-per-node=1  =>  parallel POST probe on both nodes
#      (MNNVL + NVLink counters + torch.cuda health probe)
#
# Each container-start on Lyris costs ~30 s. Twelve sruns = 6 min of
# pure overhead which was blowing the entire preemption window on the
# gb200-backfill partition. Four sruns = ~2 min overhead.
#
# Pass gates (see phase5_pass_gate.py):
#   * MNNVL clique intact PRE and POST (fabric not corrupted)
#   * Victim rank 2 role=victim, inject_mode=unmap-mid-flight, survived
#   * Every peer role=peer, survived
#   * At least ONE observability signal: XID on peer, IMEX error on peer,
#     or positive NVLink counter delta
#
# Required env (from an active salloc):
#   SLURM_JOB_ID
#   SLURM_JOB_NODELIST   (>= 2 nodes)
#
# Optional env:
#   TIMING      elastic.py --fault-kill-timing (default: before-dispatch)
#   NPROCS_PN   workers per node (default: 4)
#   PLAN_FILE   default: nvlink_fault_tolerance_2node_unmap.json
#   IN_KERNEL_SPIN_CYCLES  GPU spin cycles after in-kernel entered marker
#                          (default 1,000,000 ~ 500 us on GB200). Ignored for
#                          CPU-level timings (before-dispatch, after-dispatch,
#                          etc.). Give the host helper thread time to observe
#                          entered && !exited and fire the injection while the
#                          kernel is still in the marked window.
#   PHASE5_ALLOW_NO_FAULT=1  downgrade observability gate to WARN

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
RUN_DIR_HOST="${TEST_DIR_HOST}/results/phase5_2node_${UTC}_${MASTER_HOST}_${WORKER_HOST}_${TIMING}"
RUN_DIR_CONT="${TEST_DIR_CONT}/results/phase5_2node_${UTC}_${MASTER_HOST}_${WORKER_HOST}_${TIMING}"
mkdir -p "${RUN_DIR_HOST}"
chmod 2777 "${RUN_DIR_HOST}" 2>/dev/null || true

echo "=========================================================================="
echo " Phase 5: REAL unmap fault across 2 nodes over MNNVL (fast-path)"
echo "   job        = ${SLURM_JOB_ID}"
echo "   master     = ${MASTER_HOST}"
echo "   worker     = ${WORKER_HOST}"
echo "   plan       = ${PLAN_FILE}  (victim rank = 2, on master)"
echo "   timing     = ${TIMING}"
echo "   spin_cycles= ${IN_KERNEL_SPIN_CYCLES} (in-kernel only)"
echo "   nprocs/nod = ${NPROCS_PN}   total procs = ${TOTAL_PROCS}"
echo "   RUN_DIR    = ${RUN_DIR_HOST}"
echo "=========================================================================="

PROBE_SCRIPT="${TEST_DIR_CONT}/phase5_node_probe.sh"

# ---------------------------------------------------------------------------
# STEP 1: PRE probe (parallel on both nodes in one srun step)
# ---------------------------------------------------------------------------
echo "[phase5] STEP 1/4: PRE probe (MNNVL + NVLink counters, parallel on both nodes)..."
srun --jobid="${SLURM_JOB_ID}" --overlap \
    --nodes=2 --ntasks-per-node=1 --nodelist="${MASTER_HOST},${WORKER_HOST}" \
    --container-image="${SQSH}" \
    --container-mounts="${CONT_MOUNTS}" \
    --container-workdir=/workspace/lishapira \
    --export=ALL,RUN_DIR="${RUN_DIR_CONT}",STAGE=pre,TEST_DIR="${TEST_DIR_CONT}" \
    bash "${PROBE_SCRIPT}" 2>&1 | sed 's/^/  /' \
    || { echo "[phase5] PRE probe failed" >&2; exit 3; }

# Aggregate PRE artifacts on the login side into the shapes phase5_pass_gate expects.
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
print(f"[phase5] MNNVL PRE: clusters={sorted(clusters)} cliques={sorted(cliques)}")
if len(clusters) != 1 or len(cliques) != 1:
    print("[phase5] FAIL: master and worker are NOT MNNVL-coupled (different Cluster/Clique)")
    print("[phase5] Refusing to run Phase 5 -- unmap on non-MNNVL nodes tells us nothing.")
    sys.exit(2)
print("[phase5] MNNVL PRE OK")
PY
mnnvl_rc=$?
if (( mnnvl_rc != 0 )); then
    echo "[phase5] aborting: MNNVL pre-check failed (rc=${mnnvl_rc})"
    exit "${mnnvl_rc}"
fi
pre_lines=$(wc -l < "${RUN_DIR_HOST}/nvlink_pre.csv" 2>/dev/null || echo 0)
echo "[phase5] PRE artifacts: nvlink_pre.csv=${pre_lines} lines, mnnvl_pre.json written"

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
echo \"[phase5-\$(hostname -s)] launching elastic.py ${tcp_arg}\"
python3 -u elastic.py \\
    --plan ${PLAN_FILE} \\
    --num-processes ${NPROCS_PN} \\
    --fault-kill-signal sigkill \\
    --fault-kill-timing ${TIMING} \\
    --fault-inject-mode unmap-mid-flight \\
    --in-kernel-fault-spin-cycles ${IN_KERNEL_SPIN_CYCLES} \\
    --fault-evidence-dir \"\${FAULT_EVIDENCE_DIR}\" \\
    ${tcp_arg}
rc=\$?
echo \"[phase5-\$(hostname -s)] elastic rc=\${rc}\"
exit \${rc}
" > "${log_file}" 2>&1 &
    LAUNCH_BG_PID=$!
}

cleanup_bg() {
    # Kill ONLY our own srun clients, not the parent allocation.
    for p in "${MASTER_PID:-}" "${WORKER_PID:-}"; do
        [[ -z "$p" ]] && continue
        if kill -0 "$p" 2>/dev/null; then
            echo "[phase5] cleanup: killing bg srun client pid=$p"
            kill -TERM "$p" 2>/dev/null || true
        fi
    done
}
# EXIT trap only -- do NOT trap INT/TERM so ^C on the orchestrator does
# not propagate SIGTERM to the srun clients we launched (that could
# cascade back to the allocation shell if slurm relays it).
trap 'cleanup_bg' EXIT

echo "[phase5] STEP 2/4: launching MASTER on ${MASTER_HOST}..."
launch_elastic "${MASTER_HOST}" "" "${MASTER_LOG}"
MASTER_PID="${LAUNCH_BG_PID}"

MASTER_READY_TIMEOUT=90
deadline=$(( $(date +%s) + MASTER_READY_TIMEOUT ))
master_ready=0
n_reg=0
while (( $(date +%s) < deadline )); do
    if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
        echo "[phase5] master died during head-start; log tail:" >&2
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
    echo "[phase5] master not ready (${n_reg}/${NPROCS_PN}) after ${MASTER_READY_TIMEOUT}s" >&2
    tail -n 60 "${MASTER_LOG}" | sed 's/^/  /' >&2
    kill -TERM "${MASTER_PID}" 2>/dev/null || true
    wait "${MASTER_PID}" 2>/dev/null || true
    MASTER_PID=""
    exit 5
fi

echo "[phase5] master ready (${n_reg}/${NPROCS_PN})"
echo "[phase5] STEP 3/4: launching WORKER on ${WORKER_HOST}..."
launch_elastic "${WORKER_HOST}" "${MASTER_HOST}" "${WORKER_LOG}"
WORKER_PID="${LAUNCH_BG_PID}"

wait "${WORKER_PID}"; worker_rc=$?
WORKER_PID=""
wait "${MASTER_PID}"; master_rc=$?
MASTER_PID=""

echo "[phase5] MASTER rc=${master_rc}   WORKER rc=${worker_rc}"

# Best-effort settle for driver state.
sleep 5

# ---------------------------------------------------------------------------
# STEP 4: POST probe (parallel on both nodes in one srun step)
# ---------------------------------------------------------------------------
echo "[phase5] STEP 4/4: POST probe (MNNVL + NVLink + health, parallel)..."
srun --jobid="${SLURM_JOB_ID}" --overlap \
    --nodes=2 --ntasks-per-node=1 --nodelist="${MASTER_HOST},${WORKER_HOST}" \
    --container-image="${SQSH}" \
    --container-mounts="${CONT_MOUNTS}" \
    --container-workdir=/workspace/lishapira \
    --export=ALL,RUN_DIR="${RUN_DIR_CONT}",STAGE=post,TEST_DIR="${TEST_DIR_CONT}" \
    bash "${PROBE_SCRIPT}" 2>&1 | sed 's/^/  /' \
    || echo "[phase5] POST probe partial failure (continuing to gate)"

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
print(f"[phase5] MNNVL POST: clusters={sorted(clusters)} cliques={sorted(cliques)}")
PY

# Per-node health summary line so the log makes the SUSPECT case obvious.
for h in "${MASTER_HOST}" "${WORKER_HOST}"; do
    log="${RUN_DIR_HOST}/health_${h}.log"
    if [[ -f "${log}" ]] && grep -q "CTX OK" "${log}"; then
        echo "[phase5] health ${h}: OK"
    else
        echo "[phase5] health ${h}: SUSPECT (see ${log})"
    fi
done

# ---------------------------------------------------------------------------
# Pass-gate assertions on the login side.
# ---------------------------------------------------------------------------
echo ""
echo "=== PHASE 5 PASS GATES ==="
python3 "${TEST_DIR_HOST}/phase5_pass_gate.py" \
    --run-dir "${RUN_DIR_HOST}" \
    --num-procs "${TOTAL_PROCS}" \
    ${PHASE5_ALLOW_NO_FAULT:+--allow-no-observable-fault}
gate_rc=$?

echo ""
echo "=========================================================================="
echo " Phase 5 result: master_rc=${master_rc} worker_rc=${worker_rc} gate_rc=${gate_rc}"
echo " Artifacts: ${RUN_DIR_HOST}"
echo "=========================================================================="
exit $(( master_rc | worker_rc | gate_rc ))
