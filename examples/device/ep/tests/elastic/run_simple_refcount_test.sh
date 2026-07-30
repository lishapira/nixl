#!/usr/bin/env bash
#
# Simple 2-peer / 2-node cross-node VMM fabric refcount test — runner.
#
# Runs simple_refcount_test.py in BOTH death modes back-to-back on the
# same 2-node allocation:
#
#   Variant 1 (release):  victim runs full cuMemUnmap + cuMemAddressFree
#                         + cuMemRelease (elastic SIGTERM path)
#   Variant 2 (sigkill):  victim self-SIGKILLs (elastic
#                         --fault-kill-signal=sigkill path)
#
# Only 2 processes per variant (one per node). No poisoner. After victim
# dies, survivor waits 15s then verifies it can still read/write.
#
# Usage (from login-node shell, outside any container):
#   NODE_A=<host1> NODE_B=<host2> SLURM_JOB_ID=<id> \
#       bash nixl/examples/device/ep/tests/elastic/run_simple_refcount_test.sh <SLURM_JOB_ID>
#
# Optional env:
#   MODES        - space-separated list of modes to run
#                  (default: "release sigkill")
#   PORT_BASE    - TCP port for variant 1 (default 27184). Variant 2
#                  uses PORT_BASE+1 to avoid TIME_WAIT collisions.
#   SIZE_BYTES   - per-allocation size (default 64 MiB)
#   PROBE_BYTES  - bytes filled/read per proof (default 8 MiB)
#   DEATH_WAIT   - seconds survivor waits after victim signals death (default 15)
#   DEVICE       - CUDA device index (default 0)
#   TCP_TIMEOUT  - TCP connect timeout in s (default 120)
#   HARD_TIMEOUT - seconds each srun is allowed to run (default 300)
#   OUT          - log directory (default ~/runs/simple_refcount_<ts>)
#

set -euo pipefail

JOBID="${1:-${SLURM_JOB_ID:-}}"
if [ -z "$JOBID" ]; then
    echo "ERROR: SLURM_JOB_ID not set. Pass jobid as first arg, or export SLURM_JOB_ID." >&2
    exit 2
fi
export SLURM_JOB_ID="$JOBID"

MODES="${MODES:-release sigkill}"
PORT_BASE="${PORT_BASE:-27184}"
SIZE_BYTES="${SIZE_BYTES:-$((64 * 1024 * 1024))}"
PROBE_BYTES="${PROBE_BYTES:-$((8 * 1024 * 1024))}"
DEATH_WAIT="${DEATH_WAIT:-15}"
DEVICE="${DEVICE:-0}"
TCP_TIMEOUT="${TCP_TIMEOUT:-120}"
HARD_TIMEOUT="${HARD_TIMEOUT:-300}"

TS=$(date +%Y%m%d_%H%M%S)
OUT="${OUT:-/lustre/fsw/network_research_advdev/lishapira/runs/simple_refcount_${TS}}"
mkdir -p "$OUT"

SQSH=/lustre/fsw/network_research_advdev/lishapira/nixl-hybrid-ep-cuda2.sqsh
CONT_MOUNTS=/lustre/fsw/network_research_advdev/lishapira:/workspace/lishapira
SCRIPT=/workspace/lishapira/nixl/examples/device/ep/tests/elastic/simple_refcount_test.py

# --- resolve node list (short-circuit on NODE_A/NODE_B env) ---
declare -a NODES_ARR=()
if [ -n "${NODE_A:-}" ] && [ -n "${NODE_B:-}" ]; then
    NODES_ARR=( "$NODE_A" "$NODE_B" )
    NODELIST_RAW="${NODE_A},${NODE_B}"
else
    NODELIST_RAW=""
    if command -v scontrol >/dev/null 2>&1; then
        NODELIST_RAW=$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null \
                       | grep -oE ' NodeList=[^ ]+' | head -1 \
                       | sed -E 's/^ NodeList=//')
    fi
    if [ -z "$NODELIST_RAW" ]; then
        NODELIST_RAW="${SLURM_JOB_NODELIST:-}"
    fi
    if command -v scontrol >/dev/null 2>&1 && [ -n "$NODELIST_RAW" ]; then
        mapfile -t NODES_ARR < <(scontrol show hostnames "$NODELIST_RAW" 2>/dev/null || true)
    fi
fi
if [ "${#NODES_ARR[@]}" -lt 2 ]; then
    echo "ERROR: need at least 2 nodes; got ${#NODES_ARR[@]}: ${NODES_ARR[*]:-<none>}" >&2
    exit 3
fi
VICTIM_HOST="${NODES_ARR[0]}"
SURVIVOR_HOST="${NODES_ARR[1]}"

SETUP="source /workspace/lishapira/setup_node.sh >/dev/null 2>&1"

echo "SLURM job     : $SLURM_JOB_ID"
echo "SLURM nodes   : $NODELIST_RAW"
echo "Victim node   : $VICTIM_HOST"
echo "Survivor node : $SURVIVOR_HOST"
echo "Modes         : $MODES"
echo "Size bytes    : $SIZE_BYTES"
echo "Probe bytes   : $PROBE_BYTES"
echo "Death wait    : ${DEATH_WAIT}s"
echo "Output dir    : $OUT"
echo

# ----------------------------------------------------------------------
run_one_variant() {
    local mode="$1"
    local port="$2"
    local subdir="$OUT/$mode"
    mkdir -p "$subdir"
    local vlog="$subdir/victim_${VICTIM_HOST}.log"
    local slog="$subdir/survivor_${SURVIVOR_HOST}.log"

    echo "========================================================="
    echo "VARIANT: death_mode=$mode  port=$port"
    echo "========================================================="

    local cmd_victim="$SETUP; python3 $SCRIPT \
        --role victim --port $port --device $DEVICE \
        --death-mode $mode \
        --size-bytes $SIZE_BYTES --probe-bytes $PROBE_BYTES \
        --tcp-timeout $TCP_TIMEOUT"

    local cmd_survivor="$SETUP; python3 $SCRIPT \
        --role survivor --peer-host $VICTIM_HOST \
        --port $port --device $DEVICE \
        --death-mode $mode --death-wait $DEATH_WAIT \
        --size-bytes $SIZE_BYTES --probe-bytes $PROBE_BYTES \
        --tcp-timeout $TCP_TIMEOUT"

    echo "[launch] VICTIM   on $VICTIM_HOST   -> $vlog"
    timeout "$HARD_TIMEOUT" srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --ntasks=1 --nodelist="$VICTIM_HOST" \
        --container-image="$SQSH" \
        --container-mounts="$CONT_MOUNTS" \
        --container-workdir=/workspace/lishapira \
        --export=ALL \
        bash -c "$cmd_victim" > "$vlog" 2>&1 &
    local victim_srun_pid=$!

    sleep 5

    echo "[launch] SURVIVOR on $SURVIVOR_HOST -> $slog"
    set +e
    timeout "$HARD_TIMEOUT" srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --ntasks=1 --nodelist="$SURVIVOR_HOST" \
        --container-image="$SQSH" \
        --container-mounts="$CONT_MOUNTS" \
        --container-workdir=/workspace/lishapira \
        --export=ALL \
        bash -c "$cmd_survivor" 2>&1 | tee "$slog"
    local survivor_rc=${PIPESTATUS[0]}
    set -e

    wait "$victim_srun_pid" 2>/dev/null || true
    local victim_rc=$?

    echo
    echo "----- VICTIM log tail ($vlog) -----"
    tail -20 "$vlog" 2>/dev/null || echo "(no log)"
    echo
    echo "victim_rc=$victim_rc  survivor_rc=$survivor_rc"
    echo

    if [ "$survivor_rc" -eq 0 ]; then
        echo "[$mode] PASS"
        SUMMARY+=( "$mode:PASS" )
    else
        echo "[$mode] FAIL (survivor_rc=$survivor_rc)"
        SUMMARY+=( "$mode:FAIL" )
    fi
    echo
}

SUMMARY=()
PORT="$PORT_BASE"
for mode in $MODES; do
    if [ "$mode" != "release" ] && [ "$mode" != "sigkill" ]; then
        echo "ERROR: unknown mode '$mode'; expected 'release' or 'sigkill'" >&2
        exit 4
    fi
    run_one_variant "$mode" "$PORT"
    PORT=$((PORT + 1))
done

echo "========================================================="
echo "COMBINED RESULT"
echo "========================================================="
FINAL_RC=0
for entry in "${SUMMARY[@]}"; do
    echo "  $entry"
    if [[ "$entry" == *":FAIL" ]]; then
        FINAL_RC=1
    fi
done
echo "========================================================="
echo "logs: $OUT"
if [ "$FINAL_RC" -eq 0 ]; then
    echo "[simple-refcount] OVERALL PASS: cross-node refcount HOLDS under all tested death modes"
else
    echo "[simple-refcount] OVERALL FAIL: at least one variant reported a failing proof"
fi
exit "$FINAL_RC"
