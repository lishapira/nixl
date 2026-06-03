#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <label> <timeout-seconds> <plan-container-path> [extra-elastic-args...]" >&2
  exit 2
fi

LABEL="$1"
RUN_TIMEOUT="$2"
PLAN_PATH="$3"
shift 3

RUN_DIR="${RUN_DIR:-/lustre/fsw/network_research_advdev/lishapira/nixl/nixl_ep_nvlink_understanding/old_2nodes_sigkill_non_inkernel}"
RESULTS_DIR="${RESULTS_DIR:-$RUN_DIR/results}"
JOB_ID="${JOB_ID:-1857351}"
MASTER_NODE="${MASTER_NODE:-theia0153}"
WORKER_NODE="${WORKER_NODE:-theia0156}"
MASTER_ADDR="${MASTER_ADDR:-$MASTER_NODE}"
IMAGE="/lustre/fsw/network_research_advdev/lishapira/nixl-hybrid-ep-cuda2.sqsh"
MOUNTS="/lustre/fsw/network_research_advdev/lishapira:/workspace/lishapira"
ELASTIC="/workspace/lishapira/nixl/examples/device/ep/tests/elastic/elastic.py"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/${LABEL}.log"
RC_LOG="$RESULTS_DIR/${LABEL}.rc"

rm -f "$LOG" "$RC_LOG"

printf 'label=%s\njob=%s\nmaster=%s\nworker=%s\nplan=%s\ntimeout=%s\nextra_args=%q' \
  "$LABEL" "$JOB_ID" "$MASTER_NODE" "$WORKER_NODE" "$PLAN_PATH" "$RUN_TIMEOUT" "$*" > "$RC_LOG"
printf '\n' >> "$RC_LOG"

run_node() {
  local node="$1"
  local role="$2"
  shift 2
  local -a args=("$@")

  {
    echo "===== ${role} ${node} command ====="
    printf 'python3 %q --plan %q --num-processes 4' "$ELASTIC" "$PLAN_PATH"
    for arg in "${args[@]}"; do
      printf ' %q' "$arg"
    done
    echo
    echo "===== ${role} ${node} output ====="
    timeout --kill-after=20s "${RUN_TIMEOUT}s" \
      srun --overlap --jobid="$JOB_ID" -w "$node" -N1 -n1 \
        --container-image="$IMAGE" \
        --container-mounts="$MOUNTS" \
        --container-workdir=/workspace/lishapira \
        bash -lc 'source /workspace/lishapira/setup_node.sh; export PYTHONUNBUFFERED=1; python3 "$@"' \
        bash "$ELASTIC" --plan "$PLAN_PATH" --num-processes 4 "${args[@]}"
    local rc=$?
    echo "===== ${role} ${node} rc=${rc} ====="
    return "$rc"
  } 2>&1 | awk -v role="$role" -v node="$node" '{ print strftime("%Y-%m-%dT%H:%M:%S"), "[" role ":" node "]", $0; fflush(); }' >> "$LOG"
}

run_node "$MASTER_NODE" master "$@" &
master_pid=$!
sleep 6
run_node "$WORKER_NODE" worker --tcp-server "$MASTER_ADDR" "$@" &
worker_pid=$!

wait "$master_pid"
master_rc=$?
wait "$worker_pid"
worker_rc=$?

{
  echo "master_rc=${master_rc}"
  echo "worker_rc=${worker_rc}"
  if [[ "$master_rc" -eq 0 && "$worker_rc" -eq 0 ]]; then
    echo "overall_rc=0"
    exit 0
  fi
  if [[ "$master_rc" -eq 124 || "$worker_rc" -eq 124 ]]; then
    echo "overall_rc=124"
    exit 124
  fi
  echo "overall_rc=1"
  exit 1
} >> "$RC_LOG"
