#!/usr/bin/env bash
set -u -o pipefail

# Run the four in-kernel SIGKILL coverage cases in an existing two-node Slurm
# allocation. Override JOB_ID, MASTER_NODE, WORKER_NODE, or SPIN_CYCLES if needed.

ROOT="/lustre/fsw/network_research_advdev/lishapira"
RUN_ROOT="$ROOT/nixl/nixl_ep_nvlink_understanding/common/legacy_helpers"
IMAGE="$ROOT/nixl-hybrid-ep-cuda2.sqsh"
MOUNTS="$ROOT:/workspace/lishapira"
ELASTIC="/workspace/lishapira/nixl/examples/device/ep/tests/elastic/elastic.py"
FAULT_PLAN="/workspace/lishapira/nixl/examples/device/ep/tests/elastic/expansion_fault_contraction_kill_2.json"
BASELINE_PLAN="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/common/plans/expansion_contraction_no_fault_rank2.json"
SPIN_CYCLES="${SPIN_CYCLES:-100000000}"

JOB_ID="${JOB_ID:-${SLURM_JOB_ID:-}}"
if [[ -z "$JOB_ID" ]]; then
  echo "JOB_ID is required. Run inside salloc or export JOB_ID=<slurm-job-id>." >&2
  exit 2
fi

if [[ -z "${MASTER_NODE:-}" || -z "${WORKER_NODE:-}" ]]; then
  nodes=($(scontrol show hostnames "$(squeue -j "$JOB_ID" --noheader --format=%N)"))
  if [[ "${#nodes[@]}" -lt 2 ]]; then
    echo "Expected at least two nodes for job $JOB_ID, got: ${nodes[*]:-<none>}" >&2
    exit 2
  fi
  MASTER_NODE="${MASTER_NODE:-${nodes[0]}}"
  WORKER_NODE="${WORKER_NODE:-${nodes[1]}}"
fi
MASTER_ADDR="${MASTER_ADDR:-$MASTER_NODE}"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="$RUN_ROOT/phase3_in_kernel_sigkill_job${JOB_ID}_${MASTER_NODE}_${WORKER_NODE}_${timestamp}"
RESULTS_DIR="$RUN_DIR/results"
RUN_DIR_CTR="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/common/legacy_helpers/$(basename "$RUN_DIR")"
RESULTS_DIR_CTR="$RUN_DIR_CTR/results"
mkdir -p "$RESULTS_DIR"

echo "run_dir=$RUN_DIR"
echo "job_id=$JOB_ID"
echo "master=$MASTER_NODE"
echo "worker=$WORKER_NODE"
echo "spin_cycles=$SPIN_CYCLES"

run_elastic_pair() {
  local label="$1"
  local timeout_seconds="$2"
  local plan_path="$3"
  shift 3
  local log="$RESULTS_DIR/${label}.log"
  local rc_log="$RESULTS_DIR/${label}.rc"
  rm -f "$log" "$rc_log"

  {
    printf 'label=%s\njob=%s\nmaster=%s\nworker=%s\nplan=%s\ntimeout=%s\n' \
      "$label" "$JOB_ID" "$MASTER_NODE" "$WORKER_NODE" "$plan_path" "$timeout_seconds"
  } > "$rc_log"

  run_node() {
    local node="$1"
    local role="$2"
    shift 2
    local -a args=("$@")
    {
      echo "===== ${role} ${node} command ====="
      printf 'python3 %q --plan %q --num-processes 4' "$ELASTIC" "$plan_path"
      for arg in "${args[@]}"; do
        printf ' %q' "$arg"
      done
      echo
      echo "===== ${role} ${node} output ====="
      timeout --kill-after=20s "${timeout_seconds}s" \
        srun --overlap --jobid="$JOB_ID" -w "$node" -N1 -n1 \
          --container-image="$IMAGE" \
          --container-mounts="$MOUNTS" \
          --container-workdir=/workspace/lishapira \
          bash -lc 'source /workspace/lishapira/setup_node.sh; export PYTHONUNBUFFERED=1; python3 "$@"' \
          bash "$ELASTIC" --plan "$plan_path" --num-processes 4 "${args[@]}"
      local rc=$?
      echo "===== ${role} ${node} rc=${rc} ====="
      return "$rc"
    } 2>&1 | awk -v role="$role" -v node="$node" '{ print strftime("%Y-%m-%dT%H:%M:%S"), "[" role ":" node "]", $0; fflush(); }' >> "$log"
  }

  run_node "$MASTER_NODE" master "$@" &
  local master_pid=$!
  sleep 6
  run_node "$WORKER_NODE" worker --tcp-server "$MASTER_ADDR" "$@" &
  local worker_pid=$!

  wait "$master_pid"
  local master_rc=$?
  wait "$worker_pid"
  local worker_rc=$?

  {
    echo "master_rc=${master_rc}"
    echo "worker_rc=${worker_rc}"
    if [[ "$master_rc" -eq 0 && "$worker_rc" -eq 0 ]]; then
      echo "overall_rc=0"
      return 0
    fi
    if [[ "$master_rc" -eq 124 || "$worker_rc" -eq 124 ]]; then
      echo "overall_rc=124"
      return 124
    fi
    echo "overall_rc=1"
    return 1
  } >> "$rc_log"
}

cleanup_nodes() {
  local label="$1"
  local log="$RESULTS_DIR/${label}_cleanup.log"
  rm -f "$log"
  for node in "$MASTER_NODE" "$WORKER_NODE"; do
    {
      echo "===== cleanup ${node} ====="
      srun --overlap --jobid="$JOB_ID" -w "$node" -N1 -n1 \
        --container-image="$IMAGE" \
        --container-mounts="$MOUNTS" \
        --container-workdir=/workspace/lishapira \
        bash -lc '
          echo "COMMAND: pgrep -af \"[e]lastic.py|[r]ank_server|[s]pawn_main\" || true"
          pgrep -af "[e]lastic.py|[r]ank_server|[s]pawn_main" || true
          pkill -9 -f "[e]lastic.py|[r]ank_server|[s]pawn_main" || true
          echo "COMMAND: nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits"
          nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits || true
          echo "COMMAND: nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
          nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits || true
        '
    } 2>&1 | awk -v node="$node" '{ print strftime("%Y-%m-%dT%H:%M:%S"), "[" node "]", $0; fflush(); }' >> "$log"
  done
}

timings=(
  dispatch-send-during-kernel
  dispatch-receive-during-kernel
  combine-send-during-kernel
  combine-receive-during-kernel
)

for timing in "${timings[@]}"; do
  echo "===== timing ${timing}: fault run ====="
  run_elastic_pair "fault_${timing}" 180 "$FAULT_PLAN" \
    --fault-kill-signal sigkill \
    --fault-kill-timing "$timing" \
    --in-kernel-fault-spin-cycles "$SPIN_CYCLES" \
    --fault-evidence-dir "$RESULTS_DIR_CTR"
  fault_rc=$?
  echo "fault_${timing}_rc=${fault_rc}"

  echo "===== timing ${timing}: cleanup ====="
  cleanup_nodes "fault_${timing}"

  hit_files=("$RESULTS_DIR"/in_kernel_fault_*_"$timing"_*.log)
  if [[ ! -e "${hit_files[0]}" ]] || ! rg -q "HIT_IN_KERNEL_WINDOW" "${hit_files[@]}"; then
    echo "missing HIT_IN_KERNEL_WINDOW evidence for ${timing}; stopping sweep" >&2
    exit 1
  fi
  if rg -q "MISSED_IN_KERNEL_TIMING|IN_KERNEL_MARKER_TIMEOUT" "${hit_files[@]}"; then
    echo "missed in-kernel timing evidence found for ${timing}; stopping sweep" >&2
    exit 1
  fi

  echo "===== timing ${timing}: post baseline ====="
  run_elastic_pair "post_baseline_${timing}" 240 "$BASELINE_PLAN"
  baseline_rc=$?
  echo "post_baseline_${timing}_rc=${baseline_rc}"
  if [[ "$baseline_rc" -ne 0 ]]; then
    echo "post baseline failed after ${timing}; stopping sweep" >&2
    exit "$baseline_rc"
  fi
done

echo "===== final cleanup ====="
cleanup_nodes final
echo "===== final baseline ====="
run_elastic_pair final_baseline 240 "$BASELINE_PLAN"
