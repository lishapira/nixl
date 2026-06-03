#!/usr/bin/env bash
#SBATCH --account=network_research_advdev
#SBATCH --partition=gb300-backfill
#SBATCH --job-name=network_research_advdev-nixl.ep.inkernel-sigkill
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/network_research_advdev/lishapira/nixl/nixl_ep_nvlink_understanding/2nodes_sigkill_inkernel/results/slurm-%j.out

# Two-node GB200/GB300 run for NIXL EP in-kernel SIGKILL timing.
# Proves SIGKILL is sent after GPU entered, before GPU exited, in dispatch/combine
# send/receive, then checks surviving ranks continue communication. Build must
# include commit 4087c88 or later.
# Outputs per run directory:
#   SUMMARY.md: short setup/result summary.
#   topology.log: node/GPU topology.
#   build.log: build output.
#   <timing>.log: fault, evidence, cleanup, and post-baseline for that timing.
#   *_failure.log: only kept when an initial/final baseline fails.
# Pass: HIT_IN_KERNEL_WINDOW, exited_before_sigkill=0, no missed/timeout/errors,
# healthy ranks detect {2}, continue bandwidth, non-killed ranks done, baselines rc=0.
set -u -o pipefail

ROOT="/lustre/fsw/network_research_advdev/lishapira"
RUN_DIR="$ROOT/nixl/nixl_ep_nvlink_understanding/2nodes_sigkill_inkernel"
BASE_RESULTS_DIR="$RUN_DIR/results"
IMAGE="$ROOT/nixl-hybrid-ep-cuda2.sqsh"
MOUNTS="$ROOT:/workspace/lishapira"
ELASTIC="/workspace/lishapira/nixl/examples/device/ep/tests/elastic/elastic.py"
FAULT_PLAN="/workspace/lishapira/nixl/examples/device/ep/tests/elastic/expansion_fault_contraction_kill_2.json"
BASELINE_PLAN="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/2nodes_sigkill_inkernel/plans/expansion_contraction_no_fault_rank2.json"
SPIN_CYCLES="${SPIN_CYCLES:-100000000}"

mkdir -p "$BASE_RESULTS_DIR"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [[ "${#nodes[@]}" -lt 2 ]]; then
  echo "Expected two nodes, got: ${nodes[*]:-<none>}" >&2
  exit 2
fi
MASTER_NODE="${nodes[0]}"
WORKER_NODE="${nodes[1]}"
MASTER_ADDR="$MASTER_NODE"
JOB_ID="$SLURM_JOB_ID"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_job${JOB_ID}_${MASTER_NODE}_${WORKER_NODE}}"
RESULTS_DIR="$BASE_RESULTS_DIR/$RUN_ID"
RESULTS_DIR_CTR="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/2nodes_sigkill_inkernel/results/$RUN_ID"
SUMMARY="$RESULTS_DIR/SUMMARY.md"
mkdir -p "$RESULTS_DIR"

branch="$(git -C "$ROOT/nixl" branch --show-current)"
commit="$(git -C "$ROOT/nixl" log -1 --oneline)"
{
  echo "# Two-Node In-Kernel SIGKILL Run"
  echo
  echo "- Started UTC: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
  echo "- Job: \`$JOB_ID\`"
  echo "- Nodes: \`${nodes[*]}\`"
  echo "- Master: \`$MASTER_NODE\`"
  echo "- Worker: \`$WORKER_NODE\`"
  echo "- Branch: \`$branch\`"
  echo "- Commit: \`$commit\`"
  echo "- Spin cycles: \`$SPIN_CYCLES\`"
  echo "- Results: \`$RESULTS_DIR\`"
} > "$SUMMARY"

for node in "$MASTER_NODE" "$WORKER_NODE"; do
  {
    echo "===== topology ${node} ====="
    srun --overlap --jobid="$JOB_ID" -w "$node" -N1 -n1 \
      --container-image="$IMAGE" \
      --container-mounts="$MOUNTS" \
      --container-workdir=/workspace/lishapira \
      bash -lc 'hostname -f; nvidia-smi -L; nvidia-smi topo -m'
  } >> "$RESULTS_DIR/topology.log" 2>&1
done

echo "===== build on ${MASTER_NODE} ====="
srun --overlap --jobid="$JOB_ID" -w "$MASTER_NODE" -N1 -n1 \
  --container-image="$IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace/lishapira \
  bash -lc '/workspace/lishapira/build_nixl_aarch64.sh' \
  > "$RESULTS_DIR/build.log" 2>&1
build_rc=$?
if [[ "$build_rc" -ne 0 ]]; then
  echo "build failed; see $RESULTS_DIR/build.log" >&2
  {
    echo
    echo "- Final status: FAIL"
    echo "- Build rc: \`$build_rc\`"
    echo "- Failure detail: see \`build.log\`."
  } >> "$SUMMARY"
  exit "$build_rc"
fi
{
  echo "- Build rc: \`0\`"
  echo
  echo "| Timing | Fault rc | Hit | Exited before SIGKILL | Cleanup | Post baseline rc | Result |"
  echo "|---|---:|---|---:|---|---:|---|"
} >> "$SUMMARY"

run_elastic_pair() {
  local label="$1"
  local timeout_seconds="$2"
  local plan_path="$3"
  local log="$4"
  shift 4

  printf '\n===== %s metadata =====\njob=%s\nmaster=%s\nworker=%s\nplan=%s\ntimeout=%s\n' \
    "$label" "$JOB_ID" "$MASTER_NODE" "$WORKER_NODE" "$plan_path" "$timeout_seconds" >> "$log"

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
  } >> "$log"
}

cleanup_nodes() {
  local label="$1"
  local log="$2"
  local cleanup_rc=0
  printf '\n===== %s cleanup =====\n' "$label" >> "$log"
  for node in "$MASTER_NODE" "$WORKER_NODE"; do
    {
      echo "===== cleanup ${node} ====="
      srun --overlap --jobid="$JOB_ID" -w "$node" -N1 -n1 \
        --container-image="$IMAGE" \
        --container-mounts="$MOUNTS" \
        --container-workdir=/workspace/lishapira \
        bash -lc '
          echo "node=$(hostname -f)"
          echo "COMMAND: pgrep -af \"[e]lastic.py|[r]ank_server|[s]pawn_main\" || true"
          pgrep -af "[e]lastic.py|[r]ank_server|[s]pawn_main" || true
          pkill -9 -f "[e]lastic.py|[r]ank_server|[s]pawn_main" || true
          sleep 2
          echo "COMMAND: pgrep -af \"[e]lastic.py|[r]ank_server|[s]pawn_main\" || true"
          pgrep -af "[e]lastic.py|[r]ank_server|[s]pawn_main" || true
          echo "COMMAND: nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits"
          nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits || true
          echo "COMMAND: nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits"
          nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true
        '
    } 2>&1 | awk -v node="$node" '{ print strftime("%Y-%m-%dT%H:%M:%S"), "[" node "]", $0; fflush(); }' >> "$log"
    node_cleanup_rc=$?
    if [[ "$node_cleanup_rc" -ne 0 ]]; then
      cleanup_rc=1
    fi
  done
  if ! awk '
      /COMMAND: pgrep/ { pgrep_count++; in_after=(pgrep_count >= 2); next }
      /COMMAND: nvidia-smi --query-compute-apps/ { in_compute=1; in_after=0; next }
      /COMMAND: nvidia-smi --query-gpu/ { in_compute=0; next }
      in_after && /elastic.py|rank_server|spawn_main/ { bad=1 }
      in_compute && NF > 0 { bad=1 }
      END { exit bad ? 1 : 0 }
    ' "$log"; then
    cleanup_rc=1
  fi
  return "$cleanup_rc"
}

timings=(
  dispatch-send-during-kernel
  dispatch-receive-during-kernel
  combine-send-during-kernel
  combine-receive-during-kernel
)

echo "===== initial baseline ====="
initial_log="$RESULTS_DIR/initial_baseline_failure.log"
run_elastic_pair initial_baseline 240 "$BASELINE_PLAN" "$initial_log"
initial_baseline_rc=$?
if [[ "$initial_baseline_rc" -ne 0 ]]; then
  echo "initial baseline failed; stopping" >&2
  echo >> "$SUMMARY"
  echo "Initial baseline failed with rc \`$initial_baseline_rc\`; see \`$(basename "$initial_log")\`." >> "$SUMMARY"
  exit "$initial_baseline_rc"
fi
rm -f "$initial_log"

for timing in "${timings[@]}"; do
  timing_log="$RESULTS_DIR/${timing}.log"
  rm -f "$timing_log" "$RESULTS_DIR"/in_kernel_fault_*_"$timing"_*.log
  echo "===== timing ${timing}: fault run ====="
  run_elastic_pair "fault_${timing}" 180 "$FAULT_PLAN" \
    "$timing_log" \
    --fault-kill-signal sigkill \
    --fault-kill-timing "$timing" \
    --in-kernel-fault-spin-cycles "$SPIN_CYCLES" \
    --fault-evidence-dir "$RESULTS_DIR_CTR"
  fault_rc=$?
  printf '\nFAULT_RC=%s\n' "$fault_rc" >> "$timing_log"

  echo "===== timing ${timing}: cleanup ====="
  cleanup_nodes "fault_${timing}" "$timing_log"
  cleanup_rc=$?

  hit_files=("$RESULTS_DIR"/in_kernel_fault_*_"$timing"_*.log)
  hit="no"
  exited_before="NA"
  if [[ -e "${hit_files[0]}" ]]; then
    {
      echo
      echo "===== in-kernel evidence ====="
      sed 's/^/evidence: /' "${hit_files[@]}"
    } >> "$timing_log"
    grep -q "HIT_IN_KERNEL_WINDOW" "${hit_files[@]}" && hit="yes"
    exited_before="$(sed -n 's/^exited_before_sigkill=//p' "${hit_files[@]}" | head -n 1)"
  fi
  if [[ "$hit" != "yes" ]]; then
    echo "missing HIT_IN_KERNEL_WINDOW evidence for ${timing}; stopping sweep" >&2
    echo "| \`$timing\` | $fault_rc | $hit | $exited_before | rc=$cleanup_rc | NA | FAIL missing hit |" >> "$SUMMARY"
    exit 1
  fi
  if grep -E -q "MISSED_IN_KERNEL_TIMING|IN_KERNEL_MARKER_TIMEOUT" "${hit_files[@]}"; then
    echo "missed in-kernel timing evidence found for ${timing}; stopping sweep" >&2
    echo "| \`$timing\` | $fault_rc | $hit | $exited_before | rc=$cleanup_rc | NA | FAIL missed timing |" >> "$SUMMARY"
    exit 1
  fi
  rm -f -- "${hit_files[@]}"
  if [[ "$cleanup_rc" -ne 0 ]]; then
    echo "cleanup checks failed after ${timing}; stopping sweep" >&2
    echo "| \`$timing\` | $fault_rc | $hit | $exited_before | FAIL | NA | FAIL cleanup |" >> "$SUMMARY"
    exit 1
  fi

  echo "===== timing ${timing}: post baseline ====="
  run_elastic_pair "post_baseline_${timing}" 240 "$BASELINE_PLAN" "$timing_log"
  baseline_rc=$?
  printf '\nPOST_BASELINE_RC=%s\n' "$baseline_rc" >> "$timing_log"
  if [[ "$baseline_rc" -ne 0 ]]; then
    echo "post baseline failed after ${timing}; stopping sweep" >&2
    echo "| \`$timing\` | $fault_rc | $hit | $exited_before | PASS | $baseline_rc | FAIL baseline |" >> "$SUMMARY"
    exit "$baseline_rc"
  fi
  echo "| \`$timing\` | $fault_rc | $hit | $exited_before | PASS | $baseline_rc | PASS |" >> "$SUMMARY"
done

echo "===== final cleanup ====="
final_log="$RESULTS_DIR/final_baseline_failure.log"
cleanup_nodes final "$final_log"
final_cleanup_rc=$?
echo "===== final baseline ====="
run_elastic_pair final_baseline 240 "$BASELINE_PLAN" "$final_log"
final_rc=$?
{
  echo
  if [[ "$final_cleanup_rc" -eq 0 && "$final_rc" -eq 0 ]]; then
    echo "- Final status: PASS"
    echo "- Final cleanup: PASS"
    echo "- Final baseline rc: \`0\`"
    rm -f "$final_log"
  else
    echo "- Final status: FAIL"
    echo "- Final cleanup rc: \`$final_cleanup_rc\`"
    echo "- Final baseline rc: \`$final_rc\`"
    echo "- Failure detail: see \`$(basename "$final_log")\`."
  fi
} >> "$SUMMARY"
exit "$final_rc"
