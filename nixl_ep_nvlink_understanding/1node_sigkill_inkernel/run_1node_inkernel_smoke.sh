#!/usr/bin/env bash
# One-node GB200/GB300 run for NIXL EP in-kernel SIGKILL timing.
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
RUN_DIR="$ROOT/nixl/nixl_ep_nvlink_understanding/1node_sigkill_inkernel"
BASE_RESULTS_DIR="$RUN_DIR/results"
IMAGE="$ROOT/nixl-hybrid-ep-cuda2.sqsh"
MOUNTS="$ROOT:/workspace/lishapira"
ELASTIC="/workspace/lishapira/nixl/examples/device/ep/tests/elastic/elastic.py"
FAULT_PLAN="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/1node_sigkill_inkernel/plans/one_node_fault_rank2.json"
BASELINE_PLAN="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/1node_sigkill_inkernel/plans/one_node_baseline.json"
SPIN_CYCLES="${SPIN_CYCLES:-100000000}"

mkdir -p "$BASE_RESULTS_DIR"

JOB_ID="${JOB_ID:-${SLURM_JOB_ID:-}}"
NODE="${NODE:-}"
if [[ -z "$NODE" && -n "${SLURM_JOB_NODELIST:-}" ]]; then
  nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
  NODE="${nodes[0]:-}"
fi
if [[ -z "$NODE" ]]; then
  NODE="$(hostname -s)"
fi
if [[ -z "$JOB_ID" ]]; then
  echo "JOB_ID is required. Run inside a Slurm allocation or set JOB_ID=<allocation_id>." >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_job${JOB_ID}_${NODE}}"
RESULTS_DIR="$BASE_RESULTS_DIR/$RUN_ID"
RESULTS_DIR_CTR="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/1node_sigkill_inkernel/results/$RUN_ID"
SUMMARY="$RESULTS_DIR/SUMMARY.md"
mkdir -p "$RESULTS_DIR"

branch="$(git -C "$ROOT/nixl" branch --show-current)"
commit="$(git -C "$ROOT/nixl" log -1 --oneline)"
{
  echo "# One-Node In-Kernel SIGKILL Run"
  echo
  echo "- Started UTC: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
  echo "- Job: \`$JOB_ID\`"
  echo "- Node: \`$NODE\`"
  echo "- Branch: \`$branch\`"
  echo "- Commit: \`$commit\`"
  echo "- Spin cycles: \`$SPIN_CYCLES\`"
  echo "- Results: \`$RESULTS_DIR\`"
} > "$SUMMARY"

run_on_node() {
  local label="$1"
  shift
  srun --overlap --jobid="$JOB_ID" -w "$NODE" -N1 -n1 \
    --container-image="$IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace/lishapira \
    "$@"
}

echo "===== topology ====="
run_on_node topology bash -lc 'hostname -f; nvidia-smi -L; nvidia-smi topo -m' \
  > "$RESULTS_DIR/topology.log" 2>&1

echo "===== build ====="
run_on_node build bash -lc '/workspace/lishapira/build_nixl_aarch64.sh' \
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

run_elastic() {
  local label="$1"
  local timeout_seconds="$2"
  local plan_path="$3"
  local log="$4"
  shift 4

  printf '\n===== %s metadata =====\njob=%s\nnode=%s\nplan=%s\ntimeout=%s\n' \
    "$label" "$JOB_ID" "$NODE" "$plan_path" "$timeout_seconds" >> "$log"

  {
    echo "===== ${label} command ====="
    printf 'python3 %q --plan %q --num-processes 4' "$ELASTIC" "$plan_path"
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    echo
    echo "===== ${label} output ====="
    timeout --kill-after=20s "${timeout_seconds}s" \
      srun --overlap --jobid="$JOB_ID" -w "$NODE" -N1 -n1 \
        --container-image="$IMAGE" \
        --container-mounts="$MOUNTS" \
        --container-workdir=/workspace/lishapira \
        bash -lc 'source /workspace/lishapira/setup_node.sh; export PYTHONUNBUFFERED=1; python3 "$@"' \
        bash "$ELASTIC" --plan "$plan_path" --num-processes 4 "$@"
    local rc=$?
    echo "===== ${label} rc=${rc} ====="
    return "$rc"
  } 2>&1 | awk -v label="$label" -v node="$NODE" '{ print strftime("%Y-%m-%dT%H:%M:%S"), "[" label ":" node "]", $0; fflush(); }' >> "$log"
}

cleanup_node() {
  local label="$1"
  local log="$2"
  local cleanup_rc=0
  printf '\n===== %s cleanup =====\n' "$label" >> "$log"
  {
    echo "===== cleanup ${NODE} ====="
    run_on_node cleanup bash -lc '
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
  } 2>&1 | awk -v node="$NODE" '{ print strftime("%Y-%m-%dT%H:%M:%S"), "[" node "]", $0; fflush(); }' >> "$log"
  node_cleanup_rc=$?
  if [[ "$node_cleanup_rc" -ne 0 ]]; then
    cleanup_rc=1
  fi
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
run_elastic initial_baseline 180 "$BASELINE_PLAN" "$initial_log"
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
  run_elastic "fault_${timing}" 180 "$FAULT_PLAN" \
    "$timing_log" \
    --fault-kill-signal sigkill \
    --fault-kill-timing "$timing" \
    --in-kernel-fault-spin-cycles "$SPIN_CYCLES" \
    --fault-evidence-dir "$RESULTS_DIR_CTR"
  fault_rc=$?
  printf '\nFAULT_RC=%s\n' "$fault_rc" >> "$timing_log"

  echo "===== timing ${timing}: cleanup ====="
  cleanup_node "fault_${timing}" "$timing_log"
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
  run_elastic "post_baseline_${timing}" 180 "$BASELINE_PLAN" "$timing_log"
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
cleanup_node final "$final_log"
final_cleanup_rc=$?
echo "===== final baseline ====="
run_elastic final_baseline 180 "$BASELINE_PLAN" "$final_log"
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
if [[ "$final_cleanup_rc" -ne 0 ]]; then
  exit 1
fi
exit "$final_rc"
