#!/usr/bin/env bash
# One-node GB200/GB300 smoke for NIXL EP in-kernel SIGKILL timing.
# Proves the marker/helper can send SIGKILL after GPU entered, before GPU exited,
# for dispatch/combine send/receive. Build must include commit 4087c88 or later.
# Outputs:
#   in_kernel_fault_*.log: entered/exited timing evidence.
#   fault_*.log/.rc: fault run logs and return code.
#   post_baseline_*.log/.rc, final_baseline.*: recovery baselines.
#   *_cleanup.log: leftover process/GPU cleanup checks.
# Pass: HIT_IN_KERNEL_WINDOW, exited_before_sigkill=0, no missed/timeout/errors,
# healthy ranks detect {2}, continue bandwidth, non-killed ranks done, baselines rc=0.
set -u -o pipefail

ROOT="/lustre/fsw/network_research_advdev/lishapira"
RESULTS_DIR="$ROOT/nixl/nixl_ep_nvlink_understanding/1node_sigkill_inkernel/results"
RESULTS_DIR_CTR="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/1node_sigkill_inkernel/results"
IMAGE="$ROOT/nixl-hybrid-ep-cuda2.sqsh"
MOUNTS="$ROOT:/workspace/lishapira"
JOB_ID="${JOB_ID:-1995953}"
NODE="${NODE:-lyris0145}"
ELASTIC="/workspace/lishapira/nixl/examples/device/ep/tests/elastic/elastic.py"
FAULT_PLAN="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/1node_sigkill_inkernel/plans/one_node_fault_rank2.json"
BASELINE_PLAN="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/1node_sigkill_inkernel/plans/one_node_baseline.json"
SPIN_CYCLES="${SPIN_CYCLES:-100000000}"

mkdir -p "$RESULTS_DIR"

{
  echo "scope=1node-gb200-smoke"
  echo "job_id=$JOB_ID"
  echo "node=$NODE"
  echo "spin_cycles=$SPIN_CYCLES"
  echo "results_dir=$RESULTS_DIR"
  git -C "$ROOT/nixl" branch --show-current
  git -C "$ROOT/nixl" log -3 --oneline
  git -C "$ROOT/nixl" status --short
} > "$RESULTS_DIR/run_metadata.log" 2>&1

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
  > "$RESULTS_DIR/topology_${NODE}.log" 2>&1

echo "===== build ====="
if [[ "${SKIP_BUILD:-0}" == "1" && -f "$RESULTS_DIR/build.rc" ]] && [[ "$(sed -n 's/^build_rc=//p' "$RESULTS_DIR/build.rc")" == "0" ]]; then
  echo "Reusing previous successful build" >> "$RESULTS_DIR/build.log"
  build_rc=0
else
  run_on_node build bash -lc '/workspace/lishapira/build_nixl_aarch64.sh' \
    > "$RESULTS_DIR/build.log" 2>&1
  build_rc=$?
fi
echo "build_rc=$build_rc" > "$RESULTS_DIR/build.rc"
if [[ "$build_rc" -ne 0 ]]; then
  echo "build failed; see $RESULTS_DIR/build.log" >&2
  exit "$build_rc"
fi

run_elastic() {
  local label="$1"
  local timeout_seconds="$2"
  local plan="$3"
  shift 3
  local log="$RESULTS_DIR/${label}.log"
  local rc_log="$RESULTS_DIR/${label}.rc"
  rm -f "$log" "$rc_log"
  {
    printf 'label=%s\nscope=1node-gb200-smoke\njob=%s\nnode=%s\nplan=%s\ntimeout=%s\n' \
      "$label" "$JOB_ID" "$NODE" "$plan" "$timeout_seconds"
    printf 'extra_args='
    printf ' %q' "$@"
    printf '\n'
  } > "$rc_log"

  timeout --kill-after=20s "${timeout_seconds}s" \
    srun --overlap --jobid="$JOB_ID" -w "$NODE" -N1 -n1 \
      --container-image="$IMAGE" \
      --container-mounts="$MOUNTS" \
      --container-workdir=/workspace/lishapira \
      bash -lc 'source /workspace/lishapira/setup_node.sh; export PYTHONUNBUFFERED=1; python3 "$@"' \
      bash "$ELASTIC" --plan "$plan" --num-processes 4 "$@" \
      > "$log" 2>&1
  local rc=$?
  echo "rc=$rc" >> "$rc_log"
  return "$rc"
}

cleanup_node() {
  local label="$1"
  local log="$RESULTS_DIR/${label}_cleanup.log"
  rm -f "$log"
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
  ' > "$log" 2>&1
}

timings=(
  dispatch-send-during-kernel
  dispatch-receive-during-kernel
  combine-send-during-kernel
  combine-receive-during-kernel
)

echo "===== initial baseline ====="
run_elastic initial_baseline 180 "$BASELINE_PLAN"
initial_rc=$?
echo "initial_baseline_rc=$initial_rc" > "$RESULTS_DIR/initial_baseline.summary"
if [[ "$initial_rc" -ne 0 ]]; then
  echo "initial baseline failed; stopping" >&2
  exit "$initial_rc"
fi

for timing in "${timings[@]}"; do
  echo "===== fault ${timing} ====="
  rm -f "$RESULTS_DIR"/in_kernel_fault_*_"$timing"_*.log
  run_elastic "fault_${timing}" 180 "$FAULT_PLAN" \
    --fault-kill-signal sigkill \
    --fault-kill-timing "$timing" \
    --in-kernel-fault-spin-cycles "$SPIN_CYCLES" \
    --fault-evidence-dir "$RESULTS_DIR_CTR"
  fault_rc=$?
  echo "fault_${timing}_rc=$fault_rc" > "$RESULTS_DIR/fault_${timing}.summary"

  cleanup_node "fault_${timing}"

  hit_files=("$RESULTS_DIR"/in_kernel_fault_*_"$timing"_*.log)
  hit_found=0
  miss_found=0
  for evidence_file in "${hit_files[@]}"; do
    [[ -e "$evidence_file" ]] || continue
    evidence_text="$(<"$evidence_file")"
    [[ "$evidence_text" == *"HIT_IN_KERNEL_WINDOW"* ]] && hit_found=1
    if [[ "$evidence_text" == *"MISSED_IN_KERNEL_TIMING"* || "$evidence_text" == *"IN_KERNEL_MARKER_TIMEOUT"* ]]; then
      miss_found=1
    fi
  done
  if [[ "$hit_found" -ne 1 ]]; then
    echo "missing HIT_IN_KERNEL_WINDOW evidence for ${timing}" >&2
    exit 1
  fi
  if [[ "$miss_found" -ne 0 ]]; then
    echo "missed in-kernel timing evidence found for ${timing}" >&2
    exit 1
  fi

  echo "===== post baseline ${timing} ====="
  run_elastic "post_baseline_${timing}" 180 "$BASELINE_PLAN"
  baseline_rc=$?
  echo "post_baseline_${timing}_rc=$baseline_rc" > "$RESULTS_DIR/post_baseline_${timing}.summary"
  if [[ "$baseline_rc" -ne 0 ]]; then
    echo "post baseline failed after ${timing}" >&2
    exit "$baseline_rc"
  fi
done

cleanup_node final
run_elastic final_baseline 180 "$BASELINE_PLAN"
final_rc=$?
echo "final_baseline_rc=$final_rc" > "$RESULTS_DIR/final_baseline.summary"
exit "$final_rc"
