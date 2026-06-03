#!/usr/bin/env bash
set -u -o pipefail

RUN_DIR="${RUN_DIR:-/lustre/fsw/network_research_advdev/lishapira/nixl/nixl_ep_nvlink_understanding/old_2nodes_sigkill_non_inkernel}"
FAULT_PLAN="/workspace/lishapira/nixl/examples/device/ep/tests/elastic/expansion_fault_contraction_kill_2.json"
BASELINE_PLAN="/workspace/lishapira/nixl/nixl_ep_nvlink_understanding/old_2nodes_sigkill_non_inkernel/plans/expansion_contraction_no_fault_rank2.json"

timings=(
  before-dispatch
  after-dispatch
  between-dispatch-combine
  dispatch-between-send-receive
  before-combine
  combine-between-send-receive
  after-combine
)

for timing in "${timings[@]}"; do
  echo "===== timing ${timing}: fault run ====="
  "$RUN_DIR/run_elastic_pair.sh" "fault_${timing}" 180 "$FAULT_PLAN" \
    --fault-kill-signal sigkill \
    --fault-kill-timing "$timing"
  fault_rc=$?
  echo "fault_${timing}_rc=${fault_rc}"

  echo "===== timing ${timing}: cleanup ====="
  "$RUN_DIR/cleanup_nodes.sh" "fault_${timing}"

  echo "===== timing ${timing}: post baseline ====="
  "$RUN_DIR/run_elastic_pair.sh" "post_baseline_${timing}" 240 "$BASELINE_PLAN"
  baseline_rc=$?
  echo "post_baseline_${timing}_rc=${baseline_rc}"

  if [[ "$baseline_rc" -ne 0 ]]; then
    echo "post baseline failed after ${timing}; stopping sweep"
    exit "$baseline_rc"
  fi
done

echo "===== final cleanup ====="
"$RUN_DIR/cleanup_nodes.sh" final
echo "===== final baseline ====="
"$RUN_DIR/run_elastic_pair.sh" final_baseline 240 "$BASELINE_PLAN"
