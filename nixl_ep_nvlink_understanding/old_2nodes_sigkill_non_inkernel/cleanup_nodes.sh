#!/usr/bin/env bash
# Cleanup/check helper for NIXL EP SIGKILL fault runs.
# It scans both allocated nodes for leftover elastic/rank-server processes,
# force-kills any leftovers, then records nvidia-smi compute apps and GPU memory.
# Empty "compute apps" sections plus passing post-fault baselines mean SIGKILL
# did not leave stuck GPU compute processes in this allocation.
set -u -o pipefail

LABEL="${1:-cleanup}"
RUN_DIR="${RUN_DIR:-/lustre/fsw/network_research_advdev/lishapira/nixl/nixl_ep_nvlink_understanding/old_2nodes_sigkill_non_inkernel}"
RESULTS_DIR="${RESULTS_DIR:-$RUN_DIR/results}"
JOB_ID="${JOB_ID:-1857351}"
MASTER_NODE="${MASTER_NODE:-theia0153}"
WORKER_NODE="${WORKER_NODE:-theia0156}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/${LABEL}_cleanup.log"

rm -f "$LOG"

for node in "$MASTER_NODE" "$WORKER_NODE"; do
  {
    echo "===== cleanup ${node} ====="
    srun --overlap --jobid="$JOB_ID" -w "$node" -N1 -n1 bash -lc '
      echo "node=$(hostname -f)"
      echo
      echo "COMMAND: pgrep -af \"[e]lastic.py|[r]ank_server|[s]pawn_main\" || true"
      echo "EXPECTED PASS RESULT: no output. Any listed process is a leftover test process."
      echo "before process scan:"
      pgrep -af "[e]lastic.py|[r]ank_server|[s]pawn_main" || true

      echo
      echo "COMMAND: pkill -9 -f \"[e]lastic.py\"; pkill -9 -f \"[r]ank_server\"; pkill -9 -f \"[s]pawn_main\""
      echo "EXPECTED PASS RESULT: no output. This force-cleans leftover harness/test processes if any exist."
      for pat in "[e]lastic.py" "[r]ank_server" "[s]pawn_main"; do
        pkill -9 -f "$pat" || true
      done
      sleep 2

      echo
      echo "COMMAND: pgrep -af \"[e]lastic.py|[r]ank_server|[s]pawn_main\" || true"
      echo "EXPECTED PASS RESULT: no output after cleanup."
      echo "after process scan:"
      pgrep -af "[e]lastic.py|[r]ank_server|[s]pawn_main" || true

      echo
      echo "COMMAND: nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits"
      echo "EXPECTED PASS RESULT: no output. Any listed row means a GPU compute process is still attached."
      echo "compute apps:"
      nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits || true

      echo
      echo "COMMAND: nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits"
      echo "EXPECTED PASS RESULT: one row per GPU with only low idle memory, not large/stuck test allocations."
      echo "gpu memory:"
      nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true
    '
  } >> "$LOG" 2>&1
done
