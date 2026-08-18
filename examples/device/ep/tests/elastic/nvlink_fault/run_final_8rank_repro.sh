#!/bin/bash
# Reproduce an 8-rank NVLink FORCE_LINK_DOWN failure and archive application
# and kernel evidence that attributes the resulting CUDA errors by PID.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ELASTIC_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TOOLS_DIR=${NIXL_EP_NVLINK_TOOLS_DIR:-/tmp/nixl_ep_nvlink_tools}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${NIXL_EP_NVLINK_RESULTS_DIR:-/var/tmp/nixl_ep_nvlink/final_8rank_$RUN_ID}
IMAGE=${NIXL_EP_IMAGE:-nixl-ep:local}

mkdir -p "$TOOLS_DIR" "$OUT_DIR"
gcc -O1 -Wall -o "$TOOLS_DIR/nvlink_hwinject" \
    "$SCRIPT_DIR/nvlink_hwinject.c" || exit 1

echo "=== pre-run health ===" | tee "$OUT_DIR/health_before.log"
healthy=1
for i in 0 1 2 3 4 5 6 7; do
    links=$(nvidia-smi nvlink -s -i "$i" 2>/dev/null |
        awk '/53[.]125 GB[/]s/ {count++} END {print count + 0}')
    recovery=$(nvidia-smi -q -i "$i" 2>/dev/null |
        awk -F: 'tolower($1) ~ /recovery action/ {
            gsub(/^ +| +$/, "", $2); print $2; exit
        }')
    printf 'GPU%d links=%s recovery=%s\n' "$i" "$links" "${recovery:-N/A}" |
        tee -a "$OUT_DIR/health_before.log"
    if [ "$links" != "18" ] || [ "$recovery" != "None" ]; then
        healthy=0
    fi
done

if [ "$healthy" != "1" ]; then
    echo "ABORT: all GPUs must have 18 links and Recovery Action None."
    exit 2
fi

compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$compute_pids" ]; then
    echo "ABORT: another GPU workload is active."
    nvidia-smi --query-compute-apps=pid,gpu_bus_id,process_name \
        --format=csv,noheader
    exit 3
fi

echo "=== non-destructive injector probe ===" |
    tee "$OUT_DIR/injector_probe.log"
sudo "$TOOLS_DIR/nvlink_hwinject" 2 2>&1 |
    tee -a "$OUT_DIR/injector_probe.log"
probe_rc=${PIPESTATUS[0]}
if [ "$probe_rc" != "0" ]; then
    echo "ABORT: injector probe failed."
    exit 4
fi

nvidia-smi --query-gpu=index,pci.bus_id,uuid --format=csv,noheader \
    > "$OUT_DIR/gpu_map.csv"
(
    while true; do
        printf '%s\n' "--- uptime=$(awk '{print $1}' /proc/uptime)"
        nvidia-smi \
            --query-compute-apps=pid,gpu_bus_id,used_memory,process_name \
            --format=csv,noheader
        sleep 0.2
    done
) > "$OUT_DIR/pidmap.log" 2>&1 &
monitor_pid=$!

echo "=== starting 8-rank run ===" | tee "$OUT_DIR/run_meta.log"
date -Ins | tee -a "$OUT_DIR/run_meta.log"
awk '{print "uptime=" $1}' /proc/uptime | tee -a "$OUT_DIR/run_meta.log"

sudo docker run --rm --gpus all --ipc=host --cap-add=SYS_ADMIN \
    -v "$ELASTIC_DIR":/workspace/nixl/examples/device/ep/tests/elastic \
    -v "$TOOLS_DIR":/tools \
    -w /workspace/nixl "$IMAGE" \
    python3 -u examples/device/ep/tests/elastic/elastic.py \
        --plan examples/device/ep/tests/elastic/nvlink_fault_rank2.json \
        --num-processes 8 \
        --num-tokens 8192 \
        --fault-nvlink \
        --fault-nvlink-link 0 \
        --fault-nvlink-tool /tools/nvlink_hwinject \
    > "$OUT_DIR/elastic.log" 2>&1
run_rc=$?

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

printf 'docker_exit=%s\n' "$run_rc" | tee -a "$OUT_DIR/run_meta.log"
date -Ins | tee -a "$OUT_DIR/run_meta.log"
sudo dmesg -T > "$OUT_DIR/dmesg.log"

echo "=== post-run health ===" | tee "$OUT_DIR/health_after.log"
for i in 0 1 2 3 4 5 6 7; do
    links=$(nvidia-smi nvlink -s -i "$i" 2>/dev/null |
        awk '/53[.]125 GB[/]s/ {count++} END {print count + 0}')
    recovery=$(nvidia-smi -q -i "$i" 2>/dev/null |
        awk -F: 'tolower($1) ~ /recovery action/ {
            gsub(/^ +| +$/, "", $2); print $2; exit
        }')
    printf 'GPU%d links=%s recovery=%s\n' "$i" "$links" "${recovery:-N/A}" |
        tee -a "$OUT_DIR/health_after.log"
done

python3 - "$OUT_DIR/elastic.log" > "$OUT_DIR/error_summary.txt" <<'PY'
import re
import sys
from collections import Counter, defaultdict

by_pid = defaultdict(Counter)
worker_exit = None
with open(sys.argv[1], errors="replace") as stream:
    for line in stream:
        if "Worker processes failed:" in line:
            worker_exit = line.rstrip()
        match = re.search(
            r"\[[0-9a-f]+:(\d+)\s+:\d+\].*"
            r"(uncorrectable NVLink error detected|unspecified launch failure)",
            line,
        )
        if match:
            by_pid[int(match.group(1))][match.group(2)] += 1

print(f"distinct_error_pids={len(by_pid)}")
for pid in sorted(by_pid):
    details = ", ".join(
        f"{error}={count}" for error, count in sorted(by_pid[pid].items())
    )
    print(f"pid={pid}: {details}")
all_uncorrectable = len(by_pid) == 8 and all(
    "uncorrectable NVLink error detected" in errors for errors in by_pid.values()
)
print(f"all_8_pids_uncorrectable={str(all_uncorrectable).lower()}")
print(f"worker_exit_line={worker_exit}")
PY

echo "=== error summary ==="
while IFS= read -r line; do
    printf '%s\n' "$line"
done < "$OUT_DIR/error_summary.txt"
echo
echo "Results saved to: $OUT_DIR"
echo "GPU 2 requires a BMC power cycle after a successful injection."

# A successful experiment is expected to return nonzero because all workers fail.
exit "$run_rc"
