#!/bin/bash
# RDMA-only NVLink fault test on the fixed node (default: 4 ranks, victim = rank 2 / GPU 2).
# Every rank is pinned to the same 400G RoCE NIC (mlx5_4), with host networking and
# the rc_gda device path. This matches the proven 4-rank baseline and avoids cross-NIC
# routing. cuda_ipc/NVLink is excluded via --disable-ll-nvlink.
#
#   Step 2: RDMA-only baseline (no fault) -> must prove RDMA (rc_mlx5/rc_gda) + no cuda_ipc, else abort.
#   Step 3/4: DRY RUN (NIXL_EP_DRY_RUN=1) repeats the three-phase plan without a
#             fault; REAL HW injection forces the NVLink link down during phase 1.
#
# The image's OWN elastic.py is patched in-container (patch_fault.py) to stay API-compatible
# with the compiled nixl_ep. After a real injection the victim GPU needs a BMC power cycle.
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${NIXL_EP_IMAGE:-nixl-ep:master}
NUM=4
VICTIM=${NIXL_EP_VICTIM_RANK:-2}     # victim rank == victim physical GPU (CVD masking)
TOKENS=${NIXL_EP_TOKENS:-256}
NIC=${NIXL_EP_NIC:-mlx5_4}           # shared 400G RoCE HCA; proven 4-rank rc_gda baseline
GID_INDEX=${NIXL_EP_GID_INDEX:-1}    # proven RoCE GID index with --network=host
TOOLS=${NIXL_EP_TOOLS_DIR:-/tmp/nixl_ep_rdma_tools}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT=${NIXL_EP_RESULTS_DIR:-/var/tmp/nixl_ep_nvlink/rdma_inject_${NUM}rank_$RUN_ID}
DRY_RUN=${NIXL_EP_DRY_RUN:-0}

[ "$VICTIM" -ge 0 ] && [ "$VICTIM" -lt "$NUM" ] ||
    { echo "ABORT: victim rank must be in [0, $((NUM - 1))]."; exit 1; }
mkdir -p "$TOOLS" "$OUT"

capture_ib() {
    local output=$1
    local counters="/sys/class/infiniband/$NIC/ports/1/counters"
    printf '%s %s\n' \
        "$(cat "$counters/port_xmit_data")" \
        "$(cat "$counters/port_rcv_data")" > "$output"
}

capture_nvlink() {
    local output=$1
    : > "$output"
    for i in $(seq 0 7); do
        nvidia-smi nvlink -gt d -i "$i" >> "$output" 2>&1 || true
    done
}

echo "=== 0. build injector + stage three-phase plan (NUM=$NUM victim=GPU$VICTIM dry_run=$DRY_RUN) ==="
gcc -O1 -Wall -o "$TOOLS/nvlink_hwinject" \
    "$SCRIPT_DIR/nvlink_hwinject.c" || exit 1
cp "$SCRIPT_DIR/patch_fault.py" "$TOOLS/patch_fault.py"
cp "$SCRIPT_DIR/three_phase_all_active_4rank.json" "$TOOLS/plan_3phase.json"
echo "plan: $(tr -d '[:space:]' < "$TOOLS/plan_3phase.json")"

echo "=== 1. preflight health ==="
healthy=1
for i in $(seq 0 7); do
    links=$(nvidia-smi nvlink -s -i "$i" 2>/dev/null | awk '/GB\/s/{c++} END{print c+0}')
    rec=$(nvidia-smi -q -i "$i" 2>/dev/null | awk -F: 'tolower($1)~/recovery action/{gsub(/^ +| +$/,"",$2);print $2;exit}')
    printf 'GPU%d links=%s recovery=%s\n' "$i" "$links" "${rec:-N/A}" | tee -a "$OUT/health_before.log"
    [ "$links" = "18" ] && [ "$rec" = "None" ] || healthy=0
done
[ "$healthy" = "1" ] || { echo "ABORT: all GPUs must have 18 links and Recovery None."; exit 2; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "ABORT: a GPU workload is active."; exit 3; }
nic_state=$(ibv_devinfo -d "$NIC" 2>/dev/null | awk '/state:/{print $2; exit}')
[ "$nic_state" = "PORT_ACTIVE" ] || { echo "ABORT: $NIC must be PORT_ACTIVE (current: ${nic_state:-unknown})."; exit 4; }

echo "NIC pin: every rank -> $NIC"

GDA=${NIXL_EP_GDA:-1}   # proven baseline uses cuda0-mlx5_4 and rc_gda
NVLINK_DEFAULT=${NIXL_EP_NVLINK_DEFAULT:-0}   # 1 => DEFAULT NVLink/cuda_ipc path (no RDMA forcing) for comparison
if [ "$NVLINK_DEFAULT" = "1" ]; then
    DISABLE_LL=""                # keep NVLink low-latency kernels -> cuda_ipc over NVLink
    echo "PATH MODE: DEFAULT NVLink/cuda_ipc (no --disable-ll-nvlink; NIC-only UCX backend pin, no GDA) -- comparison run"
else
    DISABLE_LL="--disable-ll-nvlink"
    echo "PATH MODE: RDMA-only (--disable-ll-nvlink, single-NIC $NIC, GDA=$GDA, GID=$GID_INDEX, host network)"
fi

DOCKER_COMMON=(--rm --network=host --gpus all --ipc=host --device /dev/infiniband
    --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864
    -e "NIXL_EP_NIC=$NIC" -e "NIXL_EP_GDA=$GDA" -e "NIXL_EP_NVLINK_DEFAULT=$NVLINK_DEFAULT"
    -e "UCX_IB_GID_INDEX=$GID_INDEX"
    -e UCX_IB_GDA_MAX_HCA_PER_GPU=16 -e UCX_IB_GDA_RETAIN_INACTIVE_CTX=yes
    -e UCX_LOG_LEVEL=info -v "$TOOLS":/tools -w /workspace/nixl)

run_elastic() {  # $1=plan file in /tools   (fault env + SYS_ADMIN come from caller via extra args array)
    local plan="$1"; shift
    sudo docker run "${DOCKER_COMMON[@]}" "$@" "$IMAGE" bash -lc '
        unset UCX_TLS UCX_NET_DEVICES NIXL_ETCD_ENDPOINTS NIXL_ETCD_PEER_URLS NIXL_ETCD_NAMESPACE
        python3 /tools/patch_fault.py || exit 90
        python3 -u examples/device/ep/tests/elastic/elastic.py \
            --plan '"/tools/$plan"' --num-processes '"$NUM"' \
            --num-experts-per-rank 32 --num-topk 8 --num-tokens '"$TOKENS"' \
            --timeout-ms 10000 '"$DISABLE_LL"'
    '
}

if [ "$NVLINK_DEFAULT" = "1" ]; then
    echo "=== 2. NVLink/cuda_ipc baseline (no fault) -> expect cuda_ipc present, all ranks done ==="
    capture_ib "$OUT/baseline_ib_before.txt"
    capture_nvlink "$OUT/baseline_nvlink_before.log"
    run_elastic plan_3phase.json 2>&1 | tee "$OUT/baseline.log"
    baseline_rc=${PIPESTATUS[0]}
    capture_ib "$OUT/baseline_ib_after.txt"
    capture_nvlink "$OUT/baseline_nvlink_after.log"
    done_cnt=$(grep -c ' -> done' "$OUT/baseline.log")
    ipc_hit=$(grep -c 'cuda_ipc/' "$OUT/baseline.log")
    read -r ib_tx_before ib_rx_before < "$OUT/baseline_ib_before.txt"
    read -r ib_tx_after ib_rx_after < "$OUT/baseline_ib_after.txt"
    ib_delta=$((ib_tx_after - ib_tx_before + ib_rx_after - ib_rx_before))
    nvl_before=$(awk '/Data (Tx|Rx):/{sum+=$5} END{print sum+0}' "$OUT/baseline_nvlink_before.log")
    nvl_after=$(awk '/Data (Tx|Rx):/{sum+=$5} END{print sum+0}' "$OUT/baseline_nvlink_after.log")
    nvl_delta=$((nvl_after - nvl_before))
    echo "baseline: done=$done_cnt/$NUM cuda_ipc_usage=$ipc_hit nic_counter_delta=$ib_delta nvlink_delta_kib=$nvl_delta"
    if [ "$baseline_rc" -ne 0 ] || [ "$done_cnt" -lt "$NUM" ] ||
       [ "$ipc_hit" -lt 1 ] || [ "$nvl_delta" -le 0 ]; then
        echo "ABORT: NVLink baseline did not prove cuda_ipc/NVLink traffic with all $NUM ranks complete. See $OUT/baseline.log"
        exit 8
    fi
    echo "baseline OK: NVLink/cuda_ipc baseline completed on all $NUM ranks."
else
    echo "=== 2. RDMA-only baseline (no fault) -> prove RDMA (rc_mlx5/rc_gda), no cuda_ipc ==="
    capture_ib "$OUT/baseline_ib_before.txt"
    capture_nvlink "$OUT/baseline_nvlink_before.log"
    run_elastic plan_3phase.json 2>&1 | tee "$OUT/baseline.log"
    baseline_rc=${PIPESTATUS[0]}
    capture_ib "$OUT/baseline_ib_after.txt"
    capture_nvlink "$OUT/baseline_nvlink_after.log"
    done_cnt=$(grep -c ' -> done' "$OUT/baseline.log")
    gda_cnt=$(grep -c 'device(rc_gda/' "$OUT/baseline.log")       # GPU-initiated RDMA selected
    ipc_hit=$(grep -c 'cuda_ipc/' "$OUT/baseline.log")            # "^cuda_ipc" exclusion has no slash
    read -r ib_tx_before ib_rx_before < "$OUT/baseline_ib_before.txt"
    read -r ib_tx_after ib_rx_after < "$OUT/baseline_ib_after.txt"
    ib_delta=$((ib_tx_after - ib_tx_before + ib_rx_after - ib_rx_before))
    nvl_before=$(awk '/Data (Tx|Rx):/{sum+=$5} END{print sum+0}' "$OUT/baseline_nvlink_before.log")
    nvl_after=$(awk '/Data (Tx|Rx):/{sum+=$5} END{print sum+0}' "$OUT/baseline_nvlink_after.log")
    nvl_delta=$((nvl_after - nvl_before))
    echo "baseline: done=$done_cnt/$NUM rc_gda_lines=$gda_cnt cuda_ipc_usage=$ipc_hit nic_counter_delta=$ib_delta nvlink_delta_kib=$nvl_delta"
    if [ "$baseline_rc" -ne 0 ] || [ "$done_cnt" -lt "$NUM" ] ||
       [ "$gda_cnt" -lt 1 ] || [ "$ipc_hit" -ne 0 ] ||
       [ "$ib_delta" -le 0 ] || [ "$nvl_delta" -ne 0 ]; then
        echo "ABORT: RDMA-only baseline did not prove NIC traffic with zero cuda_ipc/NVLink traffic. See $OUT/baseline.log"
        exit 8
    fi
    echo "baseline OK: RDMA NIC traffic established on all $NUM ranks; cuda_ipc and NVLink traffic absent."
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "=== 3. DRY RUN: repeat three-phase plan, NO hardware injection ==="
    run_elastic plan_3phase.json 2>&1 | tee "$OUT/fault.log"
    fault_rc=${PIPESTATUS[0]}
else
    echo "=== 3. non-destructive injector probe on GPU$VICTIM ==="
    sudo "$TOOLS/nvlink_hwinject" "$VICTIM" 2>&1 | tee "$OUT/injector_probe.log"
    [ "${PIPESTATUS[0]}" = "0" ] || { echo "ABORT: injector probe failed."; exit 10; }
    echo "=== 4. REAL HW injection: force NVLink link 0 down on GPU$VICTIM (rank $VICTIM victim) ==="
    sudo dmesg -T > "$OUT/.dmesg_before.tmp" 2>/dev/null || true
    capture_ib "$OUT/fault_ib_before.txt"
    capture_nvlink "$OUT/fault_nvlink_before.log"
    run_elastic plan_3phase.json \
        --cap-add SYS_ADMIN \
        -e NIXL_EP_FAULT_NVLINK=1 -e "NIXL_EP_FAULT_RANK=$VICTIM" -e NIXL_EP_FAULT_PHASE=1 \
        -e NIXL_EP_FAULT_LINK=0 -e NIXL_EP_FAULT_TOOL=/tools/nvlink_hwinject \
        2>&1 | tee "$OUT/fault.log"
    fault_rc=${PIPESTATUS[0]}
    capture_ib "$OUT/fault_ib_after.txt"
    capture_nvlink "$OUT/fault_nvlink_after.log"
    sudo dmesg -T > "$OUT/.dmesg_after.tmp" 2>/dev/null || true
    python3 - "$OUT/.dmesg_before.tmp" "$OUT/.dmesg_after.tmp" \
        "$OUT/dmesg_fault_delta.log" <<'PY'
import sys
from collections import Counter

before_path, after_path, output_path = sys.argv[1:]
before = open(before_path, errors="replace").readlines()
after = open(after_path, errors="replace").readlines()
if after[:len(before)] == before:
    delta = after[len(before):]
else:
    remaining = Counter(before)
    delta = []
    for line in after:
        if remaining[line]:
            remaining[line] -= 1
        else:
            delta.append(line)
open(output_path, "w").writelines(delta)
PY
    rm -f "$OUT/.dmesg_before.tmp" "$OUT/.dmesg_after.tmp"
fi
for i in $(seq 0 7); do
    links=$(nvidia-smi nvlink -s -i "$i" 2>/dev/null | awk '/GB\/s/{c++} END{print c+0}')
    rec=$(nvidia-smi -q -i "$i" 2>/dev/null | awk -F: 'tolower($1)~/recovery action/{gsub(/^ +| +$/,"",$2);print $2;exit}')
    printf 'GPU%d links=%s recovery=%s\n' "$i" "$links" "${rec:-N/A}" >> "$OUT/health_after.log"
done

echo "=== 5. verdict ==="
python3 - "$OUT/fault.log" "$NUM" "$VICTIM" "$NVLINK_DEFAULT" "$DRY_RUN" <<'PY'
import re, sys
log = open(sys.argv[1], errors="replace").read()
n, v = int(sys.argv[2]), int(sys.argv[3])
nvlink_default, dry_run = sys.argv[4] == "1", sys.argv[5] == "1"
path = "NVLink/cuda_ipc" if nvlink_default else "RDMA-only (rc_gda)"
kill = "no fault" if dry_run else "REAL NVLink HW error"
survivors = set(range(n)) - {v}
all_ranks = set(range(n))
final_phase = 2
done = {int(r) for r in re.findall(r'global_rank=(\d+), local_rank=\d+ -> done', log)}
final_started = {int(r) for r in re.findall(rf'global_rank=(\d+), local_rank=\d+ -> start phase {final_phase}', log)}
final_ended = {int(r) for r in re.findall(rf'global_rank=(\d+), local_rank=\d+ -> end phase {final_phase}', log)}
exits = {int(w): int(c) for w, c in re.findall(r'worker (\d+)\s*\(exit code (-?\d+)\)', log)}
inj = 'forcing NVLink link 0 down' in log
ima = log.count('illegal memory access')
print(f'path={path}   trigger={kill}')
print(f'injection_fired={inj}   illegal_mem_access_lines={ima}')
print(f'survivors={sorted(survivors)} victim={v}')
print(f'final_phase={final_phase} started={sorted(final_started)}  ended={sorted(final_ended)}  done_ranks={sorted(done)}')
print(f'failed_worker_exits={exits}')
print()
if all_ranks.issubset(done) and all_ranks.issubset(final_ended):
    print(f'RESULT: ALL RANKS CONTINUED - every rank, including GPU{v}, finished after the {path} fault trigger.')
    verdict_ok = True
elif survivors.issubset(done) and survivors.issubset(final_ended):
    print(f'RESULT: VICTIM FAILED, SURVIVORS LIVED - GPU{v} did not finish, but every other rank completed post-fault phase 2.')
    verdict_ok = True
else:
    missing = sorted(survivors - (done & final_ended))
    entered = survivors & final_started
    print(f'RESULT: SURVIVORS DID NOT ALL COMPLETE on the {path} path - missing {missing}.')
    if entered and ima:
        print(f'        Survivors {sorted(entered)} entered phase 2, then hit a CUDA illegal memory access ({ima} lines).')
    else:
        print('        => inspect fault.log for the first CUDA or worker failure.')
    verdict_ok = False
sys.exit(0 if verdict_ok else 1)
PY
verdict_rc=$?

echo
echo "Results: $OUT"
if [ "$DRY_RUN" = "1" ]; then
    echo "MODE: THREE-PHASE DRY RUN (no fault, no HW injection)."
    echo "If all expected ranks completed and injection_fired=False, rerun with NIXL_EP_DRY_RUN=0 for real HW."
else
    echo "MODE: REAL HW INJECTION. GPU$VICTIM needs a BMC power cycle before the next run."
fi
if [ "$fault_rc" -ne 0 ] || [ "$verdict_rc" -ne 0 ]; then
    exit 11
fi
