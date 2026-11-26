#!/usr/bin/env bash
# Single-node NIXL EP run with an NVLink fault, using the same elastic.py patches
# (patch_rdma.py + patch_fault_2node.py) as run_2node.sh. All ranks are on one
# node, so every rank-to-rank transfer goes over NVLink.
#
# Defaults: adv-dev-dgxb-200, GPUs 2,3,5,6 on rails 3,4,6,7, victim rank 0 (GPU 2),
# so the victim is the same physical GPU as in the two-node runs and the other
# ranks are its neighbours.
#
# NIXL_EP_FAULT_TIMING (see patch_fault_2node.py):
#   before_send  the neighbours launch their send part after the link is down
#   after_send   the link goes down once every neighbour's send part has finished
#   during_send  the neighbours launch their send part NIXL_EP_FAULT_SEND_OFFSET_MS
#                after the victim starts the injector
#   timer, nvlink, rdma are also accepted
# NIXL_EP_FAULT_DRY_RUN=1 runs the injector's probe, which touches no link.
# A real fault leaves the victim GPU needing a power cycle.
#
#   RUN_ID=<unique-id> NIXL_EP_FAULT_TIMING=after_send bash run_1node.sh
set -euo pipefail

: "${RUN_ID:?set RUN_ID to a unique value}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${NIXL_EP_IMAGE:-nixl-ep:master}
HOST=${NIXL_EP_HOST:-adv-dev-dgxb-200}
RESULTS_ROOT=${NIXL_EP_RESULTS_ROOT:-/swgwork/lishapira/nixl_ep_b200_1node}
OUT="$RESULTS_ROOT/$RUN_ID"
INJECTOR=${NIXL_EP_INJECTOR:-$RESULTS_ROOT/nvlink_hwinject}
TOKENS=${NIXL_EP_TOKENS:-8192}
EXPERTS=${NIXL_EP_EXPERTS:-2}
TOPK=${NIXL_EP_TOPK:-8}
TIMEOUT_MS=${NIXL_EP_TIMEOUT_MS:-30000}
GID_INDEX=${NIXL_EP_GID_INDEX:-3}
FAULT=nvlink
FAULT_RANKS=${NIXL_EP_FAULT_RANKS:-0}
FAULT_PHASE=${NIXL_EP_FAULT_PHASE:-1}
FAULT_LINK=${NIXL_EP_FAULT_LINK:-0}
FAULT_TIMING=${NIXL_EP_FAULT_TIMING:-after_send}
FAULT_DRY_RUN=${NIXL_EP_FAULT_DRY_RUN:-0}
FAULT_SEND_OFFSET_MS=${NIXL_EP_FAULT_SEND_OFFSET_MS:-10}

ALL_RAIL_HCAS=(mlx5_4 mlx5_7 mlx5_8 mlx5_9 mlx5_10 mlx5_13 mlx5_14 mlx5_15)
ALL_RAIL_NETDEVS=(enp24s0np0 enp64s0np0 enp79s0np0 enp94s0np0 enp154s0np0 enp192s0np0 enp206s0np0 enp220s0np0)
RAILS_CSV=${NIXL_EP_RAILS:-3,4,6,7}
GPUS_CSV=${NIXL_EP_GPUS:-2,3,5,6}

fail() { echo "ABORT: $*" >&2; exit 1; }
IFS=, read -r -a ACTIVE_RAILS <<<"$RAILS_CSV"
IFS=, read -r -a ACTIVE_GPUS <<<"$GPUS_CSV"
PROCESSES=${#ACTIVE_RAILS[@]}
((PROCESSES == 4 && ${#ACTIVE_GPUS[@]} == 4)) || fail "NIXL_EP_RAILS and NIXL_EP_GPUS must list 4 entries each"
ACTIVE_HCAS=()
ACTIVE_NETDEVS=()
for index in "${!ACTIVE_RAILS[@]}"; do
    rail=${ACTIVE_RAILS[$index]}
    gpu=${ACTIVE_GPUS[$index]}
    [[ "$rail" =~ ^[1-8]$ && "$gpu" =~ ^[0-7]$ && "$gpu" -eq "$((rail - 1))" ]] ||
        fail "rail $rail is not adjacent to GPU $gpu"
    ACTIVE_HCAS+=("${ALL_RAIL_HCAS[$((rail - 1))]}")
    ACTIVE_NETDEVS+=("${ALL_RAIL_NETDEVS[$((rail - 1))]}")
done
NIC_MAP=$(IFS=,; echo "${ACTIVE_HCAS[*]}")
PLAN_FILE=three_phase_all_active_4rank.json

[[ "$FAULT_RANKS" =~ ^[0-3]$ ]] || fail "NIXL_EP_FAULT_RANKS must be one rank, 0-3"
[[ "$FAULT_PHASE" =~ ^[12]$ ]] || fail "NIXL_EP_FAULT_PHASE must be 1 or 2"
[[ "$FAULT_LINK" =~ ^[0-9]+$ ]] && ((FAULT_LINK < 18)) || fail "NIXL_EP_FAULT_LINK must be 0-17"
[[ "$FAULT_TIMING" =~ ^(timer|nvlink|rdma|before_send|after_send|during_send|after_victim_send|after_recv)$ ]] ||
    fail "NIXL_EP_FAULT_TIMING must be timer, nvlink, rdma, before_send, after_send, during_send, after_victim_send or after_recv"
[[ "$FAULT_SEND_OFFSET_MS" =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "NIXL_EP_FAULT_SEND_OFFSET_MS must be a number"
[[ "$FAULT_DRY_RUN" =~ ^[01]$ ]] || fail "NIXL_EP_FAULT_DRY_RUN must be 0 or 1"
NEIGHBOURS=$(seq 0 "$((PROCESSES - 1))" | grep -vx "$FAULT_RANKS" | paste -sd, -)
VICTIM_GPU=${ACTIVE_GPUS[$FAULT_RANKS]}

short_host=$(hostname -s)
[[ "$short_host" == "$HOST" ]] || fail "run this on $HOST, not $short_host"
[[ -x "$INJECTOR" ]] || fail "injector $INJECTOR not found"
CONTAINER="nixl-ep-1node-$RUN_ID"

cleanup() { sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }

rail_ip() {
    local number=$((10#${1##*-}))
    echo "5.$((number / 100)).$((number % 100)).$2"
}

ipv4_gid() {
    local a b c d
    IFS=. read -r a b c d <<<"$1"
    printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x\n' "$a" "$b" "$c" "$d"
}

# One line per GPU: gpu=N uuid=U pci=P links=L recovery=R (R may contain spaces).
capture_health() {
    local output=$1 gpu uuid pci links recovery
    : >"$output"
    for gpu in 0 1 2 3 4 5 6 7; do
        uuid="" pci="" links="" recovery=""
        read -r uuid pci < <(timeout 60s nvidia-smi -i "$gpu" \
            --query-gpu=uuid,pci.bus_id --format=csv,noheader 2>/dev/null | tr -d ,) || true
        links=$(timeout 60s nvidia-smi nvlink -s -i "$gpu" 2>/dev/null |
            awk '/GB\/s/{count++} END{print count+0}') || true
        recovery=$(timeout 60s nvidia-smi -q -i "$gpu" 2>/dev/null |
            awk -F: '!found && tolower($1)~/recovery action/ {gsub(/^ +| +$/,"",$2); print $2; found=1}') || true
        printf 'gpu=%s uuid=%s pci=%s links=%s recovery=%s\n' \
            "$gpu" "${uuid:-unknown}" "${pci:-unknown}" "${links:-0}" "${recovery:-unknown}" >>"$output"
    done
}

kmsg_clock() {
    # date runs inside the root shell; sudo itself takes ~100 ms to start
    sudo sh -c 'echo "NIXL_EP_CLOCK $1 wall=$(date +%s.%N)" > /dev/kmsg' sh "$1" || :
}

run_config() {
    printf '%s\n' "image=$IMAGE" "host=$HOST" "transport=default" "rails=$RAILS_CSV" "gpus=$GPUS_CSV" \
        "tokens=$TOKENS" "experts=$EXPERTS" "topk=$TOPK" "timeout_ms=$TIMEOUT_MS" \
        "fault=$FAULT" "fault_ranks=$FAULT_RANKS" "fault_neighbours=$NEIGHBOURS" "fault_phase=$FAULT_PHASE" \
        "fault_link=$FAULT_LINK" "fault_timing=$FAULT_TIMING" "fault_dry_run=$FAULT_DRY_RUN" \
        "fault_send_offset_ms=$FAULT_SEND_OFFSET_MS"
}

mkdir -p "$RESULTS_ROOT"
[[ ! -e "$OUT" ]] || fail "$OUT already exists; use a new RUN_ID"
mkdir "$OUT" || fail "cannot create $OUT (run as your user, not root: root cannot write to /swgwork)"
run_config >"$OUT/run_config.txt"
cp "$INJECTOR" "$OUT/nvlink_hwinject"
trap cleanup EXIT

echo "=== preflight on $short_host ==="
sudo -v
{
    echo "host=$short_host time=$(date --iso-8601=seconds)"
    echo "driver=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader)"
    nvidia-smi -L
    nvidia-smi topo -m
    ibdev2netdev
    ip -br -4 addr show
} >"$OUT/preflight_$short_host.log" 2>&1
gpu_apps=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name --format=csv,noheader | awk 'NF')
[[ -z "$gpu_apps" ]] || fail "$short_host must have no GPU processes before an NVLink fault; found: $gpu_apps"
capture_health "$OUT/gpu_health_before_$short_host.txt"
for gpu in "${ACTIVE_GPUS[@]}"; do
    grep -q "^gpu=$gpu .* links=18 recovery=None$" "$OUT/gpu_health_before_$short_host.txt" ||
        fail "GPU$gpu is not healthy: $(grep "^gpu=$gpu " "$OUT/gpu_health_before_$short_host.txt")"
done
[[ "$(cat /proc/sys/net/ipv4/conf/all/arp_ignore)" == 1 &&
    "$(cat /proc/sys/net/ipv4/conf/all/arp_announce)" == 2 ]] ||
    fail "run rdma_restore.sh on $short_host first (arp_ignore=1, arp_announce=2)"

: >"$OUT/rails_$short_host.txt"
for index in "${!ACTIVE_RAILS[@]}"; do
    rail=${ACTIVE_RAILS[$index]}
    hca=${ACTIVE_HCAS[$index]}
    dev=${ACTIVE_NETDEVS[$index]}
    ip=$(rail_ip "$short_host" "$rail")
    [[ "$(cat "/sys/class/infiniband/$hca/ports/1/state")" == "4: ACTIVE" ]] || fail "$hca ($dev) is not ACTIVE"
    ip -4 -o addr show dev "$dev" | grep -q " $ip/" || fail "$ip is not configured on $dev"
    gid=$(cat "/sys/class/infiniband/$hca/ports/1/gids/$GID_INDEX")
    [[ "$gid" == "$(ipv4_gid "$ip")" ]] || fail "$hca GID[$GID_INDEX] is $gid, expected the RoCE v2 GID of $ip"
    printf '%s %s %s %s %s\n' "$rail" "$hca" "$dev" "$ip" "$(cat "/sys/class/net/$dev/address")" \
        >>"$OUT/rails_$short_host.txt"
done
# Each rail reaches the other local rails through the switch, as in run_2node.sh.
: >"$OUT/neighbors_$short_host.txt"
while read -r _ _ dev _ _; do
    while read -r _ _ other ip mac; do
        [[ "$other" == "$dev" ]] || printf '%s %s %s\n' "$dev" "$ip" "$mac" >>"$OUT/neighbors_$short_host.txt"
    done <"$OUT/rails_$short_host.txt"
done <"$OUT/rails_$short_host.txt"
while read -r dev ip mac; do
    sudo ip neigh replace "$ip" lladdr "$mac" dev "$dev" nud permanent </dev/null
done <"$OUT/neighbors_$short_host.txt"

sudo docker image inspect "$IMAGE" >/dev/null
sudo docker run --rm -v "$SCRIPT_DIR":/tools:ro -w /workspace/nixl "$IMAGE" bash -c '
    python3 /tools/patch_rdma.py && python3 /tools/patch_fault_2node.py &&
        python3 -m py_compile examples/device/ep/tests/elastic/elastic.py' \
    >"$OUT/patch_check_$short_host.log" 2>&1 ||
    fail "the fault patch does not apply to $IMAGE; see $OUT/patch_check_$short_host.log"
victim_uuid=$(nvidia-smi -i "$VICTIM_GPU" --query-gpu=uuid --format=csv,noheader)
printf 'rank=%s gpu=%s uuid=%s link=%s\n' "$FAULT_RANKS" "$VICTIM_GPU" "$victim_uuid" "$FAULT_LINK" \
    >"$OUT/fault_target.txt"
set +e
sudo docker run --rm --gpus all --cap-add SYS_ADMIN -v "$OUT":/fault:ro "$IMAGE" bash -c '
    [[ "$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)" == "$2" ]] ||
        { echo "nvidia-smi in the container does not report $2 as GPU $1"; exit 6; }
    exec /fault/nvlink_hwinject "$2"' probe "$VICTIM_GPU" "$victim_uuid" \
    >"$OUT/injector_probe_$short_host.log" 2>&1
probe_rc=$?
set -e
((probe_rc == 0)) && grep -q 'status=0x0000 NV_OK' "$OUT/injector_probe_$short_host.log" ||
    fail "in-container injector probe of GPU$VICTIM_GPU failed (rc=$probe_rc); see $OUT/injector_probe_$short_host.log"
echo "preflight passed: victim rank $FAULT_RANKS on GPU$VICTIM_GPU ($victim_uuid), neighbours $NEIGHBOURS," \
    "timing $FAULT_TIMING, dry run $FAULT_DRY_RUN"

echo "=== EP run ==="
sudo -v
dmesg_before="$OUT/.dmesg_before"
sudo dmesg >"$dmesg_before" 2>/dev/null || : >"$dmesg_before"
kmsg_clock before
set +e
timeout -k 30s 900s sudo docker run --rm --name "$CONTAINER" \
    --network=host --gpus all --ipc=host \
    --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_ADMIN \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e "NIXL_EP_NIC_MAP=$NIC_MAP" -e "NIXL_EP_GPU_MAP=$GPUS_CSV" \
    -e "UCX_IB_GID_INDEX=$GID_INDEX" -e UCX_IB_GDA_MAX_HCA_PER_GPU=16 \
    -e UCX_IB_GDA_RETAIN_INACTIVE_CTX=yes -e UCX_LOG_LEVEL=info \
    -e "NIXL_EP_FAULT=$FAULT" -e "NIXL_EP_FAULT_RANKS=$FAULT_RANKS" -e "NIXL_EP_FAULT_PHASE=$FAULT_PHASE" \
    -e "NIXL_EP_FAULT_LINK=$FAULT_LINK" -e NIXL_EP_FAULT_TOOL=/fault/nvlink_hwinject \
    -e "NIXL_EP_FAULT_TIMING=$FAULT_TIMING" -e "NIXL_EP_FAULT_NEIGHBOURS=$NEIGHBOURS" \
    -e "NIXL_EP_FAULT_DRY_RUN=$FAULT_DRY_RUN" -e "NIXL_EP_FAULT_SEND_OFFSET_MS=$FAULT_SEND_OFFSET_MS" \
    -v "$SCRIPT_DIR":/tools:ro -v "$OUT":/fault:ro -w /workspace/nixl \
    "$IMAGE" bash -lc '
        set -euo pipefail
        unset UCX_TLS UCX_NET_DEVICES NIXL_ETCD_ENDPOINTS NIXL_ETCD_PEER_URLS NIXL_ETCD_NAMESPACE
        python3 /tools/patch_rdma.py
        python3 /tools/patch_fault_2node.py
        python3 -u examples/device/ep/tests/elastic/elastic.py \
            --plan /tools/'"$PLAN_FILE"' --num-processes '"$PROCESSES"' \
            --num-experts-per-rank '"$EXPERTS"' --num-topk '"$TOPK"' --num-tokens '"$TOKENS"' \
            --timeout-ms '"$TIMEOUT_MS"'
    ' 2>&1 | tee "$OUT/ep_$short_host.log"
ep_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$ep_rc" >"$OUT/ep_$short_host.rc"

capture_health "$OUT/gpu_health_after_$short_host.txt"
kmsg_clock after
sudo dmesg >"$OUT/.dmesg_after" 2>/dev/null || : >"$OUT/.dmesg_after"
{
    boot=$(date --iso-8601=seconds -d "@$(($(date +%s) - $(cut -d. -f1 /proc/uptime)))")
    echo "# kernel timestamps are seconds since boot; $short_host booted at $boot"
    awk 'FILENAME == ARGV[1] {seen[$0] = 1; next} !($0 in seen)' "$dmesg_before" "$OUT/.dmesg_after"
} >"$OUT/dmesg_delta_$short_host.log"
rm -f "$dmesg_before" "$OUT/.dmesg_after"

set +e
python3 "$SCRIPT_DIR/summarize_1node.py" "$OUT" | tee "$OUT/summary.log"
summary_rc=${PIPESTATUS[0]}
python3 "$SCRIPT_DIR/fault_timeline.py" "$OUT" | tee "$OUT/timeline.log"
set -e
echo "EP rc=$ep_rc. Results: $OUT"
[[ "$FAULT_DRY_RUN" == 1 ]] || echo "REAL FAULT: GPU$VICTIM_GPU on $short_host needs a power cycle before the next run."
exit "$summary_rc"
