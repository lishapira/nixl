#!/usr/bin/env bash
# Two-node NIXL EP run, with or without an injected fault. By default all
# eight GPU/NIC rails are used. Set matching NIXL_EP_RAILS and NIXL_EP_GPUS
# lists to run a healthy subset, for example rails 3,4,6,7 on GPUs 2,3,5,6.
#
# NIXL_EP_TRANSPORT selects the data path:
#   default  NVLink (CUDA IPC) inside each node, GPU-initiated RDMA between nodes
#   rdma     GPU-initiated RDMA to every peer (--disable-ll-nvlink)
#
# NIXL_EP_FAULT fires a fault at the first dispatch of phase NIXL_EP_FAULT_PHASE
# (default 1). The plan keeps every rank active, so the master ranks keep
# sending to the faulted ranks and have to detect the loss themselves:
#   none     no fault, plus an ib_write_bw test per rail (default)
#   sigterm  peer ranks NIXL_EP_FAULT_RANKS (default: all) SIGTERM themselves
#   nvlink   peer rank NIXL_EP_FAULT_RANKS (default: the first) forces NVLink
#            link NIXL_EP_FAULT_LINK (default 0) of its own GPU down. The peer
#            node must be idle, and needs a power cycle afterwards.
#
# With nvlink, NIXL_EP_FAULT_TIMING sets where the link drop falls relative to
# the first dispatch of the phase on the victim's neighbours (the other ranks
# on its node):
#   timer    the victim starts the injector from a timer and keeps running, so
#            the drop lands wherever it lands (default)
#   nvlink   the neighbours hold their first dispatch until the injector has
#            returned, so its NVLink sends go to the downed GPU
#   rdma     the victim holds its first dispatch, the neighbours send theirs
#            over NVLink and wait for the victim's data, and only then is the
#            link forced down
# The split timings use the phase's first dispatch whose send and receive parts
# run as separate kernels. The victim holds it until the injector returns; each
# neighbour times its send part on the GPU and receives only after the injector:
#   before_send  the neighbours send after the link is down
#   after_send   the link goes down once every neighbour's send has finished
#   during_send  the neighbours start sending NIXL_EP_FAULT_SEND_OFFSET_MS
#                (default 10) after the injector starts, aiming the drop into it
#   before_all_send  before_send, and the master ranks also send after the
#                link is down
#   before_send_masked  before_send, but the neighbours mask the victim first
# Two more split timings hold nobody's send part; each rank on the victim's node
# logs when its send and receive parts finish on the GPU:
#   after_victim_send  the link goes down once the victim's own send has finished
#   after_recv         the link goes down once the victim and every neighbour
#                      have received; they all wait for the injector to go on
# NIXL_EP_FAULT_DRY_RUN=1 runs the same steps with the injector's probe, which
# touches no link: every rank must finish and no power cycle is needed.
#
# Start the master role first, then the peer role with the same RUN_ID and
# NIXL_EP_* settings. Each role waits up to NIXL_EP_WAIT_SECONDS (default 600)
# for the other one:
#   RUN_ID=<unique-id> bash run_2node.sh master   # on adv-dev-dgxb-201
#   RUN_ID=<same-id>   bash run_2node.sh peer     # on adv-dev-dgxb-200
#
# Both roles install permanent ARP entries for all 16 rail addresses on every
# rail interface. A reboot clears them.
set -euo pipefail

ROLE=${1:-}
if [[ "$ROLE" != "master" && "$ROLE" != "peer" ]]; then
    echo "usage: RUN_ID=<unique-id> $0 {master|peer}" >&2
    exit 2
fi
: "${RUN_ID:?set RUN_ID to the same unique value on both hosts}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${NIXL_EP_IMAGE:-nixl-ep:master}
TRANSPORT=${NIXL_EP_TRANSPORT:-default}
MASTER_HOST=${NIXL_EP_MASTER_HOST:-adv-dev-dgxb-201}
PEER_HOST=${NIXL_EP_PEER_HOST:-adv-dev-dgxb-200}
MASTER_IP=${NIXL_EP_MASTER_IP:-10.230.249.75}
RESULTS_ROOT=${NIXL_EP_RESULTS_ROOT:-/swgwork/lishapira/nixl_ep_b200_2node}
OUT="$RESULTS_ROOT/$RUN_ID"
TOKENS=${NIXL_EP_TOKENS:-8192}
EXPERTS=${NIXL_EP_EXPERTS:-2}
TOPK=${NIXL_EP_TOPK:-8}
TIMEOUT_MS=${NIXL_EP_TIMEOUT_MS:-30000}
GID_INDEX=${NIXL_EP_GID_INDEX:-3}
WAIT_SECONDS=${NIXL_EP_WAIT_SECONDS:-600}
FAULT=${NIXL_EP_FAULT:-none}
FAULT_PHASE=${NIXL_EP_FAULT_PHASE:-1}
FAULT_LINK=${NIXL_EP_FAULT_LINK:-0}
FAULT_TIMING=${NIXL_EP_FAULT_TIMING:-timer}
FAULT_DRY_RUN=${NIXL_EP_FAULT_DRY_RUN:-0}
FAULT_SEND_OFFSET_MS=${NIXL_EP_FAULT_SEND_OFFSET_MS:-10}
INJECTOR_SOURCE="$SCRIPT_DIR/../nvlink_fault/nvlink_hwinject.c"

# DGX B200 physical rail map (rail N is adjacent to GPU N-1).
ALL_RAIL_HCAS=(mlx5_4 mlx5_7 mlx5_8 mlx5_9 mlx5_10 mlx5_13 mlx5_14 mlx5_15)
ALL_RAIL_NETDEVS=(enp24s0np0 enp64s0np0 enp79s0np0 enp94s0np0 enp154s0np0 enp192s0np0 enp206s0np0 enp220s0np0)
RAILS_CSV=${NIXL_EP_RAILS:-1,2,3,4,5,6,7,8}
GPUS_CSV=${NIXL_EP_GPUS:-0,1,2,3,4,5,6,7}
config_fail() { echo "ABORT: $*" >&2; exit 2; }
IFS=, read -r -a ACTIVE_RAILS <<<"$RAILS_CSV"
IFS=, read -r -a ACTIVE_GPUS <<<"$GPUS_CSV"
PROCESSES=${#ACTIVE_RAILS[@]}
((PROCESSES > 0 && PROCESSES == ${#ACTIVE_GPUS[@]})) ||
    config_fail "NIXL_EP_RAILS and NIXL_EP_GPUS must be nonempty lists of equal length"

ACTIVE_HCAS=()
ACTIVE_NETDEVS=()
for index in "${!ACTIVE_RAILS[@]}"; do
    rail=${ACTIVE_RAILS[$index]}
    gpu=${ACTIVE_GPUS[$index]}
    [[ "$rail" =~ ^[1-8]$ && "$gpu" =~ ^[0-7]$ ]] ||
        config_fail "invalid rail/GPU pair: rail=$rail gpu=$gpu"
    [[ "$gpu" -eq "$((rail - 1))" ]] ||
        config_fail "rail $rail is not adjacent to GPU $gpu (expected GPU $((rail - 1)))"
    ACTIVE_HCAS+=("${ALL_RAIL_HCAS[$((rail - 1))]}")
    ACTIVE_NETDEVS+=("${ALL_RAIL_NETDEVS[$((rail - 1))]}")
done
NIC_MAP=$(IFS=,; echo "${ACTIVE_HCAS[*]}")
case "$((2 * PROCESSES))" in
    8) PLAN_FILE=three_phase_all_active_8rank.json ;;
    16) PLAN_FILE=three_phase_all_active_16rank.json ;;
    *) config_fail "only 4 or 8 ranks per node are currently supported" ;;
esac
EXPECTED_MASTER_RANKS=$(seq -s ' ' 0 "$((PROCESSES - 1))")

case "$FAULT" in
    none) FAULT_RANKS="" ;;
    sigterm) FAULT_RANKS=${NIXL_EP_FAULT_RANKS:-$(seq -s , "$PROCESSES" "$((2 * PROCESSES - 1))")} ;;
    nvlink) FAULT_RANKS=${NIXL_EP_FAULT_RANKS:-$PROCESSES} ;;
    *) config_fail "NIXL_EP_FAULT must be none, sigterm or nvlink" ;;
esac
if [[ "$FAULT" != none ]]; then
    [[ "$FAULT_PHASE" =~ ^[12]$ ]] || config_fail "NIXL_EP_FAULT_PHASE must be 1 or 2"
    [[ "$FAULT_LINK" =~ ^[0-9]+$ ]] && ((FAULT_LINK < 18)) ||
        config_fail "NIXL_EP_FAULT_LINK must be 0-17"
    IFS=, read -r -a fault_rank_list <<<"$FAULT_RANKS"
    for rank in "${fault_rank_list[@]}"; do
        [[ "$rank" =~ ^[0-9]+$ ]] && ((rank >= PROCESSES && rank < 2 * PROCESSES)) ||
            config_fail "fault rank $rank is not a peer rank ($PROCESSES-$((2 * PROCESSES - 1)))"
    done
    [[ "$FAULT" != nvlink || ${#fault_rank_list[@]} -eq 1 ]] ||
        config_fail "NIXL_EP_FAULT=nvlink takes exactly one rank"
fi
[[ "$FAULT_TIMING" =~ ^(timer|nvlink|rdma|before_send|after_send|during_send|before_all_send|before_send_masked|after_victim_send|after_recv)$ ]] ||
    config_fail "NIXL_EP_FAULT_TIMING must be timer, nvlink, rdma, before_send, after_send, during_send, before_all_send, before_send_masked, after_victim_send or after_recv"
[[ "$FAULT_SEND_OFFSET_MS" =~ ^[0-9]+(\.[0-9]+)?$ ]] || config_fail "NIXL_EP_FAULT_SEND_OFFSET_MS must be a number"
[[ "$FAULT_DRY_RUN" =~ ^[01]$ ]] || config_fail "NIXL_EP_FAULT_DRY_RUN must be 0 or 1"
[[ "$FAULT" == nvlink || ("$FAULT_TIMING" == timer && "$FAULT_DRY_RUN" == 0) ]] ||
    config_fail "NIXL_EP_FAULT_TIMING and NIXL_EP_FAULT_DRY_RUN apply only to NIXL_EP_FAULT=nvlink"
NEIGHBOURS=""
if [[ "$FAULT" == nvlink ]]; then
    NEIGHBOURS=$(seq "$PROCESSES" "$((2 * PROCESSES - 1))" | grep -vx "$FAULT_RANKS" | paste -sd, -)
fi

case "$TRANSPORT" in
    default) DISABLE_LL="" ;;
    rdma) DISABLE_LL="--disable-ll-nvlink" ;;
    *) echo "ABORT: NIXL_EP_TRANSPORT must be 'default' or 'rdma'" >&2; exit 2 ;;
esac

short_host=$(hostname -s)
if [[ "$ROLE" == "master" ]]; then
    expected_host=$MASTER_HOST
    remote_host=$PEER_HOST
    remote_role=peer
else
    expected_host=$PEER_HOST
    remote_host=$MASTER_HOST
    remote_role=master
fi
if [[ "$short_host" != "$expected_host" ]]; then
    echo "ABORT: role=$ROLE must run on $expected_host, not $short_host" >&2
    exit 3
fi
CONTAINER="nixl-ep-2node-$RUN_ID-$ROLE"

fail() {
    echo "ABORT: $*" >&2
    exit 1
}

cleanup() {
    sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

wait_for_file() {
    local path=$1 timeout_seconds=${2:-$WAIT_SECONDS}
    local deadline=$((SECONDS + timeout_seconds))
    until [[ -f "$path" ]]; do
        ((SECONDS < deadline)) || fail "timed out waiting for $path"
        sleep 1
    done
}

rail_ip() {
    local host=$1 rail=$2 number
    number=$((10#${host##*-}))
    echo "5.$((number / 100)).$((number % 100)).$rail"
}

ipv4_gid() {
    local a b c d
    IFS=. read -r a b c d <<<"$1"
    printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x\n' "$a" "$b" "$c" "$d"
}

capture_ib() {
    local output=$1 hca counters
    : >"$output"
    for hca in "${ACTIVE_HCAS[@]}"; do
        counters="/sys/class/infiniband/$hca/ports/1/counters"
        printf '%s %s %s\n' \
            "$hca" \
            "$(cat "$counters/port_xmit_data")" \
            "$(cat "$counters/port_rcv_data")" >>"$output"
    done
}

capture_nvlink() {
    local output=$1 gpu
    : >"$output"
    for gpu in "${ACTIVE_GPUS[@]}"; do
        timeout 60s nvidia-smi nvlink -gt d -i "$gpu" >>"$output" 2>&1 || true
    done
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
            awk -F: '
                !found && tolower($1)~/recovery action/ {
                    gsub(/^ +| +$/,"",$2)
                    print $2
                    found=1
                }
            ') || true
        printf 'gpu=%s uuid=%s pci=%s links=%s recovery=%s\n' \
            "$gpu" "${uuid:-unknown}" "${pci:-unknown}" "${links:-0}" "${recovery:-unknown}" >>"$output"
    done
}

# Raw timestamps: -T/iso times are recomputed on every call, so old lines would look new.
dmesg_snapshot() {
    sudo dmesg >"$1" 2>/dev/null || : >"$1"
}

# Raw kernel times drift from the wall clock; this line pins them to it for fault_timeline.py.
kmsg_clock() {
    # date runs inside the root shell; sudo itself takes ~100 ms to start
    sudo sh -c 'echo "NIXL_EP_CLOCK $1 wall=$(date +%s.%N)" > /dev/kmsg' sh "$1" || :
}

dmesg_delta() {
    local before=$1 after=$2 output=$3 boot
    boot=$(date --iso-8601=seconds -d "@$(($(date +%s) - $(cut -d. -f1 /proc/uptime)))")
    {
        echo "# kernel timestamps are seconds since boot; $short_host booted at $boot"
        awk 'FILENAME == ARGV[1] {seen[$0] = 1; next} !($0 in seen)' "$before" "$after"
    } >"$output"
    rm -f "$before" "$after"
}

capture_after() {
    capture_ib "$OUT/ib_after_$short_host.txt"
    capture_nvlink "$OUT/nvlink_after_$short_host.log"
    if [[ "$FAULT" != none ]]; then
        capture_health "$OUT/gpu_health_after_$short_host.txt"
        kmsg_clock after
        dmesg_snapshot "$OUT/.dmesg_after_$short_host"
        dmesg_delta "$OUT/.dmesg_before_$short_host" "$OUT/.dmesg_after_$short_host" \
            "$OUT/dmesg_delta_$short_host.log"
    fi
}

run_config() {
    printf '%s\n' "image=$IMAGE" "transport=$TRANSPORT" "rails=$RAILS_CSV" "gpus=$GPUS_CSV" \
        "tokens=$TOKENS" "experts=$EXPERTS" "topk=$TOPK" "timeout_ms=$TIMEOUT_MS" \
        "fault=$FAULT" "fault_ranks=$FAULT_RANKS" "fault_phase=$FAULT_PHASE" "fault_link=$FAULT_LINK" \
        "fault_timing=$FAULT_TIMING" "fault_dry_run=$FAULT_DRY_RUN" "fault_send_offset_ms=$FAULT_SEND_OFFSET_MS"
}

preflight() {
    local rail hca dev ip owners gid gid_type gpu gpu_apps links recovery victim_gpu victim_uuid probe_rc
    {
        echo "host=$short_host role=$ROLE transport=$TRANSPORT fault=$FAULT time=$(date --iso-8601=seconds)"
        echo "image=$IMAGE tokens=$TOKENS experts=$EXPERTS topk=$TOPK timeout_ms=$TIMEOUT_MS"
        echo "driver=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader)"
        nvidia-smi -L
        nvidia-smi topo -m
        ibdev2netdev
        ip -br -4 addr show
        ip -br link show
    } >"$OUT/preflight_$short_host.log" 2>&1

    [[ $(nvidia-smi -L | wc -l) -eq 8 ]] || fail "$short_host does not have 8 visible GPUs"
    for gpu in "${ACTIVE_GPUS[@]}"; do
        gpu_apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader |
            awk 'NF')
        [[ -z "$gpu_apps" ]] ||
            fail "$short_host selected GPU$gpu has active process(es): $gpu_apps"
        links=$(nvidia-smi nvlink -s -i "$gpu" 2>/dev/null |
            awk '/GB\/s/{count++} END{print count+0}')
        recovery=$(nvidia-smi -q -i "$gpu" 2>/dev/null |
            awk -F: '
                !found && tolower($1)~/recovery action/ {
                    gsub(/^ +| +$/,"",$2)
                    print $2
                    found=1
                }
            ')
        [[ "$links" == 18 && "$recovery" == None ]] ||
            fail "$short_host GPU$gpu links=$links recovery=${recovery:-N/A}"
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
        [[ -e "/sys/class/infiniband/$hca/device/net/$dev" ]] ||
            fail "$short_host: $hca is not the HCA of $dev"
        [[ "$(cat "/sys/class/infiniband/$hca/ports/1/state")" == "4: ACTIVE" ]] ||
            fail "$short_host: $hca ($dev) is not ACTIVE"
        owners=$(ip -4 -o addr show |
            awk -v ip="$ip" '{split($4, address, "/")} address[1] == ip {print $2}' |
            sort -u | xargs)
        [[ "$owners" == "$dev" ]] ||
            fail "$short_host: $ip must be configured only on $dev (found on: ${owners:-none})"
        gid=$(cat "/sys/class/infiniband/$hca/ports/1/gids/$GID_INDEX")
        gid_type=$(cat "/sys/class/infiniband/$hca/ports/1/gid_attrs/types/$GID_INDEX" 2>/dev/null || true)
        [[ "$gid_type" == "RoCE v2" && "$gid" == "$(ipv4_gid "$ip")" ]] ||
            fail "$short_host: $hca GID[$GID_INDEX] is $gid ($gid_type), expected RoCE v2 for $ip"
        printf '%s %s %s %s %s\n' \
            "$rail" "$hca" "$dev" "$ip" "$(cat "/sys/class/net/$dev/address")" \
            >>"$OUT/rails_$short_host.txt"
    done

    sudo -v
    sudo docker image inspect "$IMAGE" >/dev/null

    if [[ "$FAULT" != none ]]; then
        sudo docker run --rm -v "$SCRIPT_DIR":/tools:ro -w /workspace/nixl "$IMAGE" bash -c '
            python3 /tools/patch_rdma.py && python3 /tools/patch_fault_2node.py &&
                python3 -m py_compile examples/device/ep/tests/elastic/elastic.py' \
            >"$OUT/patch_check_$short_host.log" 2>&1 ||
            fail "the fault patch does not apply to $IMAGE; see $OUT/patch_check_$short_host.log"
        capture_health "$OUT/gpu_health_before_$short_host.txt"
    fi
    if [[ "$FAULT" == nvlink && "$ROLE" == peer ]]; then
        gpu_apps=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name --format=csv,noheader |
            awk 'NF')
        [[ -z "$gpu_apps" ]] ||
            fail "$short_host must have no GPU processes before an NVLink fault; found: $gpu_apps"
        victim_gpu=${ACTIVE_GPUS[$((FAULT_RANKS - PROCESSES))]}
        victim_uuid=$(nvidia-smi -i "$victim_gpu" --query-gpu=uuid --format=csv,noheader)
        printf 'rank=%s gpu=%s uuid=%s link=%s\n' \
            "$FAULT_RANKS" "$victim_gpu" "$victim_uuid" "$FAULT_LINK" >"$OUT/fault_target.txt"
        # The victim rank looks up its own UUID with nvidia-smi inside the container.
        set +e
        sudo docker run --rm --gpus all --cap-add SYS_ADMIN -v "$OUT":/fault:ro "$IMAGE" bash -c '
            [[ "$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)" == "$2" ]] ||
                { echo "nvidia-smi in the container does not report $2 as GPU $1"; exit 6; }
            exec /fault/nvlink_hwinject "$2"' probe "$victim_gpu" "$victim_uuid" \
            >"$OUT/injector_probe_$short_host.log" 2>&1
        probe_rc=$?
        set -e
        ((probe_rc == 0)) && grep -q 'status=0x0000 NV_OK' "$OUT/injector_probe_$short_host.log" ||
            fail "in-container injector probe of GPU$victim_gpu ($victim_uuid) failed (rc=$probe_rc);" \
                "see $OUT/injector_probe_$short_host.log"
        echo "$short_host: the injector reaches GPU$victim_gpu ($victim_uuid) from the container"
    fi
    echo "$short_host preflight passed"
}

install_neighbors() {
    local local_table="$OUT/rails_$short_host.txt"
    local remote_table="$OUT/rails_$remote_host.txt"
    local table="$OUT/neighbors_$short_host.txt"
    local dev peer_dev ip mac missing

    : >"$table"
    while read -r _ _ dev _ _; do
        while read -r _ _ peer_dev ip mac; do
            [[ "$peer_dev" == "$dev" ]] || printf '%s %s %s\n' "$dev" "$ip" "$mac" >>"$table"
        done <"$local_table"
        while read -r _ _ _ ip mac; do
            printf '%s %s %s\n' "$dev" "$ip" "$mac" >>"$table"
        done <"$remote_table"
    done <"$local_table"

    while read -r dev ip mac; do
        sudo ip neigh replace "$ip" lladdr "$mac" dev "$dev" nud permanent </dev/null
    done <"$table"

    missing=$(awk '
        NR == FNR {want[$1 " " $2 " " $3] = 1; next}
        FNR > 1 && $3 == "0x6" {delete want[$6 " " $1 " " $4]}
        END {for (entry in want) print "  " entry}
    ' "$table" /proc/net/arp)
    [[ -z "$missing" ]] || fail "$short_host: permanent neighbors missing:"$'\n'"$missing"
    echo "$short_host: $(wc -l <"$table") permanent rail neighbors installed"
}

check_rail_matrix() {
    local dev ip payload failures=0
    : >"$OUT/ping_$short_host.log"
    while read -r _ _ dev _ _; do
        # RoCE sends 4096-byte packets; a path that drops full-size frames fails later as RDMA retries.
        payload=$(($(cat "/sys/class/net/$dev/mtu") - 28))
        while read -r _ _ _ ip _; do
            if ! ping -I "$dev" -c 2 -W 2 "$ip" >/dev/null 2>&1 </dev/null; then
                echo "FAIL $dev -> $ip" | tee -a "$OUT/ping_$short_host.log"
                failures=$((failures + 1))
            elif ! ping -I "$dev" -M do -s "$payload" -c 2 -W 2 "$ip" >/dev/null 2>&1 </dev/null; then
                echo "FAIL $dev -> $ip with $payload-byte DF payload" | tee -a "$OUT/ping_$short_host.log"
                failures=$((failures + 1))
            else
                echo "ok   $dev -> $ip" >>"$OUT/ping_$short_host.log"
            fi
        done <"$OUT/rails_$remote_host.txt"
    done <"$OUT/rails_$short_host.txt"
    pairs=$((PROCESSES * PROCESSES))
    ((failures == 0)) || fail "$failures of $pairs rail pairs from $short_host to $remote_host are unreachable"
    echo "$short_host: all $pairs selected rail pairs to $remote_host are reachable"
}

run_ep() {
    local tcp_server=$1 fault_args=()
    if [[ "$FAULT" != none ]]; then
        fault_args+=(-e "NIXL_EP_FAULT=$FAULT" -e "NIXL_EP_FAULT_RANKS=$FAULT_RANKS")
        fault_args+=(-e "NIXL_EP_FAULT_PHASE=$FAULT_PHASE" -e "NIXL_EP_FAULT_TIMING=$FAULT_TIMING")
    fi
    if [[ "$FAULT" == nvlink && "$ROLE" == peer ]]; then
        fault_args+=(--cap-add SYS_ADMIN -v "$OUT":/fault:ro)
        fault_args+=(-e "NIXL_EP_FAULT_LINK=$FAULT_LINK" -e NIXL_EP_FAULT_TOOL=/fault/nvlink_hwinject)
        fault_args+=(-e "NIXL_EP_FAULT_NEIGHBOURS=$NEIGHBOURS")
        fault_args+=(-e "NIXL_EP_FAULT_DRY_RUN=$FAULT_DRY_RUN" -e "NIXL_EP_FAULT_SEND_OFFSET_MS=$FAULT_SEND_OFFSET_MS")
    fi
    timeout -k 30s 900s sudo docker run --rm --name "$CONTAINER" \
        --network=host --gpus all --ipc=host \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -e "NIXL_EP_NIC_MAP=$NIC_MAP" \
        -e "NIXL_EP_GPU_MAP=$GPUS_CSV" \
        -e "NIXL_EP_TCP_SERVER=$tcp_server" \
        -e "NIXL_EP_DISABLE_LL=$DISABLE_LL" \
        -e "UCX_IB_GID_INDEX=$GID_INDEX" \
        -e UCX_IB_GDA_MAX_HCA_PER_GPU=16 \
        -e UCX_IB_GDA_RETAIN_INACTIVE_CTX=yes \
        -e UCX_LOG_LEVEL=info \
        "${fault_args[@]}" \
        -v "$SCRIPT_DIR":/tools:ro \
        -w /workspace/nixl \
        "$IMAGE" bash -lc '
            set -euo pipefail
            unset UCX_TLS UCX_NET_DEVICES
            unset NIXL_ETCD_ENDPOINTS NIXL_ETCD_PEER_URLS NIXL_ETCD_NAMESPACE
            python3 /tools/patch_rdma.py
            if [[ -n "${NIXL_EP_FAULT:-}" ]]; then
                python3 /tools/patch_fault_2node.py
            fi
            extra=()
            if [[ -n "$NIXL_EP_TCP_SERVER" ]]; then
                extra+=(--tcp-server "$NIXL_EP_TCP_SERVER")
            fi
            if [[ -n "$NIXL_EP_DISABLE_LL" ]]; then
                extra+=("$NIXL_EP_DISABLE_LL")
            fi
            python3 -u examples/device/ep/tests/elastic/elastic.py \
                --plan /tools/'"$PLAN_FILE"' \
                --num-processes '"$PROCESSES"' \
                --num-experts-per-rank '"$EXPERTS"' \
                --num-topk '"$TOPK"' \
                --num-tokens '"$TOKENS"' \
                --timeout-ms '"$TIMEOUT_MS"' \
                "${extra[@]}"
        '
}

mkdir -p "$RESULTS_ROOT"
if [[ "$ROLE" == "master" ]]; then
    mkdir "$OUT" 2>/dev/null || fail "$OUT already exists; use a new RUN_ID"
    run_config >"$OUT/run_config.txt"
    if [[ "$FAULT" == nvlink ]]; then
        gcc -O1 -Wall -Wextra -static -o "$OUT/nvlink_hwinject" "$INJECTOR_SOURCE" ||
            fail "could not build $INJECTOR_SOURCE"
    fi
    touch "$OUT/initialized"
else
    wait_for_file "$OUT/initialized"
    if ! config_diff=$(diff <(run_config) "$OUT/run_config.txt"); then
        fail "NIXL_EP_* settings differ from the master's (< peer, > master):"$'\n'"$config_diff"
    fi
fi
trap cleanup EXIT

preflight
touch "$OUT/preflight_$ROLE.ready"
wait_for_file "$OUT/preflight_$remote_role.ready"

install_neighbors
touch "$OUT/neighbors_$ROLE.ready"
wait_for_file "$OUT/neighbors_$remote_role.ready"

check_rail_matrix
touch "$OUT/matrix_$ROLE.ready"
wait_for_file "$OUT/matrix_$remote_role.ready"

if [[ "$ROLE" == "peer" ]]; then
    if [[ "$FAULT" == none ]]; then
        for index in "${!ACTIVE_RAILS[@]}"; do
            rail=${ACTIVE_RAILS[$index]}
            hca=${ACTIVE_HCAS[$index]}
            set +e
            timeout 60s ib_write_bw -d "$hca" -x "$GID_INDEX" --report_gbits \
                >"$OUT/perftest_server_rail$rail.log" 2>&1 &
            server_pid=$!
            set -e
            sleep 1
            touch "$OUT/perftest_server_rail$rail.ready"
            set +e
            wait "$server_pid"
            server_rc=$?
            set -e
            printf '%s\n' "$server_rc" >"$OUT/perftest_server_rail$rail.rc"
            ((server_rc == 0)) || fail "rail$rail ib_write_bw server failed (rc=$server_rc)"
        done
    fi

    wait_for_file "$OUT/ep_master.ready"
    sudo -v
    capture_ib "$OUT/ib_before_$short_host.txt"
    capture_nvlink "$OUT/nvlink_before_$short_host.log"
    if [[ "$FAULT" != none ]]; then
        dmesg_snapshot "$OUT/.dmesg_before_$short_host"
        kmsg_clock before
    fi
    set +e
    run_ep "$MASTER_IP" 2>&1 | tee "$OUT/ep_$short_host.log"
    ep_rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$ep_rc" >"$OUT/ep_$short_host.rc"
    capture_after
    touch "$OUT/ep_peer.done"
    echo "peer EP rc=$ep_rc; the master prints the verdict. Results: $OUT"
    exit "$ep_rc"
fi

if [[ "$FAULT" == none ]]; then
    for index in "${!ACTIVE_RAILS[@]}"; do
        rail=${ACTIVE_RAILS[$index]}
        hca=${ACTIVE_HCAS[$index]}
        destination=$(rail_ip "$PEER_HOST" "$rail")
        wait_for_file "$OUT/perftest_server_rail$rail.ready" 120
        set +e
        timeout 45s ib_write_bw -d "$hca" -x "$GID_INDEX" -n 2000 --report_gbits \
            "$destination" >"$OUT/perftest_client_rail$rail.log" 2>&1
        client_rc=$?
        set -e
        printf '%s\n' "$client_rc" >"$OUT/perftest_client_rail$rail.rc"
        ((client_rc == 0)) || fail "rail$rail ib_write_bw to $destination failed (rc=$client_rc)"
        echo "rail$rail RDMA $hca -> $destination: $(awk '$1 == 65536 {print $4 " Gb/s avg"}' \
            "$OUT/perftest_client_rail$rail.log")"
    done
fi

sudo -v
capture_ib "$OUT/ib_before_$short_host.txt"
capture_nvlink "$OUT/nvlink_before_$short_host.log"
if [[ "$FAULT" != none ]]; then
    dmesg_snapshot "$OUT/.dmesg_before_$short_host"
    kmsg_clock before
fi
master_log="$OUT/ep_$short_host.log"
: >"$master_log"
(
    set -o pipefail
    run_ep "" 2>&1 | tee "$master_log"
) &
master_job=$!

deadline=$((SECONDS + 180))
while true; do
    assigned=$(awk '
        /Process [0-9]+ -> global_rank=[0-9]+, local_rank=[0-9]+/ {
            line = $0
            sub(/.*global_rank=/, "", line)
            sub(/,.*/, "", line)
            print line
        }
    ' "$master_log" | sort -nu | xargs)
    [[ "$assigned" == "$EXPECTED_MASTER_RANKS" ]] && break
    if ! kill -0 "$master_job" 2>/dev/null; then
        wait "$master_job" || true
        fail "master EP exited before ranks $EXPECTED_MASTER_RANKS were assigned"
    fi
    ((SECONDS < deadline)) || fail "timed out assigning master ranks; got: ${assigned:-none}"
    sleep 1
done
touch "$OUT/ep_master.ready"
echo "Master ranks $EXPECTED_MASTER_RANKS assigned; the peer joins now."

set +e
wait "$master_job"
master_rc=$?
set -e
printf '%s\n' "$master_rc" >"$OUT/ep_$short_host.rc"
capture_after

if [[ "$FAULT" == none ]]; then
    wait_for_file "$OUT/ep_peer.done" 300
    summary=("$SCRIPT_DIR/summarize_baseline.py" "$OUT" "$TRANSPORT" "$PROCESSES" "$RAILS_CSV")
else
    echo "master EP rc=$master_rc; waiting for the peer to finish"
    wait_for_file "$OUT/ep_peer.done" 1000
    summary=("$SCRIPT_DIR/summarize_fault.py" "$OUT" "$FAULT" "$PROCESSES" "$FAULT_RANKS" "$FAULT_PHASE")
fi
set +e
python3 "${summary[@]}" | tee "$OUT/summary.log"
summary_rc=${PIPESTATUS[0]}
[[ "$FAULT" == none ]] || python3 "$SCRIPT_DIR/fault_timeline.py" "$OUT" | tee "$OUT/timeline.log"
set -e
echo "Results: $OUT"
exit "$summary_rc"
