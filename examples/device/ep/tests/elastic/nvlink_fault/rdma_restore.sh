#!/usr/bin/env bash
#
# Restore RoCE/RDMA host-side configuration after a reboot or power cycle.
#
# Usage:
#   sudo ./rdma_restore.sh
#   sudo ./rdma_restore.sh --test
#
# The --test form also installs static any-to-any neighbors and validates
# cross-NIC RDMA. It is required before the cross-NIC NIXL EP experiment.
set -euo pipefail

NICS="enp24s0np0 enp64s0np0 enp79s0np0 enp94s0np0 enp154s0np0 enp192s0np0 enp206s0np0 enp220s0np0"

NUM=$(hostname | grep -oE '[0-9]+$' || true)
if [[ -z "$NUM" ]]; then
    echo "ERROR: could not parse server number from hostname '$(hostname)'"
    exit 1
fi
X=$((10#$NUM / 100))
Y=$((10#$NUM % 100))
NN=5
echo "Host $(hostname): server number=$NUM -> subnet base ${NN}.${X}.${Y}.E/8"

for dev in $NICS; do
    ip link set "$dev" up
done

i=1
for dev in $NICS; do
    ip addr add "${NN}.${X}.${Y}.${i}/8" dev "$dev" 2>/dev/null || true
    i=$((i + 1))
done

sysctl -w net.ipv4.conf.all.arp_ignore=1 \
           net.ipv4.conf.all.arp_announce=2 \
           net.ipv4.conf.all.rp_filter=2 >/dev/null

echo "=== link + IP ==="
for dev in $NICS; do
    state=$(ip -br link show "$dev" | awk '{print $2}')
    ip4=$(ip -4 -o addr show dev "$dev" | awk '{print $4}')
    printf "  %-14s %-5s %s\n" "$dev" "$state" "$ip4"
done

if [[ "${1:-}" == "--test" ]]; then
    echo "=== setting same-host static neighbors (any-to-any) ==="
    declare -A IPOF MACOF
    i=1
    for dev in $NICS; do
        IPOF[$dev]="${NN}.${X}.${Y}.${i}"
        MACOF[$dev]=$(cat "/sys/class/net/$dev/address")
        i=$((i + 1))
    done

    for src in $NICS; do
        for dst in $NICS; do
            [[ "$src" == "$dst" ]] && continue
            ip neigh replace "${IPOF[$dst]}" lladdr "${MACOF[$dst]}" \
                dev "$src" nud permanent
        done
    done

    echo "=== cross-NIC RDMA test: each NIC -> mlx5_4 ==="
    for dev in mlx5_7 mlx5_8 mlx5_9 mlx5_10 mlx5_13 mlx5_14 mlx5_15; do
        ib_write_bw -d mlx5_4 -x 3 --report_gbits >/tmp/rdma_srv.log 2>&1 &
        server_pid=$!
        sleep 1.5
        res=$(ib_write_bw -d "$dev" -x 3 -n 2000 --report_gbits \
            127.0.0.1 2>/dev/null | awk '/65536/{print $3" Gb/s peak"}')
        wait "$server_pid" || true
        printf "  %-8s -> mlx5_4 : %s\n" "$dev" "${res:-FAILED}"
        sleep 0.5
    done
fi

echo "Done."
