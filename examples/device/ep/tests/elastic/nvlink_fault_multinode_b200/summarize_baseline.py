#!/usr/bin/env python3
"""Validate a two-node NIXL EP baseline.

usage: summarize_baseline.py RESULTS_DIR {default|rdma} RANKS_PER_NODE RAILS
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MASTER = "adv-dev-dgxb-201"
PEER = "adv-dev-dgxb-200"
PHASES = range(3)


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def hca_deltas(before_path: Path, after_path: Path) -> dict[str, int]:
    def load(path: Path) -> dict[str, tuple[int, int]]:
        values = {}
        for line in path.read_text().splitlines():
            hca, tx, rx = line.split()
            values[hca] = (int(tx), int(rx))
        return values

    before, after = load(before_path), load(after_path)
    return {
        hca: after[hca][0] - before[hca][0] + after[hca][1] - before[hca][1]
        for hca in before
    }


def nvlink_total(path: Path) -> int:
    return sum(
        int(match.group(1))
        for match in re.finditer(
            r"Data (?:Tx|Rx):\s+(\d+)", path.read_text(errors="replace")
        )
    )


def main(results: Path, transport: str, ranks_per_node: int, rails: list[int]) -> int:
    expected_ranks = set(range(2 * ranks_per_node))
    logs = {
        host: (results / f"ep_{host}.log").read_text(errors="replace")
        for host in (MASTER, PEER)
    }
    combined = "\n".join(logs.values())

    ranks = {}
    for host, log in logs.items():
        pairs = {
            (int(global_rank), int(local_rank))
            for global_rank, local_rank in re.findall(
                r"Process \d+ -> global_rank=(\d+), local_rank=(\d+)", log
            )
        }
        ranks[host] = pairs

    done = {
        int(rank)
        for rank in re.findall(r"global_rank=(\d+), local_rank=\d+ -> done", combined)
    }
    phase_ends = {
        phase: {
            int(rank)
            for rank in re.findall(
                rf"global_rank=(\d+), local_rank=\d+ -> end phase {phase}", combined
            )
        }
        for phase in PHASES
    }
    failures_detected = len(re.findall(r"detected rank failures", combined))

    rc_gda_lines = combined.count("device(rc_gda/")
    cuda_ipc_lines = combined.count("cuda_ipc/")
    error_counts = {
        pattern: combined.lower().count(pattern)
        for pattern in (
            "traceback",
            "ucx  error",
            "illegal memory access",
            "uncorrectable nvlink error",
        )
    }

    deltas = {}
    nvlink = {}
    for host in (MASTER, PEER):
        deltas[host] = hca_deltas(
            results / f"ib_before_{host}.txt", results / f"ib_after_{host}.txt"
        )
        nvlink[host] = nvlink_total(results / f"nvlink_after_{host}.log") - nvlink_total(
            results / f"nvlink_before_{host}.log"
        )

    perftest_rcs = [
        read_int(results / f"perftest_client_rail{rail}.rc") for rail in rails
    ]
    ep_rcs = {host: read_int(results / f"ep_{host}.rc") for host in (MASTER, PEER)}

    checks = {
        f"master holds global ranks 0-{ranks_per_node - 1}": ranks[MASTER]
        == {(rank, rank) for rank in range(ranks_per_node)},
        f"peer holds global ranks {ranks_per_node}-{2 * ranks_per_node - 1}":
        ranks[PEER]
        == {(rank + ranks_per_node, rank) for rank in range(ranks_per_node)},
        f"all {2 * ranks_per_node} ranks completed all 3 phases":
        done == expected_ranks
        and all(ended == expected_ranks for ended in phase_ends.values()),
        "no rank failures detected": failures_detected == 0,
        "no workload errors": not any(error_counts.values()),
        "both EP parents exited 0": all(rc == 0 for rc in ep_rcs.values()),
        f"all {len(rails)} selected-rail perftests passed":
        perftest_rcs == [0] * len(rails),
        "rc_gda selected": rc_gda_lines > 0,
        "every rail HCA on both hosts moved RDMA data": all(
            delta > 0 for host_deltas in deltas.values() for delta in host_deltas.values()
        ),
    }
    if transport == "default":
        checks["cuda_ipc selected inside each node"] = cuda_ipc_lines > 0
        checks["NVLink counters increased on both hosts"] = all(
            delta > 0 for delta in nvlink.values()
        )
    else:
        checks["cuda_ipc absent"] = cuda_ipc_lines == 0
        checks["NVLink counters stayed flat on both hosts"] = all(
            delta == 0 for delta in nvlink.values()
        )

    print(
        f"=== two-node NIXL EP baseline "
        f"({transport}, {ranks_per_node} ranks/node, rails {rails}) ==="
    )
    for host in (MASTER, PEER):
        print(f"{host}: global ranks {sorted(rank for rank, _ in ranks[host])}, EP rc={ep_rcs[host]}")
        print(f"  RDMA bytes/4 per HCA: {deltas[host]}")
        print(f"  NVLink data delta (KiB): {nvlink[host]}")
    print(f"done ranks: {sorted(done)}")
    print(f"phase-end counts: {[len(phase_ends[phase]) for phase in PHASES]}")
    print(f"UCX lines: rc_gda={rc_gda_lines} cuda_ipc={cuda_ipc_lines}")
    print(f"errors: {error_counts}  detected rank failures: {failures_detected}")
    print(f"perftest rcs: {perftest_rcs}")
    print()
    for description, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {description}")

    passed = all(checks.values())
    print()
    print(f"RESULT: TWO-NODE BASELINE {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[2] not in {"default", "rdma"}:
        raise SystemExit(
            f"usage: {sys.argv[0]} RESULTS_DIR "
            "{default|rdma} RANKS_PER_NODE RAILS"
        )
    selected_rails = [int(rail) for rail in sys.argv[4].split(",")]
    ranks_per_node = int(sys.argv[3])
    if ranks_per_node < 1 or len(selected_rails) != ranks_per_node:
        raise SystemExit("RANKS_PER_NODE must equal the number of selected RAILS")
    raise SystemExit(
        main(Path(sys.argv[1]), sys.argv[2], ranks_per_node, selected_rails)
    )
