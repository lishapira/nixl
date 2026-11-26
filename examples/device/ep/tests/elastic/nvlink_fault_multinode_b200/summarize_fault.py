#!/usr/bin/env python3
"""Validate a two-node NIXL EP fault run.

usage: summarize_fault.py RESULTS_DIR {sigterm|nvlink} RANKS_PER_NODE FAULT_RANKS FAULT_PHASE

The master node holds global ranks 0..RANKS_PER_NODE-1 and is expected to
survive. The faulted ranks are on the peer node.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MASTER = "adv-dev-dgxb-201"
PEER = "adv-dev-dgxb-200"
LAST_PHASE = 2
NVLINKS = 18


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def pci_key(address: str) -> str:
    bus, device = address.lower().split(".")[0].split(":")[-2:]
    return f"{bus}:{device}"


def read_health(path: Path) -> dict[int, dict[str, str]]:
    health = {}
    for line in read_text(path).splitlines():
        match = re.match(r"gpu=(\d+) uuid=(\S+) pci=(\S+) links=(\d+) recovery=(.*)$", line)
        if match:
            gpu, uuid, pci, links, recovery = match.groups()
            health[int(gpu)] = {"uuid": uuid, "pci": pci, "links": int(links), "recovery": recovery}
    return health


def xids(path: Path) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for pci, xid in re.findall(r"NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+)", read_text(path)):
        key = (pci_key(pci), int(xid))
        counts[key] = counts.get(key, 0) + 1
    return counts


def rank_set(pattern: str, text: str) -> set[int]:
    return {int(rank) for rank in re.findall(pattern, text)}


def bandwidth_by_phase(log: str) -> dict[tuple[int, int], float]:
    phase_of: dict[int, int] = {}
    bandwidth = {}
    for line in log.splitlines():
        start = re.search(r"global_rank=(\d+), local_rank=\d+ -> start phase (\d+)", line)
        if start:
            phase_of[int(start.group(1))] = int(start.group(2))
            continue
        result = re.search(r"\[rank (\d+)\] Dispatch \+ combine bandwidth: ([\d.]+) GB/s", line)
        if result and int(result.group(1)) in phase_of:
            rank = int(result.group(1))
            bandwidth[(rank, phase_of[rank])] = float(result.group(2))
    return bandwidth


def worker_ranks(log: str) -> dict[int, tuple[int, int]]:
    return {
        int(worker): (int(rank), int(local))
        for worker, rank, local in re.findall(r"Process (\d+) -> global_rank=(\d+), local_rank=(\d+)", log)
    }


def worker_exit_codes(log: str) -> dict[int, int]:
    return {
        int(worker): int(code)
        for line in re.findall(r"Worker processes failed: (.*)", log)
        for worker, code in re.findall(r"worker (\d+) \(exit code (-?\d+)\)", line)
    }


def run_config(results: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in read_text(results / "run_config.txt").splitlines() if "=" in line
    )


def run_gpus(results: Path) -> list[int]:
    return [int(gpu) for gpu in run_config(results).get("gpus", "").split(",") if gpu]


def main(results: Path, mode: str, ranks_per_node: int, fault_ranks: set[int], fault_phase: int) -> int:
    survivors = set(range(ranks_per_node))
    peer_ranks = set(range(ranks_per_node, 2 * ranks_per_node))
    config = run_config(results)
    timing = config.get("fault_timing", "timer")
    dry_run = config.get("fault_dry_run") == "1"
    logs = {host: read_text(results / f"ep_{host}.log") for host in (MASTER, PEER)}
    combined = "\n".join(logs.values())
    ep_rcs = {host: read_int(results / f"ep_{host}.rc") for host in (MASTER, PEER)}
    assigned = {
        host: rank_set(r"Process \d+ -> global_rank=(\d+), local_rank=\d+", log)
        for host, log in logs.items()
    }
    done = rank_set(r"global_rank=(\d+), local_rank=\d+ -> done", combined)
    phase_ends = {
        phase: rank_set(rf"global_rank=(\d+), local_rank=\d+ -> end phase {phase}\b", combined)
        for phase in range(LAST_PHASE + 1)
    }
    detected: dict[int, set[int]] = {}
    for rank, members in re.findall(
        r"global_rank=(\d+), local_rank=\d+ -> detected rank failures: \{([^}]*)\}", combined
    ):
        detected.setdefault(int(rank), set()).update(int(m) for m in re.findall(r"\d+", members))
    armed = {
        int(rank): (fault, int(gpu), uuid, int(phase))
        for rank, fault, gpu, uuid, phase in re.findall(
            r"\[rank (\d+)\] armed (\w+) fault on GPU (\d+)(?: \((GPU-[0-9a-fA-F-]+)\))? for phase (\d+)",
            combined,
        )
    }
    fired = rank_set(r"\[rank (\d+)\] Killing rank during dispatch/combine", combined)
    lost = peer_ranks - phase_ends[fault_phase]
    peer_finished = peer_ranks & phase_ends[fault_phase]
    errors = {
        host: {
            "uncorrectable NVLink": log.lower().count("uncorrectable nvlink error"),
            "Traceback": log.count("Traceback"),
            "SIGTERM handled": len(re.findall(r"SIGTERM \(15\) received", log)),
        }
        for host, log in logs.items()
    }
    worker_failures = {
        host: re.findall(r"Worker processes failed: (.*?)(?: \[t=[\d.]+\])?$", log, re.M)
        for host, log in logs.items()
    }
    bandwidth = bandwidth_by_phase(combined)
    health_before = {host: read_health(results / f"gpu_health_before_{host}.txt") for host in (MASTER, PEER)}
    health_after = {host: read_health(results / f"gpu_health_after_{host}.txt") for host in (MASTER, PEER)}
    dmesg_xids = {host: xids(results / f"dmesg_delta_{host}.log") for host in (MASTER, PEER)}
    gpus = run_gpus(results)
    rank_gpu: dict[int, int | None] = {}
    rank_exit: dict[int, int] = {}
    for log in logs.values():
        codes = worker_exit_codes(log)
        for worker, (rank, local) in worker_ranks(log).items():
            rank_gpu[rank] = gpus[local % len(gpus)] if gpus else None
            if worker in codes:
                rank_exit[rank] = codes[worker]

    def gpus_of(ranks) -> list:
        return [rank_gpu.get(rank) for rank in sorted(ranks)]

    def outcome(rank: int) -> str:
        if rank in done:
            return "finished all phases"
        ended = [phase for phase in range(LAST_PHASE + 1) if rank in phase_ends[phase]]
        stopped = ended[-1] + 1 if ended else 0
        if mode == "sigterm" and rank in fired:
            return f"SIGTERM in phase {stopped}, as planned"
        code = rank_exit.get(rank)
        exit_code = f" (worker exit code {code})" if code is not None else ""
        if stopped > LAST_PHASE:
            return f"finished all phases, then exited before 'done'{exit_code}"
        return f"died in phase {stopped}{exit_code}"

    def xids_by_gpu(host: str) -> dict:
        gpu_of = {
            pci_key(info["pci"]): gpu
            for health in (health_before[host], health_after[host])
            for gpu, info in health.items()
            if info["pci"] != "unknown"
        }
        by_gpu: dict = {}
        for (pci, xid), count in sorted(dmesg_xids[host].items()):
            by_gpu.setdefault(gpu_of.get(pci, f"PCI {pci}"), {})[xid] = count
        return by_gpu

    armed_ok = set(armed) == fault_ranks and all(
        fault == mode and phase == fault_phase for fault, _, _, phase in armed.values()
    )
    checks: dict[str, bool] = {
        f"master holds global ranks 0-{ranks_per_node - 1}": assigned[MASTER] == survivors,
        f"peer holds global ranks {min(peer_ranks)}-{max(peer_ranks)}": assigned[PEER] == peer_ranks,
        f"{mode} fault armed on exactly ranks {sorted(fault_ranks)}": armed_ok,
        f"fault fired in phase {fault_phase} on ranks {sorted(fault_ranks)}": fired == fault_ranks,
    }

    if timing == "before_all_send":
        checks[f"master ranks {sorted(survivors)} launched their send part after the injector"] = all(
            f"[rank {rank}] released; launching the send part" in logs[MASTER] for rank in survivors
        )
    if timing == "before_send_masked":
        neighbours = sorted(peer_ranks - fault_ranks)
        checks[f"neighbours {neighbours} masked the victim, then launched their send part after the injector"] = all(
            f"[rank {rank}] masked rank" in logs[PEER] and f"[rank {rank}] released; launching the send part" in logs[PEER]
            for rank in neighbours
        )
    if timing in ("before_send", "after_send", "during_send", "before_all_send"):
        neighbours = sorted(peer_ranks - fault_ranks)
        if timing in ("before_send", "during_send", "before_all_send") and not dry_run:
            # neighbours that die inside the send part never time it or receive
            checks[f"neighbours {neighbours} launched their send part after the injector"] = all(
                f"[rank {rank}] released; launching the send part" in logs[PEER] for rank in neighbours
            )
        else:
            checks[f"neighbours {neighbours} timed their send part and received after the injector"] = all(
                f"[rank {rank}] send part:" in logs[PEER] and f"[rank {rank}] launching the receive part" in logs[PEER]
                for rank in neighbours
            )
    elif timing == "after_victim_send":
        (victim_rank,) = fault_ranks
        checks[f"victim rank {victim_rank} finished its own send part before the injector"] = (
            f"[rank {victim_rank}] own send part done" in logs[PEER]
        )
    elif timing == "after_recv":
        waiting = sorted(peer_ranks)
        checks[f"ranks {waiting} finished the receive part before the injector"] = all(
            f"[rank {rank}] receive part done" in logs[PEER] for rank in waiting
        )
    checks["no rank gave up waiting in the fault handshake"] = "gave up waiting" not in combined

    injection_ok = True
    victim = None
    if dry_run:
        (victim_rank,) = fault_ranks
        if timing == "before_send_masked":
            # the neighbours mask the live victim, so it and possibly other ranks then time out on it
            failures_ok = all(detected.get(r) == fault_ranks for r in peer_ranks - fault_ranks) and not any(
                members & survivors for members in detected.values()
            )
            failures_check = f"neighbours reported only rank {victim_rank}; nobody reported a master rank"
        else:
            failures_ok, failures_check = not detected, "no rank reported a failure"
        checks.update(
            {
                f"victim rank {victim_rank} ran the injector probe, which returned NV_OK": (
                    f"[rank {victim_rank}] DRY RUN" in logs[PEER] and "status=0x0000 NV_OK" in logs[PEER]
                ),
                f"all ranks 0-{2 * ranks_per_node - 1} finished all phases": survivors | peer_ranks <= done,
                failures_check: failures_ok,
                "no NVLink errors or Xids on either node": not any(
                    errors[host]["uncorrectable NVLink"] or dmesg_xids[host] for host in (MASTER, PEER)
                ),
                "both EP parents exited 0": ep_rcs[MASTER] == 0 and ep_rcs[PEER] == 0,
            }
        )
    elif mode == "nvlink":
        (victim_rank,) = fault_ranks
        target = dict(
            item.split("=", 1) for item in read_text(results / "fault_target.txt").split() if "=" in item
        )
        victim = armed.get(victim_rank)
        statuses = re.findall(
            r"SET_HW_ERROR_INJECT -> ioctl=(-?\d+) status=0x([0-9a-fA-F]+) (\S+)\s*$", logs[PEER], re.M
        )
        exit_codes = [int(code) for code in re.findall(r"\[rank \d+\] injector exit code (-?\d+)", logs[PEER])]
        ioctl_ms = re.findall(r"ioctl took ([\d.]+) ms", logs[PEER])
        victim_gpu = victim[1] if victim else None
        after = health_after[PEER].get(victim_gpu, {}) if victim_gpu is not None else {}
        injection_checks = {
            f"victim rank {victim_rank} armed GPU {target.get('gpu')} ({target.get('uuid')})": victim is not None
            and str(victim[1]) == target.get("gpu")
            and victim[2] == target.get("uuid"),
            f"victim rank {victim_rank} started the injector": f"[rank {victim_rank}] forcing NVLink link" in logs[PEER],
            "injector returned NV_OK for the forced link-down": len(statuses) == 1
            and statuses[0][:2] == ("0", "0000")
            and all(code == 0 for code in exit_codes),
            f"victim GPU lost NVLink links (after: {after.get('links')} of {NVLINKS})": after.get("links", NVLINKS)
            < NVLINKS,
        }
        checks.update(injection_checks)
        injection_ok = armed_ok and fired == fault_ranks and all(injection_checks.values())

    survival_checks = {}
    if not dry_run:
        checks[f"faulted ranks {sorted(fault_ranks)} did not finish phase {fault_phase}"] = not (
            fault_ranks & phase_ends[fault_phase]
        )
        survival_checks = {
            f"master ranks 0-{ranks_per_node - 1} finished all phases": survivors <= done
            and survivors <= phase_ends[LAST_PHASE],
            f"each master rank detected the lost peer ranks {sorted(lost)}": bool(lost)
            and all(lost <= detected.get(rank, set()) for rank in survivors),
            "master ranks reported no failure of each other": not any(
                detected.get(rank, set()) & survivors for rank in survivors
            ),
            f"{MASTER} EP parent exited 0": ep_rcs[MASTER] == 0,
        }
        checks.update(survival_checks)
        if mode == "sigterm":
            checks["no NVLink errors or Xids on either node"] = not any(
                errors[host]["uncorrectable NVLink"] or dmesg_xids[host] for host in (MASTER, PEER)
            )
            checks[f"{PEER} EP parent exited 0 (SIGTERM exits are expected)"] = ep_rcs[PEER] == 0
        else:
            master_after = health_after[MASTER]
            checks[f"{MASTER} GPUs kept {NVLINKS} links and Recovery None"] = len(master_after) == 8 and all(
                gpu["links"] == NVLINKS and gpu["recovery"] == "None" for gpu in master_after.values()
            )
            checks[f"no Xid on {MASTER}"] = not dmesg_xids[MASTER]

    kind = f"{mode}, timing {timing}" + (", DRY RUN with the injector probe" if dry_run else "")
    print(f"=== two-node NIXL EP fault run ({kind}, ranks {sorted(fault_ranks)}, phase {fault_phase}) ===")
    for host in (MASTER, PEER):
        print(f"{host}: EP rc={ep_rcs[host]}, {errors[host]}")
        for rank in sorted(assigned[host]):
            tag = ""
            if rank in fault_ranks and dry_run:
                tag = " [injector probe only]"
            elif rank in fault_ranks:
                tag = " [NVLink fault injected on this GPU]" if mode == "nvlink" else " [SIGTERM]"
            print(f"  rank {rank} on GPU {rank_gpu.get(rank, '?')}{tag}: {outcome(rank)}")
        for failure in worker_failures[host]:
            print(f"  worker failures: {failure}")
        if dmesg_xids[host]:
            print(f"  Xid during the run, by GPU: {xids_by_gpu(host)}")
    print(f"armed: {armed}")
    print(f"fired: {sorted(fired)}")
    print(f"phase-end ranks: {[sorted(phase_ends[phase]) for phase in range(LAST_PHASE + 1)]}")
    print(f"done ranks: {sorted(done)}")
    print(f"detected failures: { {rank: sorted(members) for rank, members in sorted(detected.items())} }")
    print(f"peer ranks lost in phase {fault_phase}: {sorted(lost)}; peer ranks that finished it: {sorted(peer_finished)}")
    for phase in range(LAST_PHASE + 1):
        values = [bandwidth[(rank, phase)] for rank in sorted(survivors) if (rank, phase) in bandwidth]
        if values:
            print(f"master-rank dispatch+combine bandwidth, phase {phase}: {values} GB/s")
    if mode == "nvlink" and not dry_run:
        print(f"injector: {statuses}, exit codes {exit_codes}, ioctl ms {ioctl_ms}")
        if victim:
            gpu = victim[1]
            print(
                f"victim GPU {gpu} health: before {health_before[PEER].get(gpu)}, after {health_after[PEER].get(gpu)}"
            )
        degraded = {
            host: {gpu: info for gpu, info in health_after[host].items()
                   if info["links"] != NVLINKS or info["recovery"] != "None"}
            for host in (MASTER, PEER)
        }
        print(f"degraded GPUs after the run: {degraded}")
        print(
            f"side effect on {PEER}: injected only on GPU {victim_gpu}; other GPUs whose rank died: "
            f"{gpus_of(lost - fault_ranks)}; GPUs whose rank survived: {gpus_of(peer_finished)}"
        )
    print()
    for description, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {description}")
    print()

    if dry_run:
        if all(checks.values()):
            print(f"RESULT: DRY RUN PASSED - timing {timing} ran end to end; no link was touched")
            return 0
        print("RESULT: DRY RUN FAILED - see the FAIL lines above")
        return 1
    survived = all(survival_checks.values())
    if mode == "sigterm":
        passed = all(checks.values())
        if passed:
            print(
                f"RESULT: CONTROL PASSED - ranks 0-{ranks_per_node - 1} on {MASTER} detected the loss of "
                f"ranks {sorted(lost)} on {PEER} and finished all phases"
            )
        else:
            print("RESULT: CONTROL FAILED - see the FAIL lines above")
        return 0 if passed else 1

    if not injection_ok:
        print("RESULT: INJECTION NOT CONFIRMED - see the FAIL lines; this run says nothing about NVLink faults")
        return 1
    if not survived:
        print(f"RESULT: HYPOTHESIS REJECTED - the fault on {PEER} also broke {MASTER}; see the FAIL lines above")
        return 1
    # Neighbours that send to the downed GPU after the drop die with it; neighbours
    # that finished sending to it and then wait for its data time out and survive.
    # In after_victim_send and after_recv the victim's data has arrived, nothing masks it,
    # and the neighbours' next send to it kills them.
    predicted = {"nvlink": peer_ranks, "before_send": peer_ranks, "during_send": peer_ranks,
                 "before_all_send": peer_ranks, "before_send_masked": fault_ranks, "rdma": fault_ranks,
                 "after_send": fault_ranks, "after_victim_send": peer_ranks, "after_recv": peer_ranks}
    if timing not in predicted:
        verdict = f"OBSERVED (timing {timing} has no prediction)"
    elif lost == predicted[timing]:
        verdict = f"AS PREDICTED for timing {timing}"
    else:
        verdict = f"NOT AS PREDICTED for timing {timing} (expected GPUs {gpus_of(predicted[timing])} to die)"
    print(
        f"RESULT: {verdict} - the NVLink fault injected only on {PEER} GPU {victim_gpu} took down the ranks on "
        f"GPUs {gpus_of(lost)} there, the ranks on GPUs {gpus_of(peer_finished)} survived, and ranks "
        f"0-{ranks_per_node - 1} on {MASTER} detected the loss and finished all phases"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[2] not in {"sigterm", "nvlink"}:
        raise SystemExit(
            f"usage: {sys.argv[0]} RESULTS_DIR {{sigterm|nvlink}} RANKS_PER_NODE FAULT_RANKS FAULT_PHASE"
        )
    selected = {int(rank) for rank in sys.argv[4].split(",")}
    if sys.argv[2] == "nvlink" and len(selected) != 1:
        raise SystemExit("nvlink mode takes exactly one fault rank")
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), selected, int(sys.argv[5])))
