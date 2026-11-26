#!/usr/bin/env python3
"""Validate a single-node NIXL EP NVLink fault run from run_1node.sh.

usage: summarize_1node.py RESULTS_DIR
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from summarize_fault import (
    LAST_PHASE,
    NVLINKS,
    bandwidth_by_phase,
    pci_key,
    rank_set,
    read_health,
    read_int,
    read_text,
    run_config,
    run_gpus,
    worker_exit_codes,
    worker_ranks,
    xids,
)


def main(results: Path) -> int:
    config = run_config(results)
    host = config["host"]
    timing = config["fault_timing"]
    dry_run = config["fault_dry_run"] == "1"
    phase = int(config["fault_phase"])
    victim_rank = int(config["fault_ranks"])
    neighbours = {int(rank) for rank in config["fault_neighbours"].split(",")}
    all_ranks = neighbours | {victim_rank}
    log = read_text(results / f"ep_{host}.log")
    ep_rc = read_int(results / f"ep_{host}.rc")

    assigned = rank_set(r"Process \d+ -> global_rank=(\d+), local_rank=\d+", log)
    done = rank_set(r"global_rank=(\d+), local_rank=\d+ -> done", log)
    phase_ends = {
        p: rank_set(rf"global_rank=(\d+), local_rank=\d+ -> end phase {p}\b", log) for p in range(LAST_PHASE + 1)
    }
    detected: dict[int, set[int]] = {}
    for rank, members in re.findall(r"global_rank=(\d+), local_rank=\d+ -> detected rank failures: \{([^}]*)\}", log):
        detected.setdefault(int(rank), set()).update(int(m) for m in re.findall(r"\d+", members))
    armed = {
        int(rank): (int(gpu), uuid, int(p))
        for rank, gpu, uuid, p in re.findall(
            r"\[rank (\d+)\] armed nvlink fault on GPU (\d+) \((GPU-[0-9a-fA-F-]+)\) for phase (\d+)", log
        )
    }
    fired = rank_set(r"\[rank (\d+)\] Killing rank during dispatch/combine", log)
    target = dict(item.split("=", 1) for item in read_text(results / "fault_target.txt").split() if "=" in item)
    health_before = read_health(results / f"gpu_health_before_{host}.txt")
    health_after = read_health(results / f"gpu_health_after_{host}.txt")
    dmesg_xids = xids(results / f"dmesg_delta_{host}.log")
    gpus = run_gpus(results)
    rank_gpu: dict[int, int] = {}
    rank_exit: dict[int, int] = {}
    codes = worker_exit_codes(log)
    for worker, (rank, local) in worker_ranks(log).items():
        rank_gpu[rank] = gpus[local % len(gpus)]
        if worker in codes:
            rank_exit[rank] = codes[worker]
    lost = all_ranks - phase_ends[phase]
    finished = all_ranks & phase_ends[phase]

    def gpus_of(ranks) -> list:
        return [rank_gpu.get(rank) for rank in sorted(ranks)]

    def outcome(rank: int) -> str:
        if rank in done:
            return "finished all phases"
        ended = [p for p in range(LAST_PHASE + 1) if rank in phase_ends[p]]
        stopped = ended[-1] + 1 if ended else 0
        code = rank_exit.get(rank)
        exit_code = f" (worker exit code {code})" if code is not None else ""
        if stopped > LAST_PHASE:
            return f"finished all phases, then exited before 'done'{exit_code}"
        return f"died in phase {stopped}{exit_code}"

    gpu_of_pci = {pci_key(info["pci"]): gpu for gpu, info in health_before.items() if info["pci"] != "unknown"}
    xid_by_gpu: dict = {}
    for (pci, xid), count in sorted(dmesg_xids.items()):
        xid_by_gpu.setdefault(gpu_of_pci.get(pci, f"PCI {pci}"), {})[xid] = count

    checks: dict[str, bool] = {
        f"ranks {sorted(all_ranks)} assigned": assigned == all_ranks,
        f"nvlink fault armed on exactly rank {victim_rank}, GPU {target.get('gpu')} ({target.get('uuid')})": (
            set(armed) == {victim_rank}
            and str(armed[victim_rank][0]) == target.get("gpu")
            and armed[victim_rank][1] == target.get("uuid")
            and armed[victim_rank][2] == phase
        ),
        f"fault fired in phase {phase} on rank {victim_rank}": fired == {victim_rank},
        "no rank gave up waiting in the fault handshake": "gave up waiting" not in log,
    }
    if timing in ("before_send", "during_send") and not dry_run:
        # neighbours that die inside the send part never time it or receive
        checks[f"neighbours {sorted(neighbours)} launched their send part after the injector"] = all(
            f"[rank {rank}] released; launching the send part" in log for rank in neighbours
        )
    elif timing in ("before_send", "after_send", "during_send"):
        checks[f"neighbours {sorted(neighbours)} timed their send part and received after the injector"] = all(
            f"[rank {rank}] send part:" in log and f"[rank {rank}] launching the receive part" in log
            for rank in neighbours
        )
    elif timing == "after_victim_send":
        checks[f"victim rank {victim_rank} finished its own send part before the injector"] = (
            f"[rank {victim_rank}] own send part done" in log
        )
    elif timing == "after_recv":
        checks[f"ranks {sorted(all_ranks)} finished the receive part before the injector"] = all(
            f"[rank {rank}] receive part done" in log for rank in all_ranks
        )

    statuses = re.findall(r"SET_HW_ERROR_INJECT -> ioctl=(-?\d+) status=0x([0-9a-fA-F]+) (\S+)\s*$", log, re.M)
    exit_codes = [int(code) for code in re.findall(r"\[rank \d+\] injector exit code (-?\d+)", log)]
    ioctl_ms = re.findall(r"ioctl took ([\d.]+) ms", log)
    victim_gpu = int(target.get("gpu", -1))
    after = health_after.get(victim_gpu, {})
    if dry_run:
        checks.update(
            {
                f"victim rank {victim_rank} ran the injector probe, which returned NV_OK": (
                    f"[rank {victim_rank}] DRY RUN" in log and "status=0x0000 NV_OK" in log
                ),
                f"all ranks {sorted(all_ranks)} finished all phases": all_ranks <= done,
                "no rank reported a failure": not detected,
                "no Xid during the run": not dmesg_xids,
                "EP parent exited 0": ep_rc == 0,
            }
        )
    else:
        checks.update(
            {
                f"victim rank {victim_rank} started the injector": f"[rank {victim_rank}] forcing NVLink link" in log,
                "injector returned NV_OK for the forced link-down": len(statuses) == 1
                and statuses[0][:2] == ("0", "0000")
                and all(code == 0 for code in exit_codes),
                f"victim GPU lost NVLink links (after: {after.get('links')} of {NVLINKS})": after.get("links", NVLINKS)
                < NVLINKS,
                f"victim rank {victim_rank} did not finish phase {phase}": victim_rank not in phase_ends[phase],
            }
        )
        survivors = finished
        if survivors:
            checks[f"surviving ranks {sorted(survivors)} detected the loss of {sorted(lost)} and finished all phases"] = all(
                lost <= detected.get(rank, set()) and rank in done for rank in survivors
            )

    kind = f"timing {timing}" + (", DRY RUN with the injector probe" if dry_run else "")
    print(f"=== single-node NIXL EP NVLink fault run on {host} ({kind}, victim rank {victim_rank}, phase {phase}) ===")
    print(f"EP rc={ep_rc}")
    for rank in sorted(assigned):
        tag = ""
        if rank == victim_rank:
            tag = " [injector probe only]" if dry_run else " [NVLink fault injected on this GPU]"
        print(f"  rank {rank} on GPU {rank_gpu.get(rank, '?')}{tag}: {outcome(rank)}")
    for failure in re.findall(r"Worker processes failed: (.*?)(?: \[t=[\d.]+\])?$", log, re.M):
        print(f"  worker failures: {failure}")
    if xid_by_gpu:
        print(f"  Xid during the run, by GPU: {xid_by_gpu}")
    print(f"phase-end ranks: {[sorted(phase_ends[p]) for p in range(LAST_PHASE + 1)]}")
    print(f"done ranks: {sorted(done)}")
    print(f"detected failures: { {rank: sorted(members) for rank, members in sorted(detected.items())} }")
    print(f"ranks lost in phase {phase}: {sorted(lost)}; ranks that finished it: {sorted(finished)}")
    bandwidth = bandwidth_by_phase(log)
    for p in range(LAST_PHASE + 1):
        values = {rank: bandwidth[(rank, p)] for rank in sorted(all_ranks) if (rank, p) in bandwidth}
        if values:
            print(f"dispatch+combine bandwidth, phase {p}: {values} GB/s")
    if not dry_run:
        print(f"injector: {statuses}, exit codes {exit_codes}, ioctl ms {ioctl_ms}")
        print(f"victim GPU {victim_gpu} health: before {health_before.get(victim_gpu)}, after {after}")
        degraded = {gpu: info for gpu, info in health_after.items()
                    if info["links"] != NVLINKS or info["recovery"] != "None"}
        print(f"degraded GPUs after the run: {degraded}")
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
    if not all(checks.values()):
        print("RESULT: CHECKS FAILED - see the FAIL lines above")
        return 1
    # Neighbours that send to the downed GPU after the drop die with it; neighbours
    # that finished sending to it and then wait for its data time out and survive.
    predicted = {"nvlink": all_ranks, "before_send": all_ranks, "during_send": all_ranks,
                 "rdma": {victim_rank}, "after_send": {victim_rank},
                 "after_victim_send": all_ranks, "after_recv": all_ranks}
    if timing not in predicted:
        verdict = f"OBSERVED (timing {timing} has no prediction)"
    elif lost == predicted[timing]:
        verdict = f"AS PREDICTED for timing {timing}"
    else:
        verdict = f"NOT AS PREDICTED for timing {timing} (expected GPUs {gpus_of(predicted[timing])} to die)"
    print(f"RESULT: {verdict} - the NVLink fault on GPU {victim_gpu} took down the ranks on GPUs "
          f"{gpus_of(lost)}; the ranks on GPUs {gpus_of(finished)} survived")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULTS_DIR")
    raise SystemExit(main(Path(sys.argv[1])))
