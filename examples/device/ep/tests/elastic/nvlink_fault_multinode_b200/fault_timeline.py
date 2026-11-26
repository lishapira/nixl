#!/usr/bin/env python3
"""Put a two-node fault run on one clock around the fault.

usage: fault_timeline.py RESULTS_DIR

Uses the " [t=<epoch>]" suffixes that patch_fault_2node.py adds to the EP
prints, the injector's wall-clock line, and each node's kernel log, whose raw
times are mapped to the wall clock through the NIXL_EP_CLOCK lines that
run_2node.sh writes. Times are in ms from the start of the link-down ioctl
(or from the SIGTERM trigger). Each node's times come from its own clock.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from summarize_fault import MASTER, PEER, pci_key, read_health, read_text, run_config, run_gpus, worker_ranks

STAMPED = re.compile(r"^(.*?)\s*\[t=(\d+\.\d+)\]\s*$")
START = re.compile(r"global_rank=(\d+), local_rank=\d+ -> start phase (\d+)")
OP = re.compile(r"\[rank (\d+)\] (dispatch \d+\.\d+|combine \d+\.\d+|benchmark)$")
EVENT = re.compile(
    r"start phase|end phase|-> done|detected rank failures|Killing rank|forcing NVLink|DRY RUN|"
    r"injector exit code|Worker processes failed|holding the first dispatch|injector returned|"
    r"launched the first dispatch|gave up waiting|holding the send part|launching the send part|"
    r"releasing the neighbours|launching the receive part|own send part done|receive part done|continuing"
)
PART = re.compile(r"\[rank (\d+)\] (send|receive) part: launched ([\d.]+), finished by ([\d.]+), GPU time ([\d.]+) ms")
PART_LAUNCH = re.compile(r"\[rank (\d+)\] (?:released; |injector returned; )?launching the (send|receive) part")
INJECTOR = re.compile(r"wall clock: injector start ([\d.]+), ioctl begin ([\d.]+), ioctl end ([\d.]+)")
CLOCK = re.compile(r"^\[\s*([\d.]+)\] NIXL_EP_CLOCK \w+ wall=([\d.]+)", re.M)
XID = re.compile(r"^\[\s*([\d.]+)\] NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+), (.*)$", re.M)


def ep_lines(host: str, log: str):
    """Stamped EP events, and each rank's dispatch/combine/benchmark starts with their phase."""
    phase: dict[int, int] = {}
    events, ops = [], []
    for line in log.splitlines():
        stamped = STAMPED.match(line)
        if not stamped:
            continue
        message, t = stamped.group(1).strip(), float(stamped.group(2))
        start = START.search(message)
        if start:
            phase[int(start.group(1))] = int(start.group(2))
        op = OP.search(message)
        if op:
            rank = int(op.group(1))
            ops.append((t, rank, phase.get(rank), op.group(2)))
        elif EVENT.search(message):
            events.append((t, f"{host}: {message}"))
    return events, ops


def kernel_lines(host: str, results: Path):
    """Xid lines on the wall clock, interpolated between the NIXL_EP_CLOCK marks."""
    text = read_text(results / f"dmesg_delta_{host}.log")
    marks = [(float(raw), float(wall)) for raw, wall in CLOCK.findall(text)]
    if not marks:
        return None
    (raw0, wall0), (raw1, wall1) = marks[0], marks[-1]
    rate = (wall1 - wall0) / (raw1 - raw0) if raw1 > raw0 else 1.0
    gpu_of = {
        pci_key(info["pci"]): gpu
        for gpu, info in read_health(results / f"gpu_health_before_{host}.txt").items()
        if info["pci"] != "unknown"
    }
    return [
        (wall0 + (float(raw) - raw0) * rate,
         f"{host}: kernel Xid {xid} on GPU {gpu_of.get(pci_key(pci), 'PCI ' + pci)}: {detail[:100]}")
        for raw, pci, xid, detail in XID.findall(text)
    ]


def main(results: Path) -> int:
    config = run_config(results)
    phase = int(config.get("fault_phase", "1"))
    # single-node runs (run_1node.sh) have only the peer's files
    logs = {host: read_text(results / f"ep_{host}.log") for host in (MASTER, PEER)
            if (results / f"ep_{host}.log").exists()}
    logs.setdefault(PEER, "")
    gpus = run_gpus(results)
    rank_where = {
        rank: (host, gpus[local % len(gpus)] if gpus else "?")
        for host, log in logs.items()
        for rank, local in worker_ranks(log).values()
    }
    events, ops = [], []
    for host, log in logs.items():
        host_events, host_ops = ep_lines(host, log)
        events += host_events
        ops += host_ops
    if not events:
        print("no [t=...] stamps in the EP logs; this run predates the timeline prints")
        return 0

    injector = INJECTOR.search(logs[PEER])
    if injector:
        start, t0, ioctl_end = (float(value) for value in injector.groups())
        events += [(start, f"{PEER}: injector process started"),
                   (t0, f"{PEER}: injector ioctl begins"),
                   (ioctl_end, f"{PEER}: injector ioctl returns")]
    else:
        triggers = [t for t, message in events if "Killing rank" in message]
        t0 = min(triggers) if triggers else min(t for t, _ in events)
        ioctl_end = None

    for host in logs:
        lines = kernel_lines(host, results)
        if lines is None:
            print(f"note: no NIXL_EP_CLOCK lines in the {host} kernel log; its Xid lines are left out")
        else:
            events += lines
    peer_xids = [(t, message) for t, message in events if message.startswith(f"{PEER}: kernel Xid") and t >= t0 - 1]
    link_down = [t for t, message in peer_xids if "kernel Xid 149 " in message]
    if link_down:
        drop, drop_source = min(link_down), f"{PEER} link-down Xid 149"
    elif peer_xids:
        drop, drop_source = min(t for t, _ in peer_xids), f"first {PEER} kernel Xid"
    elif ioctl_end is not None:
        drop, drop_source = ioctl_end, "ioctl return (no kernel times)"
    else:
        drop, drop_source = t0, "fault trigger"

    def ms(t: float) -> str:
        return f"{(t - t0) * 1e3:+.1f} ms"

    kind = f"fault {config.get('fault', '?')}, timing {config.get('fault_timing', 'timer')}"
    if config.get("fault_dry_run") == "1":
        kind += ", DRY RUN (probe only, no link drop)"
    print(f"=== timeline: {kind}; ms from the start of the {'injector ioctl' if injector else 'fault trigger'} ===")
    print(f"link drop taken as the {drop_source}: {ms(drop)}")
    print(f"\neach rank at the drop (phase {phase} dispatch/combine/benchmark starts):")
    for rank in sorted(rank_where):
        host, gpu = rank_where[rank]
        starts = [(t, op) for t, r, p, op in ops if r == rank and p == phase]
        before = [item for item in starts if item[0] <= drop]
        after = [item for item in starts if item[0] > drop]
        state = f"in {before[-1][1]} since {ms(before[-1][0])}" if before else "no op started yet"
        if after:
            state += (f"; {len(after)} more op(s), first {after[0][1]} at {ms(after[0][0])}, "
                      f"last {after[-1][1]} at {ms(after[-1][0])}")
        else:
            state += "; no op started after the drop"
        print(f"  rank {rank} ({host} GPU {gpu}): {state}")

    all_logs = "\n".join(logs.values())
    parts = [match.groups() for match in PART.finditer(all_logs)]
    send_parts = [(rank, *rest) for rank, part, *rest in parts if part == "send"]
    if send_parts:
        print("\nsend parts (the GPU was idle at launch, so they end about launch + GPU time):")
    for rank, launched, finished, gpu_ms in send_parts:
        launched, finished, gpu_ms = float(launched), float(finished), float(gpu_ms)
        end = launched + gpu_ms / 1e3
        where = "after the drop" if launched > drop else "before the drop" if end < drop else "SPANS the drop"
        print(f"  rank {rank}: {ms(launched)} to {ms(end)} (GPU {gpu_ms:.3f} ms, host saw it done at "
              f"{ms(finished)}) - {where}")
    recv_parts = [(rank, *rest) for rank, part, *rest in parts if part == "receive"]
    if recv_parts:
        print("\nreceive parts (it waits for the senders' data, so only the host-seen end is meaningful):")
    for rank, launched, finished, gpu_ms in recv_parts:
        where = "before the drop" if float(finished) < drop else "after the drop"
        print(f"  rank {rank}: launched {ms(float(launched))}, host saw it done at {ms(float(finished))} "
              f"(GPU {float(gpu_ms):.3f} ms) - done {where}")
    # the older split timings time only the neighbours' send part
    free = config.get("fault_timing") in ("after_victim_send", "after_recv")
    victims = set(config.get("fault_ranks", "").split(","))
    reported = {(rank, part) for rank, part, *_ in parts}
    for line in all_logs.splitlines():
        stamped, launch = STAMPED.match(line), PART_LAUNCH.search(line)
        timed = launch and (free or (launch.group(2) == "send" and launch.group(1) not in victims))
        if stamped and timed and launch.groups() not in reported:
            print(f"  rank {launch.group(1)}: {launch.group(2)} part launched at {ms(float(stamped.group(2)))}, "
                  "never reported finished")

    print("\nevents:")
    for t, message in sorted(events):
        print(f"  {ms(t):>14}  {message}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULTS_DIR")
    raise SystemExit(main(Path(sys.argv[1])))
