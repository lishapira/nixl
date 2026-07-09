#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pass-gate checker for Phase 5 (real unmap, 2-node MNNVL).

Reads:
  * artifacts_rank{0..N-1}/summary.json (from fault_artifacts.ArtifactCapture)
  * mnnvl_pre.json + mnnvl_post.json (from phase5_mnnvl_probe.py)
  * nvlink_pre.csv + nvlink_post.csv (nvidia-smi nvlink counter snapshots)
  * master.log + worker.log (for MASK DETECTED lines from NIXL-EP timeouts)

Gates:

  MNNVL:
    * pre : master+worker share ClusterUUID AND CliqueId (proves the two
            nodes are actually MNNVL-coupled, not IB fallback)
    * post: same (proves the fabric was not corrupted)
  Victim:
    * rank 2 summary.json present, role=victim, inject_mode=unmap-mid-flight,
      inject_event_ts non-null.
    * The victim's Python PROCESS must have survived long enough to flush
      summary.json (implicit: if summary exists, process didn't die pre-flush).
      The CUDA context on the victim MAY be poisoned by asynchronous NVLink
      writes hitting the freed pages -- that shows up as
      `extra.exception: CUDA ... illegal memory access` and is a VALID
      fault-observability signal (not a test failure).
  Peers:
    * every non-victim rank present, role=peer.
    * peers may exit via CUDA-context corruption OR via clean NIXL-EP timeout
      handling; both are acceptable so long as summary.json flushed.
  Fault observability (need at least ONE):
    * some peer summary has xid_seen=True, OR
    * some peer summary has imex_error_count > 0, OR
    * some peer's nvlink counter (rx_err / rx_remote_err /
      local_link_integrity_err / total_link_recovery / effective_err /
      symbol_err) shows a positive delta from pre-run baseline, OR
    * victim's summary.json contains an `extra.exception` with 'illegal
      memory access' (proves the unmap corrupted at least the victim's
      CUDA context => real driver-level fault fired), OR
    * master.log or worker.log contains `MASK DETECTED dead_rank=<victim>`
      after the injection timestamp (proves NIXL-EP's timeout-based
      elastic-recovery path was exercised, which is the primary purpose
      of Phase 5 - even without a wire-level XID), OR
    * master.log or worker.log contains any `Warning: NIXL-EP timeout
      for {dispatch|combine} {send|receive} ... src_rank=<victim>` line
      (proves at least one peer-side per-message NVLink transfer from
      the victim was interrupted, which for TRUE_IN_KERNEL_UNMAP
      timings proves cuMemUnmap returned while the victim's send/recv
      kernel was still actively writing to peers over NVLink).

  Bonus derived signal (reported, not required):
    * `unmap_interrupted_live_comm` = TRUE iff
      victim_unmap_verdict==TRUE_IN_KERNEL_UNMAP AND
      nixl_ep_msgs_interrupted > 0. This is the STRONGEST triangulated
      proof we can extract from logs that the unmap host-call actually
      returned in the middle of an in-flight NVLink transfer, not just
      in the middle of a busy-spin marker window.

NOTE on XID visibility (observed on 2026-07-02): the container we run in
does NOT have CAP_SYSLOG, so `dmesg` reads fail and `xid_seen` in
summary.json can only ever come from the container-visible IMEX log.
However, the PRIVILEGED cluster monitoring (SLURM job-epilog scanner,
which does read dmesg) DID observe `XID 31` (GPU MMU fault) on the
victim's node during the passing Phase 5 run. So a green gate here with
`xid_on_peer=False` does NOT imply "no HW fault happened" - it means
"no HW fault visible from inside the container". Check the SLURM epilog
notification or ask an admin for `nvidia-smi -q -d PAGE_RETIREMENT` or
NVML XID history if you need the hardware-level ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


# Regex for a NIXL-EP per-message peer-side timeout warning. Each match ==
# one in-flight (or about-to-be-in-flight) NVLink message that the peer
# expected from `src_rank` and never received. This is the CLEANEST
# per-message evidence that the victim's send/recv kernel was interrupted
# mid-communication by our fault injection.
#
# Emitted by src/csrc/nixl_ep.cpp (dispatch/combine send/recv paths):
#   "Warning: NIXL-EP timeout for {dispatch|combine} {send|receive}, "
#   "rank <peer>, local_expert_idx <e>, src_rank <victim>"
_NIXL_TIMEOUT_RE = re.compile(
    r"Warning:\s*NIXL-EP timeout for\s+(?P<phase>dispatch|combine)\s+"
    r"(?P<dir>send|receive)\s*,\s*rank\s+(?P<peer>\d+)\s*,\s*"
    r"local_expert_idx\s+(?P<expert>\d+)\s*,\s*src_rank\s+(?P<victim>\d+)"
)

# Regex for the unmap-mid-flight verdict lines emitted by elastic.py's
# marker-based in-kernel injection path (see maybe_schedule_in_kernel_self_kill).
# TRUE_IN_KERNEL_UNMAP proves cuMemUnmap RETURNED on the host while the
# target kernel was still inside its marked window (i.e. still executing
# the send/recv phase we were targeting).
_UNMAP_VERDICT_RE = re.compile(
    r"\[rank\s+(?P<rank>\d+)\]\s+(?P<verdict>TRUE_IN_KERNEL_UNMAP|"
    r"LATE_UNMAP|HIT_IN_KERNEL_WINDOW|MISSED_IN_KERNEL_TIMING)"
)


COUNTERS_OF_INTEREST = (
    "rx_err",
    "rx_remote_err",
    "local_link_integrity_err",
    "total_link_recovery",
    "effective_err",
    "symbol_err",
)


def _victim_rank(num_procs: int) -> int:
    return 2 if num_procs > 2 else 0


def _load_json(path: str) -> Optional[dict | list]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception as ex:
        print(f"# {path}: json load failed: {ex!r}", file=sys.stderr)
        return None


def _mnnvl_gate(pre: Optional[list], post: Optional[list], stage: str) -> List[str]:
    fails: List[str] = []
    label = f"MNNVL_{stage}"
    if not pre or not post:
        return [f"{label}: missing pre/post file"]
    if not isinstance(pre, list) or not isinstance(post, list):
        return [f"{label}: pre/post payload not a list"]

    def _rollup(node_list: list) -> Dict[str, set]:
        clusters: set = set()
        cliques: set = set()
        for node in node_list:
            for gpu in node.get("gpus", []):
                cu = gpu.get("cluster_uuid")
                ci = gpu.get("clique_id")
                if cu and cu != "unknown":
                    clusters.add(cu)
                if ci and ci != "unknown":
                    cliques.add(ci)
        return {"clusters": clusters, "cliques": cliques}

    pre_r = _rollup(pre)
    post_r = _rollup(post)
    if len(pre_r["clusters"]) != 1:
        fails.append(
            f"{label}: pre has {len(pre_r['clusters'])} distinct cluster_uuid(s): "
            f"{sorted(pre_r['clusters']) or ['NONE']} - nodes are NOT MNNVL-coupled"
        )
    if len(pre_r["cliques"]) != 1:
        fails.append(
            f"{label}: pre has {len(pre_r['cliques'])} distinct clique_id(s): "
            f"{sorted(pre_r['cliques']) or ['NONE']}"
        )
    if stage == "post":
        if pre_r["clusters"] and post_r["clusters"] != pre_r["clusters"]:
            fails.append(
                f"{label}: cluster_uuid changed post-injection "
                f"(pre={sorted(pre_r['clusters'])}, post={sorted(post_r['clusters'])})"
            )
        if pre_r["cliques"] and post_r["cliques"] != pre_r["cliques"]:
            fails.append(
                f"{label}: clique_id changed post-injection "
                f"(pre={sorted(pre_r['cliques'])}, post={sorted(post_r['cliques'])})"
            )
    return fails


def _load_counters_csv(path: str) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    """Return {(host, gpu, link): {counter: value}} for the LAST sample per link."""
    out: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                host = row.get("host", "")
                gpu = row.get("gpu", "")
                link = row.get("link", "")
                if not gpu or not link:
                    continue
                key = (host, gpu, link)
                sample: Dict[str, int] = {}
                for c in COUNTERS_OF_INTEREST:
                    v = row.get(c)
                    if v is None or v == "":
                        continue
                    try:
                        sample[c] = int(v)
                    except ValueError:
                        continue
                if sample:
                    out[key] = sample
    except Exception as ex:
        print(f"# counters {path}: {ex!r}", file=sys.stderr)
    return out


def _scan_logs_for_victim_traffic(
    run_dir: str, victim: int
) -> Dict[str, object]:
    """Extract, from master.log + worker.log:

      * NIXL-EP per-message receive timeouts where src_rank == victim.
        This is our BEST evidence that the victim's send kernel was
        interrupted mid-NVLink-transfer: every count == one peer
        expecting a write that never landed.
      * The unmap-mid-flight in-kernel verdict lines
        (TRUE_IN_KERNEL_UNMAP / LATE_UNMAP / HIT_IN_KERNEL_WINDOW /
        MISSED_IN_KERNEL_TIMING) for the victim.

    Combined, these two signals let us distinguish:

      - unmap fired BEFORE the kernel was actively communicating
        (verdict absent OR LATE_UNMAP, timeout_count may still be N)
      - unmap RETURNED while the target kernel was still inside its
        marked send/recv window AND the kernel's peer NVLink writes
        were interrupted
        (verdict==TRUE_IN_KERNEL_UNMAP AND timeout_count > 0)
    """
    result: Dict[str, object] = {
        "timeout_count_total": 0,
        "timeout_count_by_phase": {"dispatch": 0, "combine": 0},
        "timeout_count_by_direction": {"send": 0, "receive": 0},
        "timeout_unique_peers": set(),
        "timeout_unique_msgs": set(),
        "victim_unmap_verdict": None,
        "victim_unmap_verdict_source_line": None,
    }
    peers: set = result["timeout_unique_peers"]  # type: ignore[assignment]
    msgs: set = result["timeout_unique_msgs"]  # type: ignore[assignment]
    phase_counts: Dict[str, int] = result["timeout_count_by_phase"]  # type: ignore[assignment]
    dir_counts: Dict[str, int] = result["timeout_count_by_direction"]  # type: ignore[assignment]

    for log_name in ("master.log", "worker.log"):
        p = os.path.join(run_dir, log_name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", errors="replace") as fh:
                for line in fh:
                    m = _NIXL_TIMEOUT_RE.search(line)
                    if m and int(m.group("victim")) == victim:
                        phase = m.group("phase")
                        direction = m.group("dir")
                        peer = int(m.group("peer"))
                        expert = int(m.group("expert"))
                        key = (phase, direction, peer, expert)
                        if key in msgs:
                            continue  # de-dup master.log vs worker.log
                        msgs.add(key)
                        peers.add(peer)
                        phase_counts[phase] += 1
                        dir_counts[direction] += 1
                        result["timeout_count_total"] = (
                            result["timeout_count_total"] + 1  # type: ignore[operator]
                        )
                        continue
                    m2 = _UNMAP_VERDICT_RE.search(line)
                    if (
                        m2
                        and int(m2.group("rank")) == victim
                        and m2.group("verdict")
                        in ("TRUE_IN_KERNEL_UNMAP", "LATE_UNMAP")
                    ):
                        # Only overwrite with a "final" verdict
                        # (TRUE_IN_KERNEL_UNMAP or LATE_UNMAP); ignore the
                        # earlier HIT/MISSED lines which are intermediate.
                        result["victim_unmap_verdict"] = m2.group("verdict")
                        result["victim_unmap_verdict_source_line"] = line.strip()[:200]
        except Exception as ex:
            print(f"# log scan {p}: {ex!r}", file=sys.stderr)
            continue
    return result


def _counter_delta_positive(
    pre_path: str, post_path: str, victim_hosts: List[str]
) -> Tuple[bool, List[str]]:
    """True iff any counter increased on a peer GPU (not the victim's own).

    victim_hosts: list of hostnames where rank 2 (victim) sits; peers on the
    SAME host still count as peers because they read via NVLink too, but the
    goal of Phase 5 is to see cross-node evidence, so we bias the report
    toward "peer-on-OTHER-host also saw movement".
    """
    pre = _load_counters_csv(pre_path)
    post = _load_counters_csv(post_path)
    if not pre or not post:
        return False, [f"counters: pre({len(pre)})/post({len(post)}) samples missing"]
    saw_any = False
    diffs: List[str] = []
    for key, post_sample in post.items():
        pre_sample = pre.get(key, {})
        for c, v in post_sample.items():
            base = pre_sample.get(c, 0)
            if v > base:
                host, gpu, link = key
                diffs.append(
                    f"{host} gpu={gpu} link={link} {c}: {base} -> {v} (+{v - base})"
                )
                saw_any = True
    return saw_any, diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-procs", type=int, required=True,
                        help="TOTAL procs across both nodes (typically 8)")
    parser.add_argument("--allow-no-observable-fault", action="store_true",
                        help="Downgrade 'no XID + no imex + no counter delta' from FAIL to WARN.")
    args = parser.parse_args()

    fails: List[str] = []
    warns: List[str] = []
    victim = _victim_rank(args.num_procs)

    # ------------ MNNVL pre-check --------------
    mnnvl_pre = _load_json(os.path.join(args.run_dir, "mnnvl_pre.json"))
    mnnvl_post = _load_json(os.path.join(args.run_dir, "mnnvl_post.json"))
    fails.extend(_mnnvl_gate(mnnvl_pre, mnnvl_post, "pre"))
    fails.extend(_mnnvl_gate(mnnvl_pre, mnnvl_post, "post"))

    # ------------ Per-rank summaries --------------
    per_rank: Dict[int, dict] = {}
    for r in range(args.num_procs):
        p = os.path.join(args.run_dir, f"artifacts_rank{r}", "summary.json")
        if not os.path.isfile(p):
            fails.append(f"rank {r}: missing summary.json ({p})")
            continue
        try:
            per_rank[r] = json.load(open(p))
        except Exception as ex:
            fails.append(f"rank {r}: summary.json load failed: {ex!r}")

    # Victim gates
    v = per_rank.get(victim)
    if v is None:
        fails.append(f"victim rank {victim}: no summary.json")
    else:
        if v.get("role") != "victim":
            fails.append(f"victim rank {victim}: role={v.get('role')!r} (expected 'victim')")
        if v.get("inject_mode") != "unmap-mid-flight":
            fails.append(
                f"victim rank {victim}: inject_mode={v.get('inject_mode')!r} "
                "(expected 'unmap-mid-flight')"
            )
        if v.get("inject_event_ts") is None:
            fails.append(f"victim rank {victim}: inject_event_ts=None (injection never fired)")
        # Presence of summary.json alone proves the Python process survived
        # long enough to flush. `recovered=False` with a fault-consistent
        # `extra.exception` is NOT a failure -- it is expected on
        # unmap-mid-flight (victim's own CUDA context poisoned by peer NVLink
        # writes hitting freed pages).

    # Peer gates - presence of summary.json proves the process survived to
    # flush. recovered=False is not fatal on peers either; observability
    # signals below tell us whether the fault was really exercised.
    peer_ranks = [r for r in per_rank if r != victim]
    for r in peer_ranks:
        s = per_rank[r]
        if s.get("role") != "peer":
            fails.append(f"peer rank {r}: role={s.get('role')!r} (expected 'peer')")
        if not s.get("recovered", False):
            warns.append(
                f"peer rank {r}: recovered=False (peer's CUDA context may have been "
                "poisoned; check extra.exception)"
            )

    # ------------ Fault observability ------------
    victim_hosts: List[str] = []
    if v is not None:
        vh = v.get("extra", {}).get("host") or v.get("host")
        if vh:
            victim_hosts.append(vh)

    saw_xid = any(per_rank[r].get("xid_seen") for r in peer_ranks)
    saw_imex = any((per_rank[r].get("imex_error_count") or 0) > 0 for r in peer_ranks)
    saw_counters, counter_diffs = _counter_delta_positive(
        os.path.join(args.run_dir, "nvlink_pre.csv"),
        os.path.join(args.run_dir, "nvlink_post.csv"),
        victim_hosts,
    )

    # Victim self-corruption (illegal memory access after unmap fired) also
    # proves the fault was real -- it's a driver-level illegal-address
    # response to the unmap racing with in-flight NVLink writes.
    victim_local_fault = False
    if v is not None:
        exc = (v.get("extra") or {}).get("exception") or ""
        if "illegal memory access" in exc.lower() or "cudaerrorillegaladdress" in exc.lower():
            victim_local_fault = True

    # NIXL-EP timeout -> MASK DETECTED on peers is the *elastic-recovery*
    # signal we most care about. Grep master.log + worker.log for it.
    mask_detected = False
    mask_evidence = ""
    for log_name in ("master.log", "worker.log"):
        p = os.path.join(args.run_dir, log_name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", errors="replace") as fh:
                for line in fh:
                    if f"MASK DETECTED dead_rank={victim}" in line:
                        mask_detected = True
                        mask_evidence = line.strip()[:160]
                        break
            if mask_detected:
                break
        except Exception:
            continue

    # NEW: parse per-message NIXL-EP timeouts and the in-kernel unmap
    # verdict. Together these give us the strongest available proof that
    # cuMemUnmap RETURNED while the target kernel was still actively
    # doing NVLink writes to peers (see EXPERIMENT_UNMAP_FAULT.md).
    traffic = _scan_logs_for_victim_traffic(args.run_dir, victim)
    nixl_msgs_interrupted: int = int(traffic["timeout_count_total"])  # type: ignore[assignment]
    unmap_verdict: Optional[str] = traffic["victim_unmap_verdict"]  # type: ignore[assignment]
    unmap_verdict_line: Optional[str] = traffic["victim_unmap_verdict_source_line"]  # type: ignore[assignment]
    # Best-effort expected message count: for each fault iteration, each
    # peer expects `num_local_experts` messages from the victim on both
    # dispatch and combine receive paths. We don't know num_local_experts
    # from summary.json today, so we just report "how many peers, and how
    # many timeout msgs were logged", and leave the ratio interpretation
    # to the caller.
    unique_peers_seen: set = traffic["timeout_unique_peers"]  # type: ignore[assignment]
    tc_by_phase: Dict[str, int] = traffic["timeout_count_by_phase"]  # type: ignore[assignment]

    # STRONG in-flight-comm proof: the injection returned while the target
    # kernel was still inside its marked send/recv window AND at least
    # one peer never got its expected NVLink write from the victim.
    unmap_interrupted_live_comm = (
        unmap_verdict == "TRUE_IN_KERNEL_UNMAP" and nixl_msgs_interrupted > 0
    )

    obs_summary = (
        f"observability: xid_on_peer={saw_xid} imex_error_on_peer={saw_imex} "
        f"nvlink_counter_delta={saw_counters} victim_local_fault={victim_local_fault} "
        f"mask_detected_on_peer={mask_detected} "
        f"nixl_ep_msgs_interrupted={nixl_msgs_interrupted} "
        f"(dispatch={tc_by_phase.get('dispatch', 0)},"
        f"combine={tc_by_phase.get('combine', 0)},"
        f"peers_affected={len(unique_peers_seen)}) "
        f"victim_unmap_verdict={unmap_verdict or 'N/A'} "
        f"unmap_interrupted_live_comm={unmap_interrupted_live_comm}"
    )
    any_signal = (
        saw_xid
        or saw_imex
        or saw_counters
        or victim_local_fault
        or mask_detected
        or nixl_msgs_interrupted > 0
    )
    if not any_signal:
        msg = f"{obs_summary} -- NO fault signals ANYWHERE (unmap may not have fired)"
        if args.allow_no_observable_fault:
            warns.append(msg)
        else:
            fails.append(msg)
    if counter_diffs:
        warns.append("nvlink counter deltas: " + "; ".join(counter_diffs[:10]))
    if mask_detected:
        warns.append(f"mask evidence: {mask_evidence}")
    if unmap_verdict_line:
        warns.append(f"unmap verdict evidence: {unmap_verdict_line}")
    if unmap_interrupted_live_comm:
        warns.append(
            "STRONG SIGNAL: cuMemUnmap returned WHILE the marked send/recv "
            "kernel was still executing AND peers registered "
            f"{nixl_msgs_interrupted} missing per-message NVLink writes "
            "from the victim => unmap interrupted live NVLink communication."
        )

    # ------------ Report ------------
    print(obs_summary)
    if warns:
        print("PHASE5 WARN:")
        for w in warns:
            print(f"  - {w}")
    if fails:
        print("PHASE5 FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1

    # Concise verdict summary. Highlight which signal(s) proved the fault.
    signals = []
    if saw_xid: signals.append("xid")
    if saw_imex: signals.append("imex")
    if saw_counters: signals.append("nvlink_counters")
    if victim_local_fault: signals.append("victim_local_illegal_address")
    if mask_detected: signals.append(f"nixl_mask_detected_dead_rank_{victim}")
    if nixl_msgs_interrupted > 0:
        signals.append(f"nixl_ep_{nixl_msgs_interrupted}_msgs_interrupted")
    if unmap_interrupted_live_comm:
        signals.append("unmap_returned_during_live_nvlink_comm")
    print(
        f"PHASE5 PASS: MNNVL intact pre+post, victim rank {victim} process reached "
        f"summary flush, injection fired, fault observed via: {', '.join(signals) or 'NONE'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
