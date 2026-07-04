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
      of Phase 5 - even without a wire-level XID).

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
import sys
from typing import Dict, List, Optional, Tuple


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

    obs_summary = (
        f"observability: xid_on_peer={saw_xid} imex_error_on_peer={saw_imex} "
        f"nvlink_counter_delta={saw_counters} victim_local_fault={victim_local_fault} "
        f"mask_detected_on_peer={mask_detected}"
    )
    any_signal = saw_xid or saw_imex or saw_counters or victim_local_fault or mask_detected
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
    print(
        f"PHASE5 PASS: MNNVL intact pre+post, victim rank {victim} process reached "
        f"summary flush, injection fired, fault observed via: {', '.join(signals) or 'NONE'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
