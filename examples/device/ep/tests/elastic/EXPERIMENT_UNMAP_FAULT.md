# NIXL EP: NVLink Unmap-Mid-Flight Fault Injection Test

Branch: `nvlink_fault_tolerance` (base `8736bb4`). Local-only; no push, no merge.

Companion branch `nvlink_fault_tolerance_all_phases` preserves the full
10-commit development history of the harness (Phase 0 tracker, Phase 1
SIGKILL-sweep + artifact-capture smoke, Phase 2 rebuild + pybind smoke,
Phase 3 detached-buffer dry-run, Phase 5 real 2-node MNNVL unmap, plus
follow-up fixes). This branch collapses to the single production test.

## Goal

Produce a real GPU-driver-level fault by tearing down a NIXL-registered,
peer-exposed VMM buffer mid-transfer, and verify that NIXL EP's
elastic-recovery mechanism correctly handles it — victim's Python
process survives, peers detect the failure via NIXL-EP mask timeout,
MNNVL fabric stays intact.

## Methodology

The victim rank calls the CUDA driver's memory-management teardown on
its live peer-exposed RDMA buffer:

    cuMemUnmap(ptr, size)
    cuMemAddressFree(ptr, size)
    cuMemRelease(handle)

Unlike SIGKILL, the victim's Python process **survives** the injection.
Peers still hold P2P mappings to the region at the moment of the unmap.
The GPU MMU catches the invalidated mapping and raises the fault at
driver level (visible as `XID 31` in the host kernel ring buffer). For
this experiment, the `XID 31` evidence came from a Slack alert emitted
by the cluster monitoring bot; we did not verify it directly against
`dmesg` (CAP_SYSLOG-blocked inside the container) or a SLURM epilog
scanner.

Implementation:

* C++: `Buffer::inject_unmap_fault()` in `nixl_ep.cpp` resets the
  `std::unique_ptr<vmm_region>`, whose destructor calls the three
  CUDA APIs above in the correct order.
* Python: forwarded via `nixl_ep.buffer.Buffer.inject_unmap_fault()`.
* Driver flag on `elastic.py`:

      --fault-inject-mode {sigkill,unmap-mid-flight}

  * `sigkill` (default) — pre-existing SIGKILL path, unchanged.
  * `unmap-mid-flight` — **this test**; tears down the real
    peer-exposed buffer.

  The `unmap-detached` dev-mode variant (Phase 3 harness dry-run) is
  not part of this branch; see the companion branch
  `nvlink_fault_tolerance_all_phases` for that code.

## The test: real 2-node MNNVL unmap

Plan `nvlink_fault_tolerance_2node_unmap.json`: 2 nodes × 4 ranks = 8
ranks total, 2 phases. Victim `rank 2` is present in BOTH phases
(unlike the SIGKILL plan, which drops the victim in phase 1) because
the victim survives at Python level. Injection fires at
`before-dispatch` of phase 1.

**Tested timings**: `elastic.py --fault-kill-timing` accepts 15 values
(7 CPU-level: `before-dispatch`, `after-dispatch`,
`between-dispatch-combine`, `dispatch-between-send-receive`,
`before-combine`, `combine-between-send-receive`, `after-combine`;
plus 8 in-kernel cells covering dispatch/combine × send/recv ×
hook-separated/no-hook). For `unmap-mid-flight` we only exercised
`before-dispatch` end-to-end; the other 14 are exposed via the
`TIMING` env var to `run_phase5_2node.sh` and should work
mechanically (same injection call site) but are unvalidated. The
pre-existing SIGKILL sweep still covers the full timing matrix.

Runner `run_phase5_2node.sh` (fast path, ~2 min end-to-end, tuned to
survive on the preemptible `gb200-backfill` partition where each srun
container cold-start costs ~30 s):

  1. `srun -N2 --ntasks-per-node=1` — parallel per-node PRE probe:
     MNNVL fabric snapshot (ClusterUUID + CliqueId) via
     `phase5_mnnvl_probe.py` + `nvidia-smi nvlink --errorcounters`.
  2. `srun -N1` on master — `elastic.py` with rank server + 4 ranks
     (including victim rank 2).
  3. `srun -N1` on worker — `elastic.py --tcp-server master` (4 peers).
  4. `srun -N2 --ntasks-per-node=1` — parallel per-node POST probe:
     MNNVL + NVLink counters + `torch.cuda` health probe.

Per-rank artifacts (`fault_artifacts.py` + `ArtifactCapture`):

  * `nvlink_counters.csv` — GB200-native 14-column `nvidia-smi nvlink
    --errorcounters` layout.
  * `dmesg.log` — best-effort; empty on Lyris because the container
    lacks `CAP_SYSLOG` and the host has `dmesg_restrict=1`. We record
    `dmesg_readable=false` in the summary so absence is not mistaken
    for silence.
  * `imex.log` — bind-mounted from host `/var/log/nvidia-imex-verbose.log`.
  * `summary.json` — rank, role (`victim`/`peer`), `inject_mode`,
    `inject_event_ts`, `recovered`, `xid_seen`, `imex_error_count`,
    and `extra.exception` if the worker raised.

Pass gates (`phase5_pass_gate.py`):

  * MNNVL PRE : master + worker share ClusterUUID + CliqueId (proves
    the two nodes are actually MNNVL-coupled, not IB-fallback).
  * MNNVL POST: same (proves the fabric was not corrupted).
  * Victim rank 2: `role=victim`, `inject_mode=unmap-mid-flight`,
    `inject_event_ts` populated, `summary.json` present.
  * Peers: `role=peer`, `summary.json` present.
  * **Fault observability** — at least ONE of:
      * `xid_seen=True` on a peer (container-visible dmesg — usually
        unavailable on Lyris),
      * `imex_error_count > 0` on a peer,
      * positive delta on any `nvidia-smi nvlink` counter,
      * victim's `extra.exception` contains `"illegal memory access"`
        (proves the unmap corrupted the victim's own CUDA context),
      * `MASK DETECTED dead_rank=<victim>` in master or worker log
        (proves NIXL-EP's timeout-based elastic-recovery fired — the
        primary signal we care about).

## Run history

* **Attempt #1** (job 2258566, `lyris0168`+`lyris0171`): FAILED at gate.
  Fault mechanics fired correctly (INJECT lines present; rank 2 raised
  `AcceleratorError`; `MASK DETECTED` on all 7 peers), but two
  test-harness bugs surfaced under real fault conditions:
  (a) `ground_truth_dead` was `[]` for `unmap-mid-flight` while peers
  correctly flagged rank 2 dead → `_check_mask_no_false_positives`
  raised; (b) no `try/finally` around the worker loop, so
  `summary.json` never flushed on peer crashes. Both fixed on this
  branch. See the companion branch for the fix commit.

* **Attempt #2** (same job, after fixes): **PASS.**
  All 8 ranks reached `WORKER DONE`, all 8 `summary.json` flushed. All
  7 peers: `MASK CHECK SUMMARY passes=48 calls=48
  ground_truth_dead=[2]`. Victim rank 2 raised `AcceleratorError` from
  its own poisoned CUDA context (`survived=false` at plan-completion
  level, but the process reached the summary flush thanks to the
  `try/finally`). MNNVL ClusterUUID + CliqueId unchanged pre/post.
  Verdict:
  `MNNVL intact pre+post, victim rank 2 process reached summary flush,
  injection fired, fault observed via: victim_local_illegal_address,
  nixl_mask_detected_dead_rank_2`.

* **Post-job cluster monitoring**: a Slack alert from the cluster
  monitoring bot reported `XID 31` (GPU MMU fault) on `lyris0168`
  during the job window. This is hardware-level evidence our
  container-side collectors could NOT see (`dmesg` CAP_SYSLOG-blocked;
  IMEX log stayed clean; `nvidia-smi nvlink` counters unchanged).
  `lyris0171` was NOT named, consistent with a same-node MMU catching
  the invalid mapping before any MNNVL wire transaction escalated.

## Research findings — fault-model taxonomy

Three orthogonal observations of the same underlying event, in
decreasing distance from the wire:

1. **HW MMU fault (XID 31)** — GPU MMU rejects an access to an
   invalidated mapping. Kernel-visible; requires `CAP_SYSLOG` or
   cluster-level monitoring. **CONFIRMED** via a Slack alert from the
   cluster monitoring bot naming `lyris0168` during the job window.
2. **CUDA runtime `illegal_address` (error 700)** — userspace
   projection of the same MMU fault; surfaces as a CUDA warning on
   peers or `torch.AcceleratorError` on the victim. **CONFIRMED** in
   per-rank artifacts (`extra.exception` on rank 2).
3. **NIXL-EP mask detection** (`MASK DETECTED dead_rank=2`) — SW-level
   30 s timeout on the mask-poll loop; triggers elastic cleanup.
   **CONFIRMED** in every peer's log; first detection lands ~30 s
   after `inject_event_ts`, matching `DEFAULT_TIMEOUT_MS = 30_000`.

What we do **NOT** observe: NVLink transport-layer fatal signature
(XID 74 / 79 / 154, contain-and-drain, NVLink counter delta, IMEX
`[ERROR]`, MNNVL clique change).

### Structural reason — why neither `unmap-mid-flight` nor SIGKILL can produce a transport-layer fatal

Both fault-injection paths on this branch (`unmap-mid-flight` and the
pre-existing `sigkill`) are **driver-cooperative teardowns**.

> **Driver-cooperative** means the resource release goes through the
> NVIDIA kernel driver's own release code path, on a CPU host thread,
> synchronously, and the driver runs its cleanup invariants *before*
> the resource is actually gone. This is the opposite of an
> "asynchronous hardware fault" (cable pull, GPU hard-hang, ECC
> exhaustion, BMC port disable), where a hardware event happens
> without the driver getting to run its cleanup logic — those bypass
> the invariants below and are the only class of events that can
> produce a real NVLink transport-layer fatal.

They share one invariant:

> *Peer P2P mappings are invalidated before the underlying resource
> is released.*

* `cuMemUnmap` — the CUDA driver walks every peer that imported the
  mapping, clears its P2P page-table entry via the VMM handle-export
  metadata, and only then frees the physical pages. This ordering is
  guaranteed by the CUDA memory-management contract; it is the whole
  point of `cuMemUnmap` being a distinct call from `cuMemRelease`.
* `SIGKILL` — the OS runs the NVIDIA driver's `.release` file-op on
  the dying process's `/dev/nvidia*` fds. That release path also
  walks and invalidates the peer imports of any exported memory
  belonging to the dying context, then reclaims the physical pages.

Consequence for any peer that next tries to load from the region:

1. The peer's own GPU MMU walks the (now invalidated) P2P PTE.
2. The MMU raises a fault **locally** → `XID 31` in the peer's kernel
   log + `cudaErrorIllegalAddress` returned to userspace.
3. **No P2P transaction is ever emitted onto the NVLink wire.**

Because no transaction leaves the sender, the NVLink transport layer
never has an outstanding request that could time out. Therefore:

* No transport-layer response timeout → no fatal escalation.
* No `contain-and-drain` → no fabric-level context teardown.
* No `XID 74 / 79 / 154`, no NVLink `nvidia-smi` counter delta, no
  IMEX `[ERROR]`, no MNNVL clique change.

This holds equally for SIGKILL and unmap-mid-flight: the two
experiments produce the same set of observable and unobservable
signals on peers, because both hit the same driver-cooperative
teardown path. The only real difference is what happens to the
*victim's* process (SIGKILL: process gone; unmap-mid-flight: process
alive but its CUDA context is poisoned by the same MMU fault the
peers see, surfacing as `torch.AcceleratorError` on the next CUDA
call).

The same limitation applies to every other cooperative userspace
teardown API — `cuIpcCloseMemHandle`, `cudaDeviceReset`,
`cuDevicePrimaryCtxReset`, process `exit()`, `kill -9`, etc. Anything
that goes through the driver's release path invalidates peer PTEs
before releasing memory. This is *by design*: the driver's contract
is to prevent stale P2P mappings, precisely so that a fault gets
caught at the source MMU rather than being allowed to hit the wire
and destabilize the whole fabric.

**To produce a transport-layer fatal event**, the victim GPU must
stop responding to NVLink transactions **without its driver getting
to run the cooperative cleanup path**. Practical ways to do that all
require privileges outside of user-space:

* Physical link disruption — cable pull, NVSwitch port disable via
  MFT / mst / BMC / NMX-M / Redfish.
* Forced GPU reset — `nvidia-smi -r` (needs `sudo`), fabric-manager
  eviction, driver-level GPU-hang recovery.
* Uncorrectable hardware failure — ECC exhausting PLR retries, real
  PHY-level link-down, SM hard-hang from a bug that survives context
  teardown.

None of these are producible from an unprivileged user-space test
without cluster-admin cooperation, which is why our container-scoped
experiment cannot exercise that code path.

### Traffic-ordering context

NIXL EP's hot path uses `ld.acquire.sys.global` /
`st.release.sys.global` / `fence.acq_rel.sys` PTX ops — strictly
ordered `.sys`-scope P2P memory accesses. In the NVLink reliability
taxonomy this is the *ordered* case: source-side retransmission
would violate ordering, so if a real NVLink-fatal event were to
occur, the hardware would necessarily escalate to contain-and-drain
and tear down the affected GPU contexts (rather than silently
retrying at the transport layer).

### What NIXL EP's elastic recovery does and does not handle

NIXL EP's elastic recovery is a **software-only** mechanism. It
targets the *timeout / silent-peer* class of faults — the peer
stops updating its mask entry, the mask-poll loop hits
`DEFAULT_TIMEOUT_MS = 30_000` ms, the survivors declare the peer
dead, and the plan proceeds without it. Both this experiment
(unmap-mid-flight) and the pre-existing SIGKILL sweep exercise
exactly this path.

It **cannot** recover from a **NVLink transport-fatal event** — i.e.,
the class of NVLink hardware errors where the transport-layer
response timeout fires, the fabric contains-and-drains, and the
hardware itself tears down the affected GPU contexts (`XID 74 / 79 /
154`). Once the hardware has torn down contexts at that level, no
in-process software recovery is possible; the only recovery path is
checkpoint-restart of the whole job. This experiment does not — and
by construction cannot — validate any behaviour on that code path.

## Reproducing the test

    # inside an --exclusive SLURM alloc across 2 MNNVL-coupled GB200 nodes:
    SLURM_JOB_ID=<job>  SLURM_JOB_NODELIST=<host1,host2> \
        bash examples/device/ep/tests/elastic/run_phase5_2node.sh

Results land in `examples/device/ep/tests/elastic/results/phase5_2node_<ts>_<hosts>_<timing>/`.
The runner prints `PHASE5 PASS: ...` on success and exits non-zero on
any gate failure.
