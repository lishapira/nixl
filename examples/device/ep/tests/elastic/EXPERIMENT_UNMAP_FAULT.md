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
hook-separated/no-hook). For `unmap-mid-flight` we validated two
end-to-end:

* `before-dispatch` (Attempt #2) — cuMemUnmap fires between-collectives.
* `dispatch-send-during-kernel-no-hook` (Attempt #5, with the
  injector's `cudaDeviceSynchronize()` removed and
  `IN_KERNEL_SPIN_CYCLES=200000000`) — cuMemUnmap fires while
  rank 2's dispatch kernel is still spinning on the peer-exposed
  buffer (`TRUE_IN_KERNEL_UNMAP` verdict, `exited_after_unmap=0`).

The remaining 13 timings are exposed via the `TIMING` env var to
`run_phase5_2node.sh` and should work mechanically (same injection
call site) but are unvalidated. The pre-existing SIGKILL sweep still
covers the full timing matrix.

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

* **Attempt #3** (job 2274658, `lyris0232`+`lyris0233`,
  `TIMING=dispatch-send-during-kernel-no-hook`,
  `IN_KERNEL_SPIN_CYCLES=1000000`, injector STILL syncs first): **PASS.**
  Purpose: prove the in-kernel marker path works with `unmap-mid-flight`
  (previously only exercised with SIGKILL). Result: helper thread
  detected `HIT_IN_KERNEL_WINDOW` (kernel entered the marker, hadn't
  exited yet), then `_do_inject_unmap()` → `cudaDeviceSynchronize()`
  blocked until the kernel finished spinning, so `cuMemUnmap` effectively
  fired POST-kernel. Peer-visible outcome identical to Attempt #2
  (before-dispatch). Bug fixed en-route: `elastic.py:1096`
  `fault_inject_mode=_fim` (undefined in worker scope) →
  `args.fault_inject_mode`.

* **Attempt #4** (same job, injector's `cudaDeviceSynchronize()`
  REMOVED, `IN_KERNEL_SPIN_CYCLES=1000000`): **PASS**, but
  `LATE_UNMAP`. Detection was in-window (`HIT_IN_KERNEL_WINDOW`,
  `exited=0`), but the three `cuMem*` calls took ~26 ms while the
  kernel only spun for ~500 µs (`1e6` cycles). Post-unmap recheck saw
  `exited=sequence` → the kernel had already exited the marker before
  `cuMemUnmap` finished its driver-side work. Peer-visible outcome
  again identical to before-dispatch. Interpretation: removing the
  sync is necessary but not sufficient — the kernel must also stay
  in-window for at least the duration of the `cuMem*` teardown.

* **Attempt #5** (same job, no sync, `IN_KERNEL_SPIN_CYCLES=200000000`
  ≈ 100 ms of GPU spin): **PASS**, and this time
  `TRUE_IN_KERNEL_UNMAP` — the first time we've observed the driver's
  peer-PTE invalidation completing while rank 2's dispatch kernel was
  still actively spinning on the freed range. Note: the spin gates
  only thread (0,0) of block 0 (the marker-owning thread — see
  `nixl_ep_ll.cu:65`); every other thread on every other SM returns
  from the marker helper immediately and proceeds into the real
  dispatch-send NVLink writes. So `TRUE_IN_KERNEL_UNMAP` means the
  unmap returned while real NVLink send traffic from rank 2 was
  in-flight on the other SMs — not just during an idle spin. We
  cannot prove packet-level in-flight state from user-space (no CUDA
  API exposes it); the strongest available signal is that the
  send-loop code is executing on non-latched threads. Post-hoc log
  grep: no peer (0/1/3 on-node, 4-7 off-node) surfaced an
  `illegal_address` / `cudaError` / `xid` between INJECT and MASK
  DETECTED — in this run the peers evidently completed their
  writes into rank 2's exposed slot before the peer-PTE downgrade
  landed, and then blocked at the post-dispatch barrier waiting
  for rank 2's ack. Rank 2's own MMU caught the freed range (its
  own next NVLink op faulted) and became the sole user-space
  fault signal; peers only saw the 30 s NIXL-EP mask timeout.
  This is consistent with the invariant (no peer packet could
  have reached the wire even under a tighter race), but it does
  not directly fingerprint a peer-side MMU hit. Timeline (rank 2):
  `HIT_IN_KERNEL_WINDOW → INJECT begin +3.6 ms → INJECT done +0.34 ms
  (3 cuMem calls) → post-check: exited_after_unmap=0`. All three
  cuMem calls together took ~340 µs on the warm path (vs. the ~26 ms
  cold-path first-run cost seen in Attempt #4). **Even under this
  race, the peer-visible outcome is identical**: MNNVL clique
  unchanged, no XID 74/79/154, no NVLink counter delta, no IMEX
  error, no contain-and-drain. Rank 2 self-faulted
  (`cudaErrorIllegalAddress`) on its next CUDA op; peers 0/1/3
  detected via `MASK DETECTED dead_rank=2` at `where=post-dispatch`
  (one collective earlier than Attempt #4's `where=post-combine`,
  since the fault landed during dispatch). Directly validates the
  "invalidate PTE before free" driver invariant under a live-kernel
  race — the MMU catches the peer access locally before any packet
  can leave.

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
  the dying process's `/dev/nvidia*` fds. That release path calls
  `RmFreeUnusedClients` → `rmapiFreeClientListWithSecInfo` →
  `serverFreeResourceTree` (all three names appear in real NVIDIA
  kernel log lines and in the open-kernel source), which walks the
  client's resource tree. For every `RsInterMapping` (a peer PTE
  mapping into the dying context's fabric memory), the tree walker
  invokes `serverInterUnmap` → `rmclientInterUnmap_IMPL` →
  `resUnmapFrom` (vtable dispatch) →
  **`memoryfabricUnmapFrom_IMPL`** (`mem_fabric.c:1271`) →
  `_memoryFabricDetachMem` (`mem_fabric.c:147`) →
  **`fabricvaspaceUnmapPhysMemdesc_IMPL`** (`mem_fabric.c:171`).
  That is the *same* function the `cuMemUnmap` path reaches (via
  the shorter `memoryfabricCtrlDetachMem_IMPL` entry at
  `mem_fabric.c:1110`). The exporter's `memoryfabricDestruct_IMPL`
  itself contains `NV_ASSERT(pNode == NULL)` on the attached-memory
  tree (`mem_fabric.c:340`, comment: *"Every attached memory should
  have been unmapped by now"*), which enforces at driver level that
  peer PTEs must be invalidated before physical pages are freed.

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
signals on peers, because both paths converge on the *same*
per-region unmap function (`fabricvaspaceUnmapPhysMemdesc_IMPL`).
`cuMemUnmap` reaches it via `memoryfabricCtrlDetachMem_IMPL` →
`_memoryFabricDetachMem`; SIGKILL reaches it via
`serverFreeResourceTree` → `serverInterUnmap` →
`memoryfabricUnmapFrom_IMPL` → `_memoryFabricDetachMem`. In both
cases the exporter's `memoryfabricDestruct_IMPL` refuses to free
physical pages until its attached-memory tree is empty
(`NV_ASSERT(pNode == NULL)` at `mem_fabric.c:340`).
The only real difference is what happens to the
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

## Next experiments

Follow-up work planned but not part of this commit:

* **Approach A — peer-slowdown for peer-side XID 31**: today we
  only ever see the initiating GPU's local MMU fault (victim rank
  hits `cudaErrorIllegalAddress` / XID 31 on its own host). Peers
  never surface an MMU fault in any of our 15+ runs because they
  finish their NVLink stores into the victim's buffer *before* the
  IMEX ack propagates the peer-PTE invalidation to their node — so
  the "peer's own MMU walks an invalid peer-PTE" window never opens.
  Approach A holds peers back by extending the in-kernel spin gate
  (`maybe_mark_in_kernel_fault_enter`) to fire on non-victim ranks
  too, controlled by a new `--peer-slowdown-spin-cycles` arg
  (wired via `PEER_SLOWDOWN_SPIN_CYCLES` env var in
  `run_phase5_2node.sh`). Only the victim runs the host-side
  injection helper; peers just spin. Recipe:
  ```
  TIMING=combine-send-during-kernel-no-hook \
  IN_KERNEL_SPIN_CYCLES=200000000 \
  PEER_SLOWDOWN_SPIN_CYCLES=200000000 \
  ./run_phase5_2node.sh
  ```
  Expected: peer-side XID 31 on at least one non-victim host (visible
  in privileged `dmesg` / SLURM epilog; the container-visible
  `AcceleratorError` on peer `summary.json` is the
  container-observable complement). Not deterministic — bump
  `PEER_SLOWDOWN_SPIN_CYCLES` if the first attempt lands within IMEX
  propagation. Fallback = **Approach B** below.
* **Approach B — explicit ordering primitive**: deterministic
  version of Approach A that uses a shared host-visible flag polled
  from GPU (via `cudaHostRegister` / `cudaHostGetDevicePointer`).
  Rank 2 sets the flag AFTER `cuMemUnmap` returns; peers only
  proceed past the flag check afterwards, so their NVLink store is
  guaranteed to be issued after IMEX ack landed on their node.
  Heavier instrumentation but should give one peer-side XID 31 per
  peer that runs the ordered store. Only implement if Approach A
  cannot be tuned to produce any peer-side XID 31.
* **Multiple timings per allocation**: run several of the CPU-level
  and in-kernel timings back-to-back inside one SLURM alloc. Add a
  per-iteration GPU health check to `run_phase5_2node.sh` —
  minimally `torch.cuda.is_available()` + a tiny cross-GPU P2P kernel
  between runs — and bail out of the sweep as soon as a check
  degrades, to avoid tripping the cluster's XID drain policy.
* **Other 7 in-kernel cells**: only
  `dispatch-send-during-kernel-no-hook` was validated end-to-end (as
  `TRUE_IN_KERNEL_UNMAP`, Attempt #5). The other seven
  (`dispatch-receive-during-kernel-*`, `combine-send-during-kernel-*`,
  `combine-receive-during-kernel-*`) are exposed by the same
  `TIMING=<value>` env var to `run_phase5_2node.sh` and should be
  mechanically identical, but are unvalidated.
* **spin_cycles auto-tune**: `IN_KERNEL_SPIN_CYCLES` currently
  defaults to `1_000_000` (~500 µs on GB200), which is enough for
  the helper thread to *detect* the window but NOT enough to
  guarantee `cuMemUnmap` completes while the kernel is still
  spinning (see Attempt #4 → `LATE_UNMAP` result). Attempt #5 used
  `200_000_000` (~100 ms) and got `TRUE_IN_KERNEL_UNMAP`. The
  right value depends on driver warm/cold state — cold-first
  `cuMem*` teardown was ~26 ms, warm was ~340 µs. Consider making
  the runner iterate spin values (`1e6, 1e7, 1e8, 2e8`) and record
  the first that yields `TRUE_IN_KERNEL_UNMAP`.
* **Deliberately-provoked transport-fatal (out of scope for
  user-space)**: catching an actual XID 74/79/154 requires a
  hardware event the driver cannot cooperatively handle — cable
  pull, forced GPU reset, GPU hard-hang, or uncorrectable ECC. All
  require cluster-admin cooperation and are outside the scope of an
  unprivileged Slurm test.

## Reproducing the test

    # inside an --exclusive SLURM alloc across 2 MNNVL-coupled GB200 nodes:
    SLURM_JOB_ID=<job>  SLURM_JOB_NODELIST=<host1,host2> \
        bash examples/device/ep/tests/elastic/run_phase5_2node.sh

Results land in `examples/device/ep/tests/elastic/results/phase5_2node_<ts>_<hosts>_<timing>/`.
The runner prints `PHASE5 PASS: ...` on success and exits non-zero on
any gate failure.

## Final conclusion — status message

> **Unmap experiments: done, all passing.** 2-node GB200. Victim
> tore down its own live peer-exposed RDMA buffer via
> `cuMemUnmap + cuMemAddressFree + cuMemRelease`. Two configs
> validated end-to-end: **`before-dispatch`** (Attempt #2, unmap
> between collectives) and **`dispatch-send-during-kernel-no-hook`
> + no sync + long spin** (Attempt #5, `TRUE_IN_KERNEL_UNMAP` —
> unmap returned while the dispatch kernel was still in the
> marked send window).
>
> Same peer-visible signature in both: victim died on next CUDA op
> with `cudaErrorIllegalAddress`; 7 peers ran normally and detected
> the victim via NIXL-EP 30-s mask timeout; MNNVL fabric intact, no
> transport-layer fatal. Slack alert on the `before-dispatch` run
> reported **XID 31** on the victim's node — independent
> hardware-level confirmation of a real driver-level fault.
>
> **Interpretation.** The in-kernel race (Attempt #5) is the
> worst-case for the driver's peer-side cleanup and produces the
> same signature as the phase-boundary case — direct empirical
> evidence that the "invalidate PTE before free" invariant in
> `fabricvaspaceUnmapPhysMemdesc_IMPL` holds under a live-kernel
> race, and that NIXL-EP's timeout-based elastic recovery is the
> correct software response for this class of fault.
>
> Details: `EXPERIMENT_UNMAP_FAULT.md`, `DRIVER_FINDINGS.md`.

<details>
<summary><b>Optional explanation — why neither SIGKILL nor unmap can produce an NVLink transport-layer fatal on multi-node MNNVL (by design)</b></summary>

**Setup.** MNNVL (Multi-Node NVLink, e.g. GB200 NVL72) puts multiple
nodes into one NVLink fabric via external NVSwitches. Each node
still runs its **own independent NVIDIA kernel driver instance** —
there is no shared kernel across nodes. So the honest question is:
if rank 2 on `lyris0232` calls `cuMemUnmap`, only `lyris0232`'s
driver runs the local invalidation. How do peer GPUs on `lyris0233`
learn to invalidate their PTEs into rank 2's fabric memory before
those pages are freed?

**Answer: IMEX.** NVIDIA's cross-node coordination for MNNVL is the
IMEX daemon (`nvidia-imex`, one instance per node) running the
Internode Memory Exchange (IMEX) protocol over a control channel
(not over NVLink data path). The relevant unmap sequence is:

1. Rank 2 on `lyris0232` calls `cuMemUnmap`.
2. `lyris0232`'s kernel driver runs
   `fabricvaspaceUnmapPhysMemdesc_IMPL` — clears LOCAL peer PTEs
   (i.e. the mappings held by GPUs on `lyris0232` into rank 2's
   fabric memory) and issues a `fabricvaspaceInvalidateTlb` with
   `PTE_DOWNGRADE` reason code, paired with the driver's usual
   fence sequence (`kbusFlush_HAL` before the invalidate, and — per
   the HW-facing GMMU docs — a `SYS_MEMBAR` / `UFLUSH`-class
   primitive on the invalidate itself) that drains in-flight
   accesses on the local GPUs' NVLink pipelines. See
   `DRIVER_FINDINGS.md` §"What `PTE_DOWNGRADE` actually is and is
   NOT (revised)" for why the drain is provided by
   `SYS_MEMBAR`/`UFLUSH`, not by `PTE_DOWNGRADE` alone.
3. Before returning, `lyris0232`'s driver posts a fabric-detach
   message to its local IMEX daemon.
4. `lyris0232`'s IMEX daemon relays the detach to `lyris0233`'s
   IMEX daemon (control-plane socket), which calls into
   `lyris0233`'s kernel driver.
5. `lyris0233`'s driver runs the *same*
   `fabricvaspaceUnmapPhysMemdesc_IMPL` locally, clearing peer PTEs
   held by its own GPUs, then acks.
6. Only after every remote node acks does `lyris0232`'s driver
   allow the physical pages to be freed
   (`memoryfabricDestruct_IMPL` won't tear the exporter down until
   its attached-memory tree is empty —
   `NV_ASSERT(pNode == NULL)` at `mem_fabric.c:340`, comment:
   *"Every attached memory should have been unmapped by now"*).

So although the two nodes' drivers are independent, they are
**serialized by the IMEX two-phase protocol**. The exporter's
"free physical pages" step is gated on every remote node having
completed its own local PTE-downgrade. If any node were unreachable
or too slow, the local unmap would block or fail rather than free
pages behind unrelayed peers — the invariant is enforced by the
kernel driver, not by user-space etiquette.

**Result on both nodes' GPUs**: any peer's next NVLink store into
the freed range is caught **locally at that peer's own GPU MMU**
(→ `XID 31` in that peer's own host kernel log), before a packet
is placed onto the NVLink wire. No packet leaves → NVLink
transport layer has nothing outstanding to time out on → no
transport-fatal escalation, no `XID 74/79/154`, no
contain-and-drain, no MNNVL clique change. The IMEX log stays
clean because IMEX only reports errors (unreachable peer, control
timeout), not successful unmap acks.

Both fault-injection paths converge on this same per-node function:

* `cuMemUnmap` → `memoryfabricCtrlDetachMem_IMPL`
  (`mem_fabric.c:1110`) → `_memoryFabricDetachMem`
  (`mem_fabric.c:147`) → `fabricvaspaceUnmapPhysMemdesc_IMPL`
  (+ IMEX broadcast to peer nodes).
* SIGKILL → driver `.release` → `RmFreeUnusedClients` →
  `serverFreeResourceTree` → `serverInterUnmap` →
  `rmclientInterUnmap_IMPL` → `resUnmapFrom` →
  `memoryfabricUnmapFrom_IMPL` (`mem_fabric.c:1271`) →
  `_memoryFabricDetachMem` → `fabricvaspaceUnmapPhysMemdesc_IMPL`
  (+ same IMEX broadcast).

Same per-region unmap function, same IMEX cross-node acks, same
invariant — this is why SIGKILL and unmap-mid-flight produce
indistinguishable peer-visible signatures across both nodes in our
runs.

A real transport-layer NVLink fatal would require an asynchronous
hardware event that bypasses this cooperative teardown entirely —
cable pull, forced GPU reset, GPU hard-hang, uncorrectable ECC,
NVSwitch port disable via BMC/MFT/NMX-M/Redfish. All require
cluster-admin cooperation and are outside the scope of an
unprivileged user-space test.

</details>
