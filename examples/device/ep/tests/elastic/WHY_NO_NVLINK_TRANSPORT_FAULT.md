# Why software fault injection cannot produce an NVLink transport-layer error

TL;DR: `cuMemUnmap` and `SIGKILL` are **cooperative** teardowns — they
go through NVIDIA's kernel driver. The driver's contract is designed to
*prevent* stale P2P mappings from ever hitting the wire, so a real
NVLink transport-fatal (`XID 74 / 79 / 154`, contain-and-drain, NVLink
counter delta, IMEX `[ERROR]`) is structurally unreachable from
user-space, no matter what timing race we set up. The only symptom we
can produce is a local MMU fault on the victim (`XID 31` on the
victim's own host, `cudaErrorIllegalAddress` at Python level).

Two independent mechanisms carry the load — one blocks the transport
fatal, the other blocks a peer-side MMU fault:

* **Reason 1** — driver-cooperative teardown of the *victim's own*
  peer-facing PTEs, before physical pages can be freed → blocks any
  packet from leaving the victim on a stale mapping, and blocks
  physical-page reuse while any importer still holds it.
* **Reason 2** — `cuMemUnmap` / SIGKILL are caller-local for
  cross-node fabric memory: peers keep their own valid imported PTEs
  pointing at a still-live physical page → blocks the peer-side MMU
  fault we naively expected.

Reason 1 explains why we never see an NVLink transport-fatal; Reason 2
explains why the only XID 31 we see is on the victim, never on peers.

---

## Reason 1 — Driver-cooperative teardown drains the victim's own peer-PTEs BEFORE freeing pages

### 1.a — Local (victim's node)

Both fault-injection paths call synchronous, host-CPU driver code
that runs entirely on the *victim's own node*:

* `cuMemUnmap` (userspace) →  `libcuda.so` (closed) → `ioctl` →
  `memoryfabricCtrlDetachMem_IMPL` → `_memoryFabricDetachMem`
  (`mem_fabric.c`) → `fabricvaspaceUnmapPhysMemdesc_IMPL`
  (`fabric_vaspace.c`).
* `SIGKILL` → Linux runs `/dev/nvidia*` `.release` file-op →
  `RmFreeUnusedClients` → `serverFreeResourceTree` → `serverInterUnmap`
  → `rmclientInterUnmap_IMPL` → `resUnmapFrom` →
  `memoryfabricUnmapFrom_IMPL` (`mem_fabric.c`) →
  same `_memoryFabricDetachMem` → same
  `fabricvaspaceUnmapPhysMemdesc_IMPL`.

`fabricvaspaceUnmapPhysMemdesc_IMPL` clears the victim's own
peer-facing PTEs (with `PTE_DOWNGRADE` + `SYS_MEMBAR`/`UFLUSH`-paired
TLB flush) synchronously before returning to userspace. So by the
time `cuMemUnmap` returns, the victim's own GMMU no longer has a valid
translation for that VA range on the victim's own GPU. The next SM
access on the victim (from the still-running dispatch/combine kernel
or from the next launched kernel) walks an invalid entry and raises
`XID 31` locally.

### 1.b — Cross-node (importers on peer nodes)

The victim's `cuMemUnmap` **does not** ship a cross-node invalidation
to importer PTEs on peer nodes. That's why the peer-side probe we
ran on `lyris[0266-0267]` shows `p2p_ptr_get(dst_rank=2)` returning
non-null for the entire 30-s fault window — the peer's own imported
PTEs stay valid.

Instead, the "no stale mapping ever leaves the wire" property is
enforced at the *physical-page-free* boundary, not at the unmap
boundary. `memoryfabricDestruct_IMPL` (`mem_fabric.c`) refuses to
free physical pages while any attached-memory node still points at
them:

```c
// Every attached memory should have been unmapped by now.
btreeEnumStart(0, &pNode, pMemdescData->pAttachMemInfoTree);
NV_ASSERT(pNode == NULL);
```

That per-memdesc `pAttachMemInfoTree` is populated by
`_memoryFabricAttachMem` on every `cuMemImportFromShareableHandle` +
`cuMemMap` (including cross-node importers reached via IMEX, the
userspace daemon that brokers fabric handles across nodes). Each
importer's detach path (`_memoryFabricDetachMem`) calls
`btreeUnlink(...)` on its own node. So the invariant "physical page
cannot be freed while any importer, local or remote, still holds it"
holds across nodes by construction of the ref-counted import tree —
the driver just refuses to free until the tree is empty.

**In our specific test** the physical page is never actually freed
(peer importers never call `cuMemRelease`; they're still using the
page, they just eventually mask rank 2 as dead at the software
level). So cross-node PTE invalidation never has to fire at all
during our fault window. The peer's imported PTE and the physical
page it points at both stay live for the entire 30-s
NIXL-EP timeout — which is exactly what the probe measured.

### Consequence

Peer nodes never see an invalid mapping on any packet that hits the
wire, because:
* The victim's local PTEs are torn down before the victim's next SM
  access → the victim faults locally, no packet leaves the victim's
  GPU on that VA (1.a).
* Peer PTEs are not touched by the victim's `cuMemUnmap`, and the
  physical page they point at is kept alive by the importer refcount
  (1.b + Reason 2).

So the NVLink transport layer has nothing outstanding that could
time out on an invalid destination → no `XID 74 / 79 / 154`, no
NVLink error-counter delta, no IMEX `[ERROR]`.

### Evidence for Reason 1

**Source-code evidence** (all publicly readable):

* Repository: [`NVIDIA/open-gpu-kernel-modules`](https://github.com/NVIDIA/open-gpu-kernel-modules)
  (dual-licensed GPL+MIT). This IS the kernel-space `nvidia.ko`
  that is actually running on the Lyris nodes (driver 580.173,
  from `nvidia-smi -L`).
* Files:
  * `src/nvidia/src/kernel/mem_mgr/mem_fabric.c`:
    * `_memoryFabricAttachMem` — calls `btreeInsert(...)` on every
      importer attach (line ~270 in R610 tag).
    * `_memoryFabricDetachMem` — calls
      `fabricvaspaceUnmapPhysMemdesc(...)` then
      `btreeUnlink(...)` on every importer detach (lines ~171, ~182).
    * `memoryfabricCtrlDetachMem_IMPL` / `memoryfabricUnmapFrom_IMPL`
      — the two ioctl entry points that fan into
      `_memoryFabricDetachMem` (one for `cuMemUnmap`, one for
      SIGKILL/`.release`).
    * `memoryfabricDestruct_IMPL` (line ~899) — contains the
      `NV_ASSERT(pNode == NULL)` cited above.
  * `src/nvidia/src/kernel/mem_mgr/fabric_vaspace.c` —
    `fabricvaspaceUnmapPhysMemdesc_IMPL` (the function that actually
    walks the peer-PTE range and issues the `PTE_DOWNGRADE` +
    fenced TLB-flush RPC to GSP).
* Line numbers drift between R580 and R610 tags; the function names
  are stable across versions. Reproduce our reading with, e.g.
  <https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.173.02/src/nvidia/src/kernel/mem_mgr/mem_fabric.c>.

**What we could NOT verify from source**:

* `libcuda.so` (userspace CUDA driver) is closed source. We only
  see the kernel-side ioctl entry point that it ends up calling.
* GSP firmware (which is what physically executes the TLB flush on
  Blackwell/GB200) ships as opaque binary blobs. The kernel driver
  builds the RPC packet in the open code we read; execution on GSP
  is not source-visible.
* IMEX daemon (`nvidia-imex`) is closed source. Its wire protocol
  is not exercised by our unmap test (physical page never freed),
  so we did not need to model it here.

**Empirical evidence** (from our runs; matches what the source
predicts):

* 15+ runs on `lyris` (2-node GB200 MNNVL), 4 CPU-level timings
  (`before-dispatch`, `after-dispatch`, `before-combine`,
  `after-combine`) plus the in-kernel `TRUE_IN_KERNEL_UNMAP` variant
  plus Approach A (peer host-sleep 1 s / 5 s):
  * Victim's `summary.json` in every run: `exc = "AcceleratorError:
    CUDA error: an illegal memory access was encountered"`.
  * SLURM cluster-monitoring bot reported `NVRM: Xid ... 31 ...` on
    the victim's host in every run.
  * Victim's `WORKER DONE survived=false` in every run; Python
    process caught the `AcceleratorError` and shut down cleanly.
* `nvidia-smi nvlink --errorcounters` PRE vs. POST delta = 0 in every
  run (see `pre_lyris*.csv` / `post_lyris*.csv` in each run's artifacts
  directory).
* MNNVL clique / cluster ID unchanged pre → post in every run
  (see `mnnvl_pre.json` / `mnnvl_post.json`).

---

## Reason 2 — `cuMemUnmap` / SIGKILL are caller-local; peer's imported fabric mapping persists

`cuMemUnmap` on rank 2 tears down **only** rank 2's own VA →
physical-page mapping (i.e. the mapping that rank 2's own SMs walk
when they access that VA). It does not touch any other rank's page
tables. `SIGKILL` is the same shape: the `.release` file-op runs
`_memoryFabricDetachMem` for every resource in rank 2's process
resource tree, which drops rank 2's own PTEs and rank 2's own ref
on each imported handle — it does not walk peer processes' resource
trees.

Peers already hold their own imported handles to the same fabric
memory. These were set up at NIXL agent connect-time via
`cuMemImportFromShareableHandle` + `cuMemMap` on the peer's device.
Each import bumps a refcount on the underlying physical fabric page
(via `_memoryFabricAttachMem` → `btreeInsert`), and installs a
separate VA → physical entry in the peer's own GPU MMU tables.
Rank 2's `cuMemUnmap` / SIGKILL decrements only rank 2's own refs;
importer entries on peer nodes and their refs stay live.

Two things follow:

1. The peer's own VA → physical PTE for "rank 2's fabric memory" is
   still valid after rank 2's `cuMemUnmap`. When a peer send warp
   issues `st.na.global` to that VA, the peer's local MMU walks a
   valid PTE → no page fault → no XID 31 on the peer.
2. Because importers still hold refcounts, the physical fabric page
   itself is not freed on rank 2's node (`memoryfabricDestruct_IMPL`'s
   `NV_ASSERT(pNode == NULL)` would trip otherwise). The store lands
   on live physical memory. It's semantically stale (rank 2 will never
   read it), but it does not fault.

**Consequence**: no peer-side MMU fault, no packet ever hits a truly
invalid mapping on the wire, no NVLink transport-fatal.

### Evidence for Reason 2

Direct probe evidence (`IN_KERNEL_P2P_NULL_COUNT` /
`IN_KERNEL_P2P_NONNULL_COUNT` slots in `nixl_ep_ll.cu`) — the probe
counts every `p2p_ptr_get(dst_rank=2)` call on non-victim send warps
and tags the result as null vs. non-null. A non-null return means
UCX's device-side handle for rank 2's fabric memory is still valid,
i.e. the peer's importer mapping is still valid.

**`cuMemUnmap` (`lyris[0266-0267]`, job `2311948`, 2026-07-08, timing
`before-dispatch`, `PEER_CPU_SLEEP_MS=1000`)** — peer sleeps 1 s on
the host **after** rank 2's `cuMemUnmap` returned, orders of magnitude
longer than any plausible cross-node invalidation delay:

```
rank 0 first post-inject probe: null=0 nonnull=81   → grows to null=0 nonnull=254
rank 1 first post-inject probe: null=0 nonnull=77   → grows to null=0 nonnull=266
rank 3 first post-inject probe: null=0 nonnull=119  → grows to null=0 nonnull=272
rank 4 first post-inject probe: null=0 nonnull=118  → grows to null=0 nonnull=236
rank 5 first post-inject probe: null=0 nonnull=137  → grows to null=0 nonnull=274
rank 6 first post-inject probe: null=0 nonnull=133  → grows to null=0 nonnull=266
rank 7 first post-inject probe: null=0 nonnull=130  → grows to null=0 nonnull=260
```

Every peer, every read, from `INJECT+1000 ms` through the 30 s
NIXL-EP timeout window: `null = 0`.

**SIGKILL (`lyris[0211,0215]`, job `2312459`, 2026-07-08)** — identical
result under two timings:

| Timing | When rank 2 dies | Total probe reads | `null=0` reads | `null≠0` reads |
|---|---|---|---|---|
| `before-dispatch` | before any communication kernel starts | 784 | 784 | 0 |
| `dispatch-send-during-kernel-no-hook` | mid-flight during the dispatch kernel (verdict `HIT_IN_KERNEL_WINDOW`, `entered=648966`, `exited_before_sigkill=0`) | 784 | 784 | 0 |

In the in-kernel SIGKILL variant, the marker evidence file confirms
peer stores were in flight to rank 2 at the exact moment rank 2's
process was destroyed by the OS. Peer P2P mappings remained valid;
peer send warps kept taking the raw `st.na.global` branch; those
stores landed on the still-live physical fabric page (kept alive by
peer importer refcount even though rank 2's process is now dead and
its refs have been released by the driver's `.release` cleanup).
No peer XID, no transport-fatal.

**Peer survival across long delays** (`lyris[0046-0047]`, job
`2301816`, Approach A CPU-side, peers sleeping **5000 ms** after
rank 2's `cuMemUnmap` returned):

* `observed_error_type = None` in all 7 peer `summary.json`s.
* `xid_seen = false` in all 7 peer `summary.json`s.
* `exc = NONE` in all 7 peer `summary.json`s.
* All 7 peers `WORKER DONE survived=true`, `phases_completed=2`.
* Post-fault peer iterations pass the correctness hook with
  numerical diff ~1e-6 (would be huge / NaN if peer's CUDA context
  were poisoned).

**Caveat on container observability**: container-visible peer
`dmesg.log` is 0-byte in every run (`Operation not permitted` under
`dmesg_restrict=1`). We cannot positively assert "no peer-side XID
appeared in the peer node's kernel ring buffer" from user-space; the
assertion is that no peer-side XID surfaced through any
Python-observable signal, and the SLURM cluster-monitoring bot
reported XID 31 *only* on the victim's host, not on peer hosts.

Reproduce: run
`P2P_PROBE_TARGET=2 PEER_CPU_SLEEP_MS=1000 TIMING=before-dispatch ./run_phase5_2node.sh`
and grep for `P2P_PROBE_COUNTS.*null=` in `master.log`/`worker.log`.

Source of the probe:
`nixl_ep.cpp` (`set_p2p_probe_target`, `reset_p2p_probe_counts`),
`nixl_ep_ll.cu` (`maybe_probe_p2p_ptr_null` inserted after each
`p2p_ptr_get`), `elastic.py` (`_log_p2p_probe_snapshot`).

---

## Summary table

| Signal we hoped to see | Observed? | Which reason blocks it |
| --- | --- | --- |
| `XID 31` on victim host (local MMU fault) | Yes | — (this IS what we observe) |
| `XID 31` on peer host | No, in-container | Reason 2 (peer PTE + physical page both live) |
| `XID 74 / 79 / 154` (NVLink transport-fatal) | No | Reason 1 (cooperative teardown drains before free) |
| `nvidia-smi nvlink --errorcounters` delta | No | Reason 1 (no packet leaves the wire) |
| IMEX `[ERROR]` log line | No | Reason 1 (IMEX only reports control-plane errors) |
| MNNVL clique change / fabric partition | No | Reason 1 |
| Peer `AcceleratorError` in Python | No | Reason 2 |
| NIXL-EP mask timeout on peer (30 s) | Yes | — (this is the software recovery signal we want) |

---

## Hardware-view conclusion (final)

From every peer's GPU perspective, rank 2 remained a **valid NVLink
destination** for the full ~30 s from the moment rank 2 was killed
until the peer software-masked it. Every peer send warp during that
window resolved rank 2's fabric memory to a valid P2P pointer
(probe: `null=0` across 15+ runs, 7 peers, 5000+ post-fault snapshots)
and took the raw `st.na.global` branch to issue the NVLink store.
The store was serviced by rank 2's still-live physical fabric page
in HBM, kept alive by the peer-importer refcount even though rank
2's own process is dead (SIGKILL) or has released its own ref
(`cuMemUnmap` + `cuMemRelease`). We infer "landed" (rather than
directly observe it) from two facts: NVLink `st.na.global` is a
fire-and-forget, no-ack store — the SM doesn't wait for a receipt —
so the software-side outcome must be inferred from what *didn't*
happen; and if the store had walked an invalid PTE it would have
raised peer-side `XID 31` on the peer's own host — which the SLURM
cluster-monitoring bot never reported on any peer host, and no
peer's `summary.json` ever surfaced `cudaErrorIllegalAddress`
across any run.

This holds:

* Whether rank 2 died gracefully via `cuMemUnmap + cuMemAddressFree
  + cuMemRelease` or was hard-killed by `SIGKILL` (OS signal 9,
  driver's `.release` file-op cleanup).
* Whether it died **before** communication kernels started
  (`before-dispatch` timing) or **mid-flight** while its own
  dispatch kernel was actively participating in an NVLink send
  (`dispatch-send-during-kernel-no-hook` with `HIT_IN_KERNEL_WINDOW`
  verdict, `entered=sequence` and `exited_before_sigkill=0`).

Empirical bounds tested:

| Fault mode | Timing | Runs | Peer XID? | Peer `null≠0`? | Peer `WORKER DONE survived=true`? | Mask detected on all 7 peers? |
|---|---|---|---|---|---|---|
| `cuMemUnmap` | 4 CPU-level timings + in-kernel `TRUE_IN_KERNEL_UNMAP` + Approach A CPU-sleep 1s/5s | 15+ | No | No | Yes | Yes |
| `SIGKILL` | `before-dispatch` | 1 (probe) + 12 sweep | No | No | Yes | Yes |
| `SIGKILL` | `dispatch-send-during-kernel-no-hook` (in-kernel `HIT_IN_KERNEL_WINDOW`) | 1 (probe) | No | No | Yes | Yes |

**Software recovery**: on every one of the above runs the surviving
7 peers correctly detected rank 2 as dead via the NIXL-EP mask
timeout at `DEFAULT_TIMEOUT_MS = 30 000` ms, marked rank 2 in the
dead-rank mask, and proceeded through the next phase's iterations
without contributions from rank 2. `phase5_pass_gate.py` triangulates
this via three independent signals: `mask_detected_on_peer`,
`nixl_ep_msgs_interrupted`, and per-peer `summary.json`
correctness.

---

## What WOULD produce an NVLink transport-fatal (out of scope for user-space)

Anything that skips the driver's cooperative cleanup path:
* Physical link disruption — cable pull, NVSwitch port disable via
  BMC / MFT / NMX-M / Redfish.
* Forced GPU reset (`nvidia-smi -r`, requires `sudo`).
* Uncorrectable ECC exhausting PLR retries; real PHY-level link down.
* SM hard-hang from a bug that survives context teardown.

All require cluster-admin cooperation. Producing them from an
unprivileged Slurm container is not possible on this cluster.
