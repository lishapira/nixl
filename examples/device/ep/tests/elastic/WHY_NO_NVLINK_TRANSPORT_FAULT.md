# Why software fault injection cannot produce an NVLink transport-layer error

## Context

We exercised two user-space fault-injection modes against a live,
peer-exposed NIXL-EP RDMA buffer on 2-node MNNVL GB200:

* `sigkill` — the victim rank SIGKILLs itself; the Linux driver's
  `/dev/nvidia*` `.release` file-op tears down its GPU resources.
* `unmap-mid-flight` — the victim rank calls
  `cuMemUnmap + cuMemAddressFree + cuMemRelease` on its own live
  peer-exposed RDMA buffer while peers are actively reading it, and
  stays alive at Python level.

## TL;DR

Two independent mechanisms carry the load.

* **Reason 1** — Both fault paths are **driver-cooperative**. The
  NVIDIA kernel driver runs a synchronous cleanup on the victim's own
  node that clears the victim's peer-facing PTEs (`vaspaceUnmap` →
  bus flush → TLB invalidate) before returning to user-space. Any
  further access from the victim's own SMs faults locally at the
  victim's own MMU, so no stale-mapping packet ever leaves the
  victim's GPU.
* **Reason 2** — `cuMemUnmap` / SIGKILL are **caller-local** for MNNVL
  fabric memory: they only touch the caller's own VA→physical mapping.
  Peers hold their own imported handles into the same fabric memory
  (installed at NIXL agent connect via
  `cuMemImportFromShareableHandle` + `cuMemMap`), and those imports
  hold refcounts on the underlying physical fabric page. The peer's
  own PTE stays valid, the peer's own send warps keep issuing
  `st.na.global` into that VA, and the store lands on the still-live
  physical page (kept alive by the peer's own importer refcount even
  after the victim's process is gone). No peer-side MMU fault.

Reason 1 explains why we never see a transport-layer fatal; Reason 2
explains why the only `XID 31` we see is on the victim, never on peers.

## Reason 1 — Driver-cooperative teardown drains the victim's peer-PTEs before freeing pages

`cuMemUnmap` (from user-space) and `SIGKILL` (via the driver's
`.release` file-op cleanup of the dying process's `/dev/nvidia*` fds)
converge on the same per-region unmap function
`fabricvaspaceUnmapPhysMemdesc_IMPL` in the exporter's kernel driver:

* `cuMemUnmap` → `memoryfabricCtrlDetachMem_IMPL` →
  `_memoryFabricDetachMem` → `fabricvaspaceUnmapPhysMemdesc_IMPL`.
* `SIGKILL` → `.release` → `RmFreeUnusedClients` →
  `serverFreeResourceTree` → `serverInterUnmap` →
  `rmclientInterUnmap_IMPL` → `resUnmapFrom` →
  `memoryfabricUnmapFrom_IMPL` → `_memoryFabricDetachMem` →
  same `fabricvaspaceUnmapPhysMemdesc_IMPL`.

`fabricvaspaceUnmapPhysMemdesc_IMPL` clears the victim's own
peer-facing PTEs synchronously (`vaspaceUnmap`), then invalidates the
TLB with `PTE_DOWNGRADE` reason code, paired with a
`SYS_MEMBAR`/`UFLUSH`-class fence that drains in-flight NVLink accesses
on the *local* GPU's pipelines. See `DRIVER_FINDINGS.md` for the exact
call chains and the `PTE_DOWNGRADE`-vs-fence attribution.

By the time `cuMemUnmap` (or the `.release` cleanup) returns, the
victim's own GMMU no longer has a valid translation for the freed VA
range on the victim's own GPU. Any further SM access from the still-
running dispatch/combine kernel — or from the next launched kernel —
walks an invalid PTE and takes a local MMU fault (`XID 31` on the
victim's host, `cudaErrorIllegalAddress` at Python level). That is
exactly the fault surface we observe.

Physical pages the buffer occupies are also not freed while any
importer (local or remote) is still holding a handle. The exporter's
destructor `memoryfabricDestruct_IMPL` refuses to free with
`NV_ASSERT(pNode == NULL)` on the attached-memory tree (comment:
*"Every attached memory should have been unmapped by now"*). The tree
is populated on every importer's `_memoryFabricAttachMem` call —
including cross-node importers reached through IMEX. So even at
end-of-test physical-page free the invariant "peer PTEs invalidated
before pages freed" is enforced by construction.

Consequence: at no point does the NVLink transport layer have an
outstanding request from the victim to an invalid destination that
could time out and escalate to contain-and-drain.

**Evidence.**

* Source: [`NVIDIA/open-gpu-kernel-modules`](https://github.com/NVIDIA/open-gpu-kernel-modules)
  (dual-licensed GPL+MIT — this IS the `nvidia.ko` running on Lyris,
  driver 580.173 per `nvidia-smi -L`). Files
  `src/nvidia/src/kernel/mem_mgr/mem_fabric.c` and
  `.../fabric_vaspace.c`. Function names are stable across R580/R610
  tags; line numbers drift.
* Empirical: 10+ runs on Lyris 2-node GB200 MNNVL, 4 CPU-level
  timings + `TRUE_IN_KERNEL_UNMAP` in-kernel timing:
  * Victim `summary.json`: `extra.exception = "AcceleratorError:
    CUDA error: an illegal memory access was encountered"` in every
    run.
  * SLURM cluster-monitoring bot reported `NVRM: Xid ... 31 ...` on
    the victim's host in every run.
  * `nvidia-smi nvlink --errorcounters` PRE vs. POST delta = 0.
  * MNNVL clique identical PRE and POST.

## Reason 2 — `cuMemUnmap` / SIGKILL are caller-local; peer's imported fabric mapping persists

The victim's `cuMemUnmap` unbinds only the victim's own VA → physical
mapping (i.e. what the victim's SMs walk). SIGKILL is the same shape —
`.release` runs `_memoryFabricDetachMem` for every resource in the
dying process's resource tree, which drops the victim's own PTEs and
the victim's own ref on each imported handle; it does not walk any
peer process's resource tree.

Peers hold their own imported handles into that fabric memory,
installed at NIXL agent connect via
`cuMemImportFromShareableHandle` + `cuMemMap`. Each import bumps a
refcount on the underlying physical fabric page (via
`_memoryFabricAttachMem`) and installs a separate VA → physical entry
in the peer's own GPU MMU. The victim's `cuMemUnmap` / SIGKILL
decrements only the victim's own refs; importer entries on peers stay
live.

Two things follow:

1. The peer's own VA → physical PTE for "rank 2's fabric memory" is
   still valid after rank 2's `cuMemUnmap` / SIGKILL. A peer send-warp
   `st.na.global` into that VA walks a valid PTE → no page fault → no
   `XID 31` on the peer.
2. The physical fabric page is not freed on the victim's node (the
   attached-memory tree from Reason 1 is non-empty). The store lands
   on live physical memory. It is semantically stale (the victim will
   never read it), but it does not fault.

### Direct probe evidence

The P2P probe (`IN_KERNEL_P2P_NULL_COUNT` /
`IN_KERNEL_P2P_NONNULL_COUNT` slots in `nixl_ep_ll.cu`) counts every
`p2p_ptr_get(dst_rank=victim)` return value from every non-victim
rank's dispatch/combine send warp. Non-null == the peer's own imported
P2P mapping into the victim's fabric memory is still valid from the
peer's GPU perspective.

Both fault modes, all runs, all timings we probed: `null = 0`. Peer
send-warps take the raw `st.na.global` branch every iteration; the
store lands on the still-live physical fabric page kept alive by the
peer's importer refcount.

Representative snapshots:

| Fault mode | Timing | Peers × probe reads | `null=0` reads | `null≠0` reads |
|---|---|---|---|---|
| `cuMemUnmap` | `before-dispatch` (~30 s NIXL-EP timeout window) | 7 × >250 | all | 0 |
| `cuMemUnmap` | `dispatch-send-during-kernel-no-hook`, `TRUE_IN_KERNEL_UNMAP` | 7 × >250 | all | 0 |
| `SIGKILL` | `before-dispatch` | 7 × 112 | 784 | 0 |
| `SIGKILL` | `dispatch-send-during-kernel-no-hook`, `HIT_IN_KERNEL_WINDOW` | 7 × 112 | 784 | 0 |

Reproduce:

    P2P_PROBE_TARGET=2 TIMING=before-dispatch \
        bash run_nvlink_fault_inject_2node.sh
    grep P2P_PROBE_COUNTS results/nvlink_fault_inject_2node_.../{master,worker}.log

## Consequence for observable signals

| Signal we hoped to see | Observed? | Blocked by |
| --- | --- | --- |
| `XID 31` on victim host | Yes | — (this IS what we observe) |
| `XID 31` on peer host | No (in-container) | Reason 2 |
| `XID 74 / 79 / 154` (NVLink transport-fatal) | No | Reason 1 |
| `nvidia-smi nvlink --errorcounters` delta | No | Reason 1 |
| IMEX `[ERROR]` line in the run window | No | Reason 1 |
| MNNVL clique change (Cluster/Clique diff) | No | Reason 1 |
| Peer `AcceleratorError` in Python | No | Reason 2 |
| NIXL-EP mask timeout on peer @ 30 s | Yes | — (this is the SW recovery signal we want) |

## Hardware-view conclusion

From every peer's GPU perspective, rank 2 remained a **valid NVLink
destination** for the full ~30 s from the moment the fault fired until
peers software-masked it. Every peer send warp during that window
resolved rank 2's fabric memory to a valid P2P pointer (P2P probe
`null=0` across every run) and took the raw `st.na.global` branch to
issue the NVLink store, which was serviced by rank 2's still-live
physical fabric page in HBM (kept alive by the peer importer refcount
even after rank 2's process died on SIGKILL or unbinding its own VA on
`cuMemUnmap`). This holds whether rank 2 died gracefully via
`cuMemUnmap` or was hard-killed by `SIGKILL`, and whether it died at a
CPU-level phase boundary or mid-flight during its own send kernel.

## Caveat on observability

Container-visible peer `dmesg.log` is 0-byte in every run
(`dmesg_restrict=1`, no `CAP_SYSLOG`). "No peer-side XID" here means
"no peer-side XID visible to any Python-observable signal". The SLURM
cluster-monitoring bot (which does read peer host `dmesg`) reported
`XID 31` only on the victim's host, never on peer hosts, corroborating
the container-side signal.
