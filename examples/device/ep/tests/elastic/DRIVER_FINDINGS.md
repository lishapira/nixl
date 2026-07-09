# Driver-side findings for the NVLink unmap-fault experiment

Notes from browsing NVIDIA's public `open-gpu-kernel-modules` source
(github.com/NVIDIA/open-gpu-kernel-modules) and CUDA driver docs, to
back up the claims in `EXPERIMENT_UNMAP_FAULT.md` about why
cooperative teardowns produce local MMU faults instead of NVLink
transport-fatals. All findings apply to the MNNVL fabric memory path
that our test exercises on GB200. Companion to `EXPERIMENT_UNMAP_FAULT.md`.

## What the driver does when a peer-exposed VMM buffer is unmapped
### via `cuMemUnmap` (source-verified)

The relevant function is `fabricvaspaceUnmapPhysMemdesc_IMPL` in
`src/nvidia/src/kernel/mem_mgr/fabric_vaspace.c`. It runs when
userspace calls `cuMemUnmap` on a MNNVL peer-exposed buffer. Its
ordering is:

1. Walk every mapping region, clear the peer's page-table entries
   (`vaspaceUnmap`).
2. Invalidate the TLBs on all affected GPUs
   (`fabricvaspaceInvalidateTlb`) with a `PTE_DOWNGRADE` reason
   code.
3. Only after this returns does the caller upstream free the
   physical pages.

A related address-range free function in the same file also flushes
the bus (`kbusFlush_HAL`) between PTE clear and TLB invalidate.

The fabric VA space destructor `fabricvaspaceDestruct_IMPL` contains
`NV_ASSERT(!gvaspaceIsInUse(...))`, i.e., it asserts that all
per-region allocations have already been torn down before the
vaspace itself is destroyed. So the individual per-region unmap has
to complete before the top-level destructor runs.

So the *ordering* "peer PTEs are cleared before the underlying
resource is released" is not empirical guesswork for the
`cuMemUnmap` path — it is directly implemented in this order by the
RM code.

### What `PTE_DOWNGRADE` actually is and is NOT (revised)

Earlier revisions of this note asserted that `PTE_DOWNGRADE` "fences
in-flight accesses". A follow-up query to NVIDIA-internal docs (via
Glean) does **not** support that phrasing, and the wording has been
corrected below.

**What `PTE_DOWNGRADE` is.** It is the `reason` / `type` field of
the GMMU TLB-invalidate command — it tells the invalidate engine
"this is a permission-removal invalidate" (as opposed to a plain
"we changed a PTE, please refresh" invalidate). It does **not** by
itself carry drain-until-visibility semantics.

**What actually provides the fence.** Per NVIDIA HW-facing docs
(GSP / "Detect memory subsystem hang from Kernel RM", section 4),
GMMU invalidate operations come in three flavours:

* `TLBI` with no membar — "ensures the GMMU is not blocked".
  **Not a fence.**
* `UFLUSH` — issues a sysmembar and flushes pending writes from FB
  to their final locations across FBHUB, HSHUBs, VidL2s, SysL2 and
  **NVLinks**. This is a full fabric-scope flush.
* `TLBI + NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_SYS_MEMBAR=TRUE`
  — TLBI from GMMU **plus** a FLUSH/Membar at TLBI completion
  covering all GPU clients (GPC + HUB side). A HW/RM comment
  summarises this as: *"A TLBI + sysmembar waits for all memory
  references (reads, writes, greds, gatoms, ...) to reach a point
  of visibility."* This is the operation that provides the drain
  semantics we care about.

So a plain PTE_DOWNGRADE TLBI ack alone is **not** documented to
guarantee "every in-flight access on the old translation has
landed or faulted". The documented drain is tied to the companion
`SYS_MEMBAR` / `UFLUSH` operations issued alongside or after the
invalidate.

**What that means for our invariant.** It's the *combination* of

* explicit ordering in the RM code (`vaspaceUnmap` → bus flush →
  TLBI → free),
* the neighbouring `kbusFlush_HAL` call visible in the same
  fabric-vaspace file, and
* the UVM-side comments (see next section) explicitly documenting
  "sysmembar before TLB invalidate" and "wait-for-idle before
  invalidating the TLB",

that gives us "in-flight peer traffic is drained before physical
pages are freed" — not `PTE_DOWNGRADE` on its own. We do **not**
have a Hopper/Blackwell HAL comment surfaced via public sources
that says "PTE_DOWNGRADE alone fences the NVLink pipeline"; anyone
claiming that (including earlier versions of this doc) was
overreaching.

### Direct evidence that explicit ordering is required (UVM-326)

The clearest documented case that "invalidate is not by itself
enough to prevent peers from observing freed pages" is UVM
issue **UVM-326**:

> "Use `MEMBAR.SYS` for PTE downgrades to local vidmem" — the
> report describes a case where TLB invalidate + `MEMBAR.GL` was
> *not* sufficient because `membar.gl` "doesn't wait for probe";
> the driver could unmap/free a physical page P, reallocate it,
> and a prior GPU access could then observe the *new* contents.
> The stated fix was to switch to `MEMBAR.SYS` when future access
> to that page can come from the CPU or a peer GPU.

This is the direct driver-side rule for MNNVL / peer-visible
memory: the fence on the downgrade path must be system-scope
(`MEMBAR.SYS`), not just GPU-local (`MEMBAR.GL`). This is
consistent with our empirical observation that we always see
local MMU faults (XID 31) on the initiating GPU and never a
transport-layer fatal — the RM path evidently uses a system-scope
fence somewhere in the sequence, even if we haven't pinpointed
the exact call in the 580.173 source.

### via SIGKILL / process death (source-verified)

When a process dies (SIGKILL, or any other abrupt exit), the OS
closes its `/dev/nvidia*` fds and the NVIDIA driver's `.release`
file-op runs. That path is:

```
RmFreeUnusedClients            (real kernel log line seen in NVIDIA issue #272)
 → rmapiFreeClientListWithSecInfo
 → Nv01FreeClientList
 → clientFreeResource_IMPL / rmclientFreeResource_IMPL   (client.c)
 → serverInitFreeParams_Recursive
 → serverFreeResourceTree                                (rs_server.c)
 → walks the dying client's resource tree; for every
   RsInterMapping (a peer PTE mapping into this context's
   fabric memory) it invokes:
     → serverInterUnmap                                  (rs_server.c)
     → rmclientInterUnmap_IMPL                           (client.c:374-397)
     → resUnmapFrom                                      (vtable dispatch)
     → memoryfabricUnmapFrom_IMPL                        (mem_fabric.c:1271)
     → _memoryFabricDetachMem(..., bRemoveInterMapping=NV_FALSE)
                                                          (mem_fabric.c:147)
     → **fabricvaspaceUnmapPhysMemdesc_IMPL**            (mem_fabric.c:171)
        │   ← same function called by the cuMemUnmap path
        └─ vaspaceUnmap (clear peer PTEs) →
           fabricvaspaceInvalidateTlb(..., PTE_DOWNGRADE)
           (+ SYS_MEMBAR / UFLUSH-class fence — see revised
            "What PTE_DOWNGRADE actually is and is NOT" above)
 → then per-resource destructor:
     → memoryfabricDestruct_IMPL                         (mem_fabric.c:899)
        → NV_ASSERT(pNode == NULL) on pAttachMemInfoTree (mem_fabric.c:340)
              *"Every attached memory should have been unmapped by now"*
        → fabricvaspaceBatchFree (releases physical pages) (mem_fabric.c:332)
```

Key point: both `cuMemUnmap` and SIGKILL converge on the *same*
per-region unmap function (`fabricvaspaceUnmapPhysMemdesc_IMPL`).
They differ only in how they get there:

* `cuMemUnmap` → `memoryfabricCtrlDetachMem_IMPL` (mem_fabric.c:1110)
  → `_memoryFabricDetachMem(..., bRemoveInterMapping=NV_TRUE)`.
* SIGKILL → resource-server tree walk → `memoryfabricUnmapFrom_IMPL`
  (mem_fabric.c:1271) → `_memoryFabricDetachMem(...,
  bRemoveInterMapping=NV_FALSE)` (the tree walker cleans up the
  RsInterMapping bookkeeping itself).

The `NV_ASSERT(pNode == NULL)` inside the exporter's destructor is
the smoking gun: the destructor of the physical-page owner refuses
to release pages until every peer mapping is torn down. That
assert exists specifically to enforce the "invalidate PTE before
free" invariant.

**Caveat**: NVIDIA issue #272 acknowledges that abrupt SIGKILL
cleanup can misbehave under some load conditions ("without going
through the proper teardown process ... breaks GSP") — those are
GSP-RPC-timeout / stuck-cleanup failure modes, not a violation of
the ordering invariant itself. The invariant is enforced by the
code path above whenever it runs to completion.

## What the UVM MMU code says about ordering

The UVM MMU code (`kernel-open/nvidia-uvm/uvm_mmu.c` and `.h`)
documents the ordering explicitly in its own comments:

- "A CPU membar is needed between the PDE writes and the subsequent
  TLB invalidate."
- "If any of the written PDEs is in sysmem, a sysmembar is needed
  before the TLB invalidate."
- "Wait For Idle" (`uvm_hal_wfi_membar`) is used before invalidating
  the TLB, to stall until previously-issued memory ops have globally
  committed.

Ordering used everywhere: write/clear PTE → memory barrier → TLB
invalidate. **The barrier and the invalidate are separate hardware
primitives.** The invalidate alone (regardless of `PTE_DOWNGRADE`
reason code) does not carry the barrier semantic — this is
consistent with the HW-facing docs cited in the previous section
that separate `TLBI` from `TLBI + SYS_MEMBAR` and from `UFLUSH`.

## Cross-node NVLink drain (revised 2026-07-08 after probe evidence)

For MNNVL peer stores that traverse NVSwitch, the HW-facing docs
say the request is "visible" only when it has reached the remote
GPU's VidL2. `UFLUSH` explicitly covers NVLinks; `TLBI + SYS_MEMBAR`
also provides completion-time flush/membar for GPU clients. A plain
`TLBI` (with any reason code, `PTE_DOWNGRADE` included) is **not**
documented to fence the NVLink pipeline on its own.

**Important correction — when the cross-node protocol actually
fires.** An earlier version of this document claimed the exporter's
`cuMemUnmap` triggers an IMEX two-phase invalidation that clears
peer PTEs on every remote node before the local unmap returns.
That claim was refuted by the P2P probe experiment on
`lyris[0266-0267]` on 2026-07-08: on all 7 peers,
`p2p_ptr_get(dst_rank=2)` kept returning **non-null** for the entire
30-s fault window after rank 2's `cuMemUnmap` returned, including
1 s and 5 s (Approach A CPU-sleep) after. If the two-phase IMEX
broadcast had run on `cuMemUnmap`, the peer's imported PTE would
be gone and `p2p_ptr_get` would return null. It didn't.

So the correct cross-node story is:

1. Exporter's `cuMemUnmap` runs **locally only** on the exporter's
   node — clears the exporter's own peer-facing PTEs with the
   fence-then-invalidate ordering above (`kbusFlush_HAL` +
   `SYS_MEMBAR`/`UFLUSH`-paired TLB flush). This is what drains
   still-in-flight NVLink transactions the exporter's own SMs had
   issued to that VA range. No IMEX message is sent on this path.
2. Peer nodes' PTEs stay valid. Cross-node coordination is deferred
   until an actual physical-page-free is attempted (typically via
   `cuMemRelease` on the exporter after all importers have
   released). At that point the attached-memory tree invariant
   (`NV_ASSERT(pNode == NULL)` in `memoryfabricDestruct_IMPL`)
   holds because IMEX-brokered importer registrations show up in
   the tree and prevent free.
3. The two-phase IMEX APIs listed below exist and do run — but on
   the *importer's* unimport/release path (or on session-level
   teardown), not on every exporter unmap.

So the drain of cross-node NVLink traffic isn't a single "magic"
`PTE_DOWNGRADE` semantic and isn't automatically triggered by a
`cuMemUnmap`. It is a combination of: (i) local fence-then-invalidate
on the exporter's node, (ii) importer refcount keeping physical
pages alive across nodes, and (iii) an IMEX two-phase invalidation
protocol that fires only when the lifecycle actually reaches
page-free.

## What happens for cross-node MNNVL specifically

### What we can say from source (verified)

Cross-node coordination goes through the IMEX (InterNode Memory
Exchange) service. The RM API surface exists at
`src/nvidia/generated/g_imex_session_api_nvoc.h` in the open kernel
modules:

- `imexsessionapiUnmap`
- `imexsessionapiCtrlCmdDisableImporters`
- `imexsessionapiCtrlCmdFinishMemUnimport`

The naming pattern (`Disable...` + `Finish...Unimport`) is idiomatic
for a two-phase begin/commit protocol.

### What we can NOT say from source (not verified — do not overclaim)

We did **not** trace:
- Whether these APIs actually send inter-node messages over the
  IMEX control socket vs. being local-node control operations.
- Whether "Finish" waits for cross-node acks vs. being purely local
  bookkeeping.
- Which caller / lifecycle event invokes
  `imexsessionapiCtrlCmdDisableImporters` and
  `imexsessionapiCtrlCmdFinishMemUnimport`. Given how strongly the
  earlier "cuMemUnmap triggers IMEX broadcast" reading was refuted
  by the P2P probe, we should not repeat the same mistake and claim
  a specific call site without tracing it.
- The IMEX daemon (`nvidia-imex`) itself is closed source and is
  the other side of any inter-node protocol these APIs interact with.

### What we can say from the probe + empirical teardown behavior

- On exporter's `cuMemUnmap`: no cross-node PTE invalidation runs
  (P2P probe evidence: `null=0` on all peers for the full 30-s
  fault window).
- At end-of-test process teardown across 15+ runs: all imports
  release, all exporters release, physical pages actually get
  freed, and no XIDs / IMEX errors / driver panics appear on any
  node. So *some* form of cross-node coordination must be running
  correctly at page-free time — otherwise clean teardown would
  either leak the physical page or fault on stale peer PTEs. This
  is indirect evidence that a working cross-node
  invalidation-before-free path exists in the driver + IMEX
  stack; but it does not by itself confirm the specific APIs
  above are what implement it.

### Honest summary of our current model

* Exporter's `cuMemUnmap`: local-only. Source-verified. Probe-verified.
* Importer refs keep the physical page alive while any importer holds
  the handle. Source-verified (`NV_ASSERT(pNode == NULL)` in
  `memoryfabricDestruct_IMPL`).
* Physical-page free at end-of-test: works cleanly across nodes.
  Empirically verified from teardown behavior.
* The specific mechanism that ensures peer PTEs are gone before the
  physical page is freed at end-of-test — presumably one of
  (a) IMEX two-phase protocol via the APIs above, (b) a different
  code path in `imex/` we didn't trace, or (c) the fact that each
  peer's own process exit tears down its imports independently
  (importers release first, tree empties, then exporter's free is
  safe with no cross-node broadcast needed). We have not
  disambiguated (a) vs (b) vs (c) from source.

For our specific test the importers never release, so this IMEX
two-phase path is never exercised. This is why the negative
observations (no peer XID, no transport-fatal) do not depend on
any IMEX cross-node PTE-broadcast semantics.

## What the GPUDirect RDMA docs say (same invariant, different client)

The GPUDirect RDMA guide describes the same design pattern applied
to third-party HCA drivers (Mellanox nvidia-peermem etc.), and spells
out the contract explicitly:

- The invalidation callback registered at `nvidia_p2p_get_pages()`
  runs SYNCHRONOUSLY when the driver revokes access.
- Inside the callback the third-party driver is expected to WAIT FOR
  OUTSTANDING DMAs TO COMPLETE.
- The corresponding mapped memory areas are only unmapped by the
  NVIDIA driver AFTER the callback returns.

This is exactly the same invariant that the RM enforces internally
between GPUs — same design pattern, different type of client.

## Historical race bug (R515–R535 drivers)

The GPUDirect RDMA docs also note a real bug in this area, fixed in
newer R525/R535 releases:

> "there is a race bug which may show up as a kernel null-pointer
> dereference. This happens when the GPU invokes the (hereby I/O)
> kernel driver invalidation callback ... concurrently with the I/O
> driver calling `nvidia_p2p_put_pages`."

It affected only the GPUDirect RDMA path (HCA to GPU), not the
GPU-to-GPU NVLink path our test uses. It produced a kernel NULL
dereference, not silent data corruption or a transport-layer fatal
— the ordering was fundamentally sound, just poorly synchronized on
concurrent access. Fixed by an API change. Still, this is a
reminder that races in this area do exist historically, and our test
does not systematically stress the concurrent-teardown space where
similar bugs could hide in the current driver.

## Doc situation on the caller for `cuMemUnmap`

The CUDA Driver API reference for `cuMemUnmap` does NOT contain an
explicit "the caller must synchronize in-flight operations first"
line. What it does say is only that the function "exhibits
synchronous behavior for most use cases" and "may return error codes
from previous, asynchronous launches."

So our `cudaDeviceSynchronize()` before `m_rdma_alloc.reset()` is
defensive best-practice, not a documented hard requirement. It
follows the general CUDA convention (sync before freeing anything a
running kernel might touch) and is what NVIDIA sample code that uses
VMM APIs does in practice. It only protects rank 2's own in-flight
kernels; peer-side safety is handled internally by the RM code above.

## Where the driver sits relative to NVLink traffic

Important context: the NVIDIA GPU driver is NOT in the data path.
Once mappings are set up, peer memory access happens as ordinary
`ld.global` / `st.global` PTX ops → GPU MMU → NVLink transaction
engine → NVSwitches → peer GPU → response back. The driver is only
invoked at setup, at teardown, and on fault interrupts. Because the
driver has no per-transaction interception point, its ONLY way to
prevent stale peer access after a resource is freed is to invalidate
the peer PTEs before the free — which is exactly what the RM code
above does.

## Caveats on driver version

The code snippets and behaviour above are from public
open-gpu-kernel-modules revisions in the 6xx driver family (e.g.,
610.43.02, plus recent main-branch tags on GitHub). Our Lyris
cluster runs 580.173. The MNNVL fabric-vaspace and IMEX code paths
have been architecturally stable since Hopper, and the ordering
invariant is a design contract, but the exact function names and
sequence in 580.173 have not been re-verified against a matching
open-source tag. If a driver-version-specific behavioural difference
matters, we would need to inspect the tag corresponding to 580.173
directly.
