# Driver-side references for the NVLink fault-injection tests

Companion to `EXPERIMENT_UNMAP_FAULT.md` (test plumbing) and
`WHY_NO_NVLINK_TRANSPORT_FAULT.md` (result interpretation). This file
records the specific driver source references and doc snippets that
back the two claims in the WHY doc:

* Reason 1 — driver-cooperative teardown drains peer-facing PTEs
  before physical pages can be freed.
* Reason 2 — `cuMemUnmap` / SIGKILL are caller-local for MNNVL fabric
  memory; peers keep their own imported PTEs pointing at a still-live
  physical page.

Everything below is readable in the public
[`NVIDIA/open-gpu-kernel-modules`](https://github.com/NVIDIA/open-gpu-kernel-modules)
repository (dual-license GPL+MIT — this IS the kernel-space `nvidia.ko`
running on the Lyris nodes, driver 580.173). Function names are stable
across recent tags; line numbers drift.

## Where the driver sits relative to NVLink traffic

Important context: the NVIDIA GPU driver is NOT in the data path. Once
mappings are set up, peer memory access happens as ordinary
`ld.global` / `st.global` PTX → GPU MMU → NVLink transaction engine →
NVSwitches → peer GPU. The driver is only invoked at setup, at
teardown, and on fault interrupts. Because the driver has no
per-transaction interception point, its ONLY way to prevent stale
peer access after a resource is freed is to invalidate the peer PTEs
before the free — which is exactly what the RM code below does.

## Where `cuMemUnmap` and SIGKILL converge in the RM

Both fault-injection paths converge on the same per-region unmap
function `fabricvaspaceUnmapPhysMemdesc_IMPL` in
`src/nvidia/src/kernel/mem_mgr/fabric_vaspace.c`.

### via `cuMemUnmap` (source-verified)

    cuMemUnmap                                    (user-space)
     → libcuda.so                                 (closed source)
     → ioctl                                      (/dev/nvidia*)
     → memoryfabricCtrlDetachMem_IMPL             (mem_fabric.c:1110)
     → _memoryFabricDetachMem(bRemoveInterMapping=NV_TRUE)  (mem_fabric.c:147)
     → fabricvaspaceUnmapPhysMemdesc_IMPL         (fabric_vaspace.c)
        ├─ vaspaceUnmap            (clear peer PTEs)
        ├─ kbusFlush_HAL           (bus flush)
        └─ fabricvaspaceInvalidateTlb(..., PTE_DOWNGRADE)
                                    (+ SYS_MEMBAR/UFLUSH-class fence,
                                     see PTE_DOWNGRADE note below)

### via SIGKILL / process death (source-verified)

When a process dies (SIGKILL or any other abrupt exit), the OS closes
its `/dev/nvidia*` fds and the NVIDIA driver's `.release` file-op
runs the RM cleanup:

    RmFreeUnusedClients                                (real NVIDIA kernel log line)
     → rmapiFreeClientListWithSecInfo
     → Nv01FreeClientList
     → clientFreeResource_IMPL / rmclientFreeResource_IMPL  (client.c)
     → serverInitFreeParams_Recursive
     → serverFreeResourceTree                          (rs_server.c)
     → (walks the dying client's resource tree; for every peer PTE
        mapping into this context's fabric memory it invokes:)
         → serverInterUnmap                            (rs_server.c)
         → rmclientInterUnmap_IMPL                     (client.c:374-397)
         → resUnmapFrom                                (vtable dispatch)
         → memoryfabricUnmapFrom_IMPL                  (mem_fabric.c:1271)
         → _memoryFabricDetachMem(bRemoveInterMapping=NV_FALSE)  (mem_fabric.c:147)
         → fabricvaspaceUnmapPhysMemdesc_IMPL          (same as cuMemUnmap)
     → (then per-resource destructor:)
         → memoryfabricDestruct_IMPL                   (mem_fabric.c:899)
             ├─ NV_ASSERT(pNode == NULL) on pAttachMemInfoTree  (mem_fabric.c:340)
             │     comment: "Every attached memory should have been unmapped by now"
             └─ fabricvaspaceBatchFree                 (mem_fabric.c:332)

The `NV_ASSERT(pNode == NULL)` inside the exporter's destructor is
the smoking gun: the destructor of the physical-page owner refuses
to release pages until every attached mapping (from any importer,
local or remote) is torn down. That assert exists specifically to
enforce the "invalidate peer PTE before free" invariant.

**Difference between the two entry points**: `cuMemUnmap` calls
`_memoryFabricDetachMem(bRemoveInterMapping=NV_TRUE)`; SIGKILL calls
it with `bRemoveInterMapping=NV_FALSE` because the resource-server
tree walker cleans up the `RsInterMapping` bookkeeping itself. The
peer-PTE work — `fabricvaspaceUnmapPhysMemdesc_IMPL` — is identical
in both paths.

## What `PTE_DOWNGRADE` actually is (and is NOT)

Earlier internal notes claimed that `PTE_DOWNGRADE` alone "fences
in-flight accesses". This is imprecise; the corrected picture is:

* `PTE_DOWNGRADE` is the *reason code* of the GMMU TLB-invalidate — it
  tells the invalidate engine "this is a permission-removal invalidate"
  rather than a "we changed a PTE, please refresh" invalidate. It does
  NOT by itself carry drain-until-visibility semantics.
* The drain semantics come from the companion `SYS_MEMBAR` / `UFLUSH`
  primitives. Per NVIDIA-internal HW docs (GSP / "Detect memory
  subsystem hang from Kernel RM"):
  * `TLBI` (no membar) — "ensures the GMMU is not blocked". Not a fence.
  * `UFLUSH` — sysmembar + flushes pending writes from FB to their
    final locations across FBHUB, HSHUBs, VidL2s, SysL2 and **NVLinks**.
    Full fabric-scope flush.
  * `TLBI + NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_SYS_MEMBAR=TRUE` —
    TLBI plus a FLUSH/Membar at TLBI completion covering GPC + HUB
    clients. HW/RM comment summary: "A TLBI + sysmembar waits for all
    memory references (reads, writes, greds, gatoms, ...) to reach a
    point of visibility." This is the operation providing the drain
    semantics we rely on.

So the invariant "in-flight peer traffic is drained before physical
pages are freed" comes from the *combination* of the explicit ordering
in the RM code (`vaspaceUnmap` → `kbusFlush_HAL` → TLBI → free), the
`SYS_MEMBAR`/`UFLUSH`-paired invalidate, and the ordering rules
documented in the UVM MMU code — not from `PTE_DOWNGRADE` alone.

## What the UVM MMU code says about ordering (source-verified)

`kernel-open/nvidia-uvm/uvm_mmu.c` / `.h` documents the ordering
explicitly in its own comments:

* "A CPU membar is needed between the PDE writes and the subsequent
  TLB invalidate."
* "If any of the written PDEs is in sysmem, a sysmembar is needed
  before the TLB invalidate."
* "Wait For Idle" (`uvm_hal_wfi_membar`) is used before invalidating
  the TLB, to stall until previously-issued memory ops have globally
  committed.

**Direct evidence that explicit ordering is required** — UVM issue
UVM-326: "Use `MEMBAR.SYS` for PTE downgrades to local vidmem". The
report describes a case where TLB invalidate + `MEMBAR.GL` was *not*
sufficient because `membar.gl` "doesn't wait for probe"; the driver
could unmap/free a physical page P, reallocate it, and a prior GPU
access could then observe the new contents. The stated fix was to
switch to `MEMBAR.SYS` when future access to that page can come from
the CPU or a peer GPU. That is the direct driver-side rule for MNNVL /
peer-visible memory: the fence on the downgrade path must be
system-scope, not just GPU-local.

## Cross-node (MNNVL) behaviour

Empirically demonstrated by the P2P probe in this branch (see
`WHY_NO_NVLINK_TRANSPORT_FAULT.md` for the numbers):

* On the exporter's `cuMemUnmap`: no cross-node PTE invalidation runs
  on peer nodes. Every peer's `p2p_ptr_get(dst_rank=victim)` keeps
  returning non-null for the full 30-s fault window. If the driver
  had shipped a cross-node PTE-invalidation broadcast on unmap, the
  peer's imported PTE would be gone and the probe would return null;
  it does not.

**What that means for the correctness invariant across nodes.** The
"peer PTE cannot outlive its physical page" invariant is enforced at
the *physical-page-free* boundary, not at the unmap boundary. The
victim's `cuMemUnmap` is local (clears the victim's own PTEs, drains
the victim's own NVLink pipeline). The physical page cannot actually
be freed while any importer — local or remote — is still holding it,
because the attached-memory tree in `memoryfabricDestruct_IMPL` is
non-empty (the `NV_ASSERT(pNode == NULL)` at `mem_fabric.c:340`).
Cross-node importer entries land in that tree via
`_memoryFabricAttachMem` on the exporter's node whenever an importer
maps in via IMEX-brokered `cuMemImportFromShareableHandle`. So the
cross-node invariant carrier is the ref-counted attached-memory tree,
enforced at free time, not at unmap time.

### IMEX APIs — honest bounds

The IMEX (Internode Memory Exchange) RM surface has a plausible
two-phase begin/commit protocol shape:

    imexsessionapiCtrlCmdDisableImporters
    imexsessionapiCtrlCmdFinishMemUnimport
    imexsessionapiUnmap

(from `src/nvidia/generated/g_imex_session_api_nvoc.h`)

But we did NOT trace which caller / lifecycle event invokes these APIs,
whether they send inter-node messages, or whether "Finish" waits for
cross-node acks. The IMEX daemon (`nvidia-imex`) is closed source. So
we do not overclaim: the two-phase IMEX protocol may well be what
enforces cross-node PTE-teardown at physical-page-free time, but our
test never reaches physical-page-free (importers keep their refs for
the entire 30-s fault window), so we did not exercise it and we do not
depend on its exact semantics for the results in the WHY doc.

The negative outcome ("no peer XID, no transport-fatal") therefore
holds regardless of whether the IMEX cross-node PTE broadcast exists,
because Reason 2 alone (importer refcount keeping physical pages
alive; peer's imported PTE never invalidated on this path) is
sufficient to prevent a peer-side MMU fault.

## What we could NOT verify from source

* `libcuda.so` (userspace CUDA driver) is closed. We only see the
  kernel-side ioctl entry point it ends up calling.
* GSP firmware (which physically executes the TLB flush on
  Blackwell/GB200) ships as opaque binary blobs. The kernel driver
  builds the RPC packet in the open code cited above; execution on
  GSP is not source-visible.
* IMEX daemon is closed. See "IMEX APIs — honest bounds" above.
* The exact 580.173 tag of `open-gpu-kernel-modules` (line numbers
  above are from R580/R610 tags; function names are stable). If a
  driver-version-specific behavioural difference matters we would need
  to inspect the tag corresponding to 580.173 directly.
