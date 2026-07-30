# SIGTERM / SIGKILL / cuMemUnmap teardown call chain

Exact call chain that executes when a rank is killed or has its
peer-exposed memory unmapped in the NIXL-EP fault-injection tests
(`nvlink_fault_tolerance` branch).

**Elastic uses VMM fabric memory exclusively.** `elastic.py:1091` calls
`buffer.update_memory_buffers(...)` without `num_nvl_bytes`, which
defaults to 0 (`nixl_ep/buffer.py:785`). Every branch in
`Buffer::destroy()` gated by `if (num_nvl_bytes > 0)` is dead code for
us: `intranode::barrier`, `cudaIpcCloseMemHandle`, and `cudaFree` on
the NVL buffer NEVER RUN in an elastic teardown. The chains below show
only what actually executes.

Three teardown paths:

1. **SIGTERM cooperative** (`fault_kill_signal="sigterm"`)
2. **SIGKILL / crash** (`fault_kill_signal="sigkill"`, segfault, OOM-kill)
3. **cuMemUnmap fault-injection** (`fault_inject_mode="unmap-mid-flight"`, victim survives)

---

## 1. SIGTERM cooperative teardown

```
os.kill(pid, SIGTERM)   ← self_kill or CPU-timed injector
   │
   ▼
handle_sigterm(...)                                 elastic.py:119
   │
   ├─ rank_client.release_rank(...)                 elastic.py:130-133
   │    → TCP write to master's RankServer
   │    → survivors' runtime mask bit for this rank flips to 1
   │
   ├─ buffer.destroy()                              elastic.py:135
   │    ▼
   │    Buffer::destroy()                           nixl_ep.cpp:266
   │      │
   │      ├─ cudaDeviceSynchronize()                          line 286
   │      │    lets peers' in-flight NVLink reads/writes drain
   │      │
   │      ├─ _nixl_ep_destroy()                               line 288
   │      │    frees intranode/internode scratch state
   │      │
   │      ├─ [if num_nvl_bytes > 0]  ← NOT REACHED in elastic
   │      │
   │      ├─ agent->invalidateLocalMD()                       line 306
   │      │    only if NIXL_ETCD_ENDPOINTS is set; removes this
   │      │    rank's UCX MD entry from the etcd rendezvous
   │      │
   │      ├─ agent->deregisterMem(rdma_reg_descs)             line 309
   │      │  agent->deregisterMem(sync_reg_descs)             line 313
   │      │  agent->deregisterMem(sync_count_reg_descs)       line 317
   │      │    each calls into UCX ucp_mem_unmap:
   │      │      - cuda_ipc TL: releases each peer's fabric-import contribution
   │      │      - rc_gda TL:   ibv_dereg_mr + drops IB rkey
   │      │    then ucp_ep_destroy for every peer endpoint
   │      │    → peers see NIXL_ERR_REMOTE_DISCONNECT on next op
   │      │
   │      ├─ m_rdma_alloc.reset()                             line 330
   │      │  m_mask_alloc.reset()                             line 332
   │      │  m_sync_alloc.reset()                             line 334
   │      │  m_sync_count_alloc.reset()                       line 336
   │      │    each unique_ptr<vmm_region>::reset() fires
   │      │    ~vmm_region() -> vmm_region::release()
   │      │    (csrc/vmm.cpp:45-66) which does, in order:
   │      │      cuMemUnmap(va, size)         clear this process's PTE
   │      │      cuMemAddressFree(va, size)   release VA reservation
   │      │      cuMemRelease(handle)         drop exporter's refcount
   │      │    Physical page freed ONLY when refcount hits zero AND
   │      │    every importer has propagated its cuMemRelease back
   │      │    (via IMEX for cross-node importers).
   │      │
   │      ├─ cudaFreeHost(in_kernel_fault_marker)             line 352
   │      ├─ m_workspace_alloc.reset()                        line 356
   │      └─ destroyed = true                                 line 359
   │
   ├─ signal.signal(SIGTERM, SIG_DFL)                 elastic.py:139
   └─ signal.raise_signal(SIGTERM)                    elastic.py:140
        → default handler exits the process; every VMM object
          was already released cleanly above.
```

### Peer effects under SIGTERM

* UCX endpoint error on peer's `ep → victim` connection surfaces as
  `NIXL_ERR_REMOTE_DISCONNECT` on next op.
* Peer's imported `m_rdma_alloc` VA still resolves to live physical
  pages until the refcount hits zero (peer's own `cuMemRelease` after
  UCX ep destroy contributes to that count).
* Rank server tells survivors the victim released → mask flips → next
  dispatch/combine skips the victim.
* No XID typically observed on either side, because
  `cudaDeviceSynchronize()` drained in-flight peer ops before we
  invalidated anything.

---

## 2. SIGKILL / crash teardown

Python's signal handler cannot catch SIGKILL, so **none** of the
user-space cleanup above runs. All teardown is done by the kernel and
the NVIDIA driver's involuntary path.

```
SIGKILL delivered
   │
   ▼
Linux kernel: do_exit(SIGKILL)
   │
   └─ exit_files() drops /dev/nvidia* fds
        │
        ▼
NVIDIA driver release path (open-gpu-kernel-modules):
   │
   ├─ RmClient teardown for this pid                 rmapi/client.c
   │    enumerates every fabric-memory allocation, VMM handle,
   │    imported handle, and legacy IPC handle this pid owned.
   │
   ├─ For each VMM physical allocation this pid owned:
   │    fabricvaspaceUnmapPhysMemdesc_IMPL(...)      mem_fabric.c
   │      - clears exporter's fabric-visible PTEs
   │      - flushes TLB
   │      - decrements physical-page refcount
   │
   ├─ For each remote importer of this pid's fabric memory:
   │    memoryfabricDestruct_IMPL / IMEX destroy signal
   │      → IMEX daemon on the peer node invalidates the peer's
   │        cuMemImportFromShareableHandle mapping
   │      (inferential: memoryfabricDestruct_IMPL is source-verified
   │       in open-gpu-kernel-modules/mem_fabric.c; the IMEX daemon
   │       side is not open source, so the cross-node propagation is
   │       inferred from documented behavior + empirical evidence,
   │       not directly source-verified — see DRIVER_FINDINGS.md.)
   │
   └─ RmClient freed; /dev/nvidia* fd close returns
```

### Peer effects under SIGKILL

* Same eventual outcome as SIGTERM (UCX ep error + mask flip), but:
  * **No `rank_client.release_rank`** → survivors only detect via UCX
    keepalive/timeout, so detection latency is strictly longer.
  * **No cooperative UCX teardown** → survivors' endpoints to victim
    dangle until UCX gives up.
  * **No pre-teardown sync** → a peer's already-issued NVLink store
    can arrive after the driver has cleared the exporter's PTE. That
    is when we do sometimes see XID 31 on the peer's GPU (MMU fault at
    the exporter's now-empty translation).
* Refcount invariant still holds. Peer's imported mapping is either
  invalidated by IMEX before the store lands, or resolves to the still-
  alive physical page (physical free is deferred until the last
  importer's release propagates back).

---

## 3. cuMemUnmap fault-injection (unmap-mid-flight)

Victim rank **survives**. The injection tears down only the RDMA
allocation (`m_rdma_alloc`); the other vmm_regions
(`m_mask_alloc`, `m_sync_alloc`, `m_sync_count_alloc`,
`m_workspace_alloc`) stay live, and NIXL/UCX endpoints stay live. No
signal, no rank release, no `Buffer::destroy()`, no UCX deregister,
no etcd invalidate.

```
Injection trigger (CPU-timed OR in-kernel marker)
   │
   ▼
_do_inject_unmap(buffer, ...)                       elastic.py
   │
   └─ buffer.inject_unmap_fault(...)
        ▼
        Buffer::inject_unmap_fault()                nixl_ep.cpp:1347
          │
          │  NO cudaDeviceSynchronize -- we WANT this to hit while
          │  peer kernels have in-flight stores. Comment at
          │  nixl_ep.cpp:1366 spells this out.
          │
          └─ m_rdma_alloc.reset()                   nixl_ep.cpp:1370
                fires ~vmm_region() -> vmm_region::release()
                (csrc/vmm.cpp:45-66), which runs in order:
                  cuMemUnmap(va, size)         clear exporter's PTE
                  cuMemAddressFree(va, size)   release VA reservation
                  cuMemRelease(handle)         drop exporter's refcount
                  ← ONLY the exporter's contributions are dropped.
                  ← Peers' imported PTEs are NOT touched.
                  ← Physical page stays alive as long as any peer
                    still holds an import (IMEX-mediated refcount > 0).
                  ← UCX endpoints not deregistered (nixl_agent_info
                    still alive; only the physical/VA slot for the
                    RDMA region goes away).
                  ← No explicit IMEX destroy notification is sent
                    from user-space; peer nodes keep their imports
                    until their own cuMemRelease runs.
          │
          └─ records injection timestamp + verdict
             (TRUE_IN_KERNEL_UNMAP vs LATE_UNMAP), returns
```

### Peer effects under cuMemUnmap-only path

* Peer's imported PTE is **unaffected**: peer's PTE maps peer's VA
  directly to the physical page via NVSwitch/NVLink; exporter's PTE
  invalidation does not remove peer's PTE.
* Peer's GPU stores through that VA **still succeed** — they walk
  peer's PTE → fabric → still-alive physical page (kept alive by
  peer imports contributing to refcount). This is why we never
  observed peer-side NVLink transport-layer XIDs (74 / 79 / 154).
* **The XID 31 we do observe** is on the *victim's* GPU: the victim's
  OWN kernels (dispatch/combine still executing on the victim's rank)
  hit the now-invalid exporter PTE for the same region, and the driver
  reports that MMU fault as XID 31 attributed to the victim.
* Mask flip: the victim did NOT release its rank, so survivors' mask
  never flips victim's bit. `fault_inject_pass_gate.py` uses
  `unmap_interrupted_live_comm = TRUE_IN_KERNEL_UNMAP AND
  nixl_msgs_interrupted > 0` to decide whether the injection landed
  inside a live comm window.

> The P2P probe (`p2p_ptr_get() non-null after unmap`) does NOT
> validate peer PTE liveness — it's pure `remote_address +
> mapped_offset` arithmetic baked in at UCX registration time and
> returns the same value regardless of mapping state. See
> `WHY_NO_NVLINK_TRANSPORT_FAULT.md`.

---

## What each path skips relative to full SIGTERM cleanup

| Step | SIGTERM | SIGKILL | cuMemUnmap |
|---|---|---|---|
| Python `handle_sigterm` | ✓ | ✗ | ✗ |
| `rank_client.release_rank` | ✓ | ✗ | ✗ |
| `Buffer::destroy()` (user-space) | ✓ | ✗ | ✗ |
| `cudaDeviceSynchronize()` pre-teardown | ✓ | ✗ | ✗ |
| `agent->deregisterMem` (UCX/RDMA) | ✓ | ✗ (peers dangle) | ✗ |
| `agent->invalidateLocalMD` (etcd) | ✓ (if `NIXL_ETCD_ENDPOINTS`) | ✗ | ✗ |
| `~vmm_region()`: `cuMemUnmap` + `cuMemAddressFree` + `cuMemRelease` on ALL allocs | ✓ | driver-forced via IMEX | ✓ but ONLY on `m_rdma_alloc` |
| Exporter `cuMemUnmap` on peer-exposed region | ✓ (via dtor) | driver-forced | ✓ |
| `~vmm_region()` on `m_mask_alloc`/`m_sync_alloc`/`m_sync_count_alloc`/`m_workspace_alloc` | ✓ | driver-forced | ✗ (survive) |
| Process exit | ✓ | ✓ | ✗ (victim survives) |
| `intranode::barrier` / `cudaIpcCloseMemHandle` / `cudaFree` (num_nvl_bytes > 0 block) | ✗ in elastic | ✗ in elastic | ✗ |

## File references

| File | Function | Line |
|---|---|---|
| `elastic.py` | `handle_sigterm` | 119-140 |
| `elastic.py` | `self_kill`, `_do_inject_unmap`, `maybe_schedule_*` | see file |
| `nixl_ep/buffer.py` | `Buffer.destroy`, `Buffer.inject_unmap_fault`, `update_memory_buffers` | 780+ |
| `csrc/nixl_ep.cpp` | `Buffer::destroy` | 266-361 |
| `csrc/nixl_ep.cpp` | `Buffer::inject_unmap_fault` | 1347-1375 |
| `csrc/vmm.cpp` | `vmm_region::release`, `~vmm_region` | 45-70 |
| `csrc/vmm.cpp` | `vmm_region::vmm_region` (fabric alloc path) | 72-175 |
| open-gpu-kernel-modules | `_memoryFabricAttachMem`, `memoryfabricDestruct_IMPL` | `mem_fabric.c` |
| open-gpu-kernel-modules | `fabricvaspaceUnmapPhysMemdesc_IMPL` | `vaspace_api.c` |
| open-gpu-kernel-modules | RmClient teardown on fd close | `rmapi/client.c` |
