# NIXL EP: NVLink fault-injection tests (2-node MNNVL)

Two fault-injection modes selectable via `--fault-inject-mode`, each
runnable at any of 15 kill timings via `--fault-kill-timing`:

* **`sigkill`** — the victim rank sends itself `SIGKILL`; the Linux
  driver's `.release` file-op cleans up its GPU resources. Pre-existing
  path, unchanged in behaviour by this branch — only wired into the new
  runner and observability plumbing.
* **`unmap-mid-flight`** — the victim rank calls the CUDA driver's
  memory-management teardown on its own live peer-exposed RDMA buffer
  and stays alive at Python level:

      cuMemUnmap(ptr, size)
      cuMemAddressFree(ptr, size)
      cuMemRelease(handle)

  Peers still hold their own imported handles into that fabric memory.
  The victim's GMMU no longer has a valid translation for that VA; the
  victim's own next SM access on the freed range takes an MMU fault
  (`XID 31` in the victim host's kernel log, `cudaErrorIllegalAddress`
  at Python level). NIXL-EP's per-message NVLink timeout on peers hits
  after `DEFAULT_TIMEOUT_MS = 30 000` ms, peers mark rank 2 dead in the
  mask, and the plan proceeds without it.

Goal: exercise NIXL-EP's timeout-based elastic-recovery under both a
process-death fault (`sigkill`) and a driver-cooperative memory-teardown
fault (`unmap-mid-flight`), and confirm the MNNVL fabric is not
corrupted by either.

## What the branch changes

### C++ (`examples/device/ep/csrc/`)

* `nixl_ep.cpp` / `.hpp` — `Buffer::inject_unmap_fault()` resets the
  `std::unique_ptr<vmm_region>`; the destructor calls the three CUDA
  APIs above in the correct order. **No `cudaDeviceSynchronize()`** at
  the top of the injector (see comment in `nixl_ep.cpp`): syncing would
  serialize with the marker-armed dispatch/combine kernel, forcing
  `cuMemUnmap` to always fire post-kernel and reducing every in-kernel
  timing to the `before-dispatch` case.
* Extended `InKernelFaultMarkerIndex` with 3 slots for the **P2P probe**
  (`IN_KERNEL_P2P_PROBE_TARGET` + null/non-null atomic counters); the
  probe target is sentinel-inited to `IN_KERNEL_P2P_PROBE_DISABLED = -1`
  in `Buffer::init` and only P2P slots survive across
  `enable_in_kernel_fault_marker` resets. New Buffer methods
  `set_p2p_probe_target` / `reset_p2p_probe_counts`, all pybind-exposed.
* `kernels/nixl_ep_ll.cu` — `maybe_probe_p2p_ptr_null()` runs
  immediately after every `p2p_ptr_get(dst_rank)` in the dispatch/combine
  send warps; if the armed probe target matches `dst_rank`, `atomicAdd`
  into the null or non-null slot. Non-null on a peer after the victim's
  fault == the peer's own imported P2P mapping into the victim's fabric
  memory is still valid — rank 2 stays a "valid NVLink destination"
  from the peer's GPU perspective. See `WHY_NO_NVLINK_TRANSPORT_FAULT.md`.

### Python driver (`examples/device/ep/tests/elastic/elastic.py`)

* `--fault-inject-mode {sigkill,unmap-mid-flight}`.
* `--in-kernel-fault-spin-cycles` — GPU spin-cycle budget inside the
  in-kernel marker; used with `*-during-kernel-*` timings so the host
  thread's fault injection returns while the kernel is still in the
  marked send/recv window. Cold-path `cuMemUnmap` teardown can be ~26 ms;
  `200_000_000` (~100 ms on GB200) reliably lands as `TRUE_IN_KERNEL_UNMAP`.
  Smaller values may collapse to `LATE_UNMAP`.
* `--p2p-probe-target <victim_rank>` — arms the P2P probe on every
  non-victim rank; each send warp bumps the null or non-null counter
  per NVLink store. `_log_p2p_probe_snapshot` prints one
  `P2P_PROBE_COUNTS tag=post-dispatch|post-combine null=<N> nonnull=<M>`
  line per iteration per peer to master/worker logs.
* Post-injection re-check of the in-kernel marker's exited slot →
  emits `TRUE_IN_KERNEL_UNMAP` or `LATE_UNMAP`, so the pass gate can
  triangulate "unmap returned while the kernel was still actively
  writing to peers over NVLink".
* `try/finally` around the worker loop so `summary.json` always flushes
  even when the victim's poisoned CUDA context raises `AcceleratorError`.

### Test harness (`examples/device/ep/tests/elastic/`)

* `fault_artifacts.py` — per-rank `ArtifactCapture`. NVLink
  error-counter sampler (`nvidia-smi nvlink --errorcounters` @ ~500 ms),
  best-effort `dmesg` snapshot (usually permission-denied on Lyris),
  IMEX log fragment (`/var/log/nvidia-imex-verbose.log`), `summary.json`
  with `role`, `inject_mode`, `inject_event_ts`, `xid_seen`,
  `imex_error_count`, `recovered`, `extra.exception`.
* `phase5_mnnvl_probe.py` — parses `nvidia-smi -q`'s Fabric block into
  `{host, gpus:[{gpu, uuid, cluster_uuid, clique_id, fabric_state}]}`.
  Two nodes are MNNVL-coupled iff every GPU on both nodes reports the
  same `(cluster_uuid, clique_id)`.
* `phase5_node_probe.sh` — per-node combined PRE/POST probe (MNNVL +
  NVLink counter CSV + POST-only `torch.cuda` health probe).
* `phase5_pass_gate.py` — MNNVL PRE/POST identical + victim
  `summary.json` present + peer `summary.json` present + at least one
  observability signal: `xid_seen` on a peer, `imex_error_count > 0` on
  a peer, positive NVLink-counter delta, victim `extra.exception`
  contains `illegal memory access`, `MASK DETECTED dead_rank=<victim>`
  in master/worker log, or a per-message NIXL-EP timeout with
  `src_rank=<victim>`. Bonus derived signal
  `unmap_interrupted_live_comm` = `TRUE_IN_KERNEL_UNMAP AND
  nixl_ep_msgs_interrupted > 0`.
* `run_phase5_2node.sh` — 4-srun orchestrator (PRE probe → master →
  worker → POST probe → pass gate). Selects mode via env vars.
* `nvlink_fault_tolerance_2node_unmap.json` — 2 nodes × 4 ranks × 2
  phases; victim `rank 2`, present in both phases (surviving-victim
  semantics of `unmap-mid-flight`).

## How to run

From an active `--exclusive` `salloc` across two MNNVL-coupled GB200
nodes (Lyris `gb200-backfill` partition, etc.):

    cd examples/device/ep/tests/elastic

    # (1) unmap-mid-flight, at a CPU-level phase boundary (~2 min)
    FAULT_INJECT_MODE=unmap-mid-flight \
    TIMING=before-dispatch \
    P2P_PROBE_TARGET=2 \
        bash run_phase5_2node.sh

    # (2) unmap-mid-flight, mid-kernel — the strongest race
    #     (TRUE_IN_KERNEL_UNMAP verdict expected)
    FAULT_INJECT_MODE=unmap-mid-flight \
    TIMING=dispatch-send-during-kernel-no-hook \
    IN_KERNEL_SPIN_CYCLES=200000000 \
    P2P_PROBE_TARGET=2 \
        bash run_phase5_2node.sh

    # (3) sigkill, same knobs (P2P probe stays armed to compare
    #     hardware-view behaviour vs. unmap-mid-flight)
    FAULT_INJECT_MODE=sigkill \
    TIMING=dispatch-send-during-kernel-no-hook \
    IN_KERNEL_SPIN_CYCLES=200000000 \
    P2P_PROBE_TARGET=2 \
        bash run_phase5_2node.sh

`TIMING=` accepts 7 CPU-level timings (`before-dispatch`,
`after-dispatch`, `between-dispatch-combine`,
`dispatch-between-send-receive`, `before-combine`,
`combine-between-send-receive`, `after-combine`) and 8 in-kernel timings
(`{dispatch,combine}-{send,receive}-during-kernel-{no-hook,hook-separated}`).

Results land under `results/phase5_2node_<UTC>_<host1>_<host2>_<timing>/`.
The runner prints `PHASE5 PASS: ...` on success and exits non-zero on
any gate failure.

## What "PASS" means

Any signal in the following list satisfies the "fault was really
exercised" gate; a healthy `unmap-mid-flight` in-kernel run typically
surfaces the first four:

* `victim_local_illegal_address` — the victim's own next CUDA op raised
  `cudaErrorIllegalAddress` (matches `XID 31` on the victim's host).
* `nixl_mask_detected_dead_rank_<victim>` — every peer's
  30-s NIXL-EP mask timeout fired; elastic recovery is exercised.
* `nixl_ep_<N>_msgs_interrupted` — how many per-message NVLink writes
  from the victim never landed on peers (with `TRUE_IN_KERNEL_UNMAP`
  this is the strongest available proof the unmap returned mid-transfer).
* `unmap_returned_during_live_nvlink_comm` — combination of the above
  two: `TRUE_IN_KERNEL_UNMAP AND nixl_ep_msgs_interrupted > 0`.
* MNNVL clique identical PRE and POST.

For hardware-view evidence — "did rank 2 stay a valid NVLink
destination from every peer's GPU perspective?" — grep the master/worker
logs for `P2P_PROBE_COUNTS`. Every peer, every read, in every run of
both fault modes: `null=0` and non-null counts grow into the hundreds.
See `WHY_NO_NVLINK_TRANSPORT_FAULT.md` for the reasoning and the full
probe results table.

## Empirical bounds tested on this branch

| Fault mode | Timing | Runs | Peer XID? | Peer P2P `null≠0`? | Peer `WORKER DONE survived=true`? | Mask detected on all peers? |
|---|---|---|---|---|---|---|
| `cuMemUnmap` | 4 CPU-level timings + `TRUE_IN_KERNEL_UNMAP` in-kernel run | 10+ | No | No | Yes | Yes |
| `SIGKILL` | `before-dispatch` (P2P probe) + `dispatch-send-during-kernel-no-hook` (`HIT_IN_KERNEL_WINDOW`) | 2 | No | No | Yes | Yes |

Observability caveat: container-visible `dmesg.log` is 0-byte in every
run (`dmesg_restrict=1`, no `CAP_SYSLOG`). "No peer XID" means "no
peer XID visible to any Python-observable signal". The cluster
monitoring bot (which does read `dmesg`) reported `XID 31` only on the
victim's host, never on peer hosts, corroborating the container-side
signal.

## Conclusion

* Both fault modes reliably fire and are cleanly recovered by
  NIXL-EP's mask-timeout elastic recovery.
* Neither produces an NVLink transport-layer fatal
  (`XID 74/79/154`, contain-and-drain, NVLink counter delta, IMEX
  `[ERROR]`, MNNVL clique change) — see `WHY_NO_NVLINK_TRANSPORT_FAULT.md`
  for the mechanism and `DRIVER_FINDINGS.md` for the driver source
  references that back it.
* The P2P probe directly measures the "hardware view": every peer, every
  read, in every run of both fault modes, `p2p_ptr_get(dst_rank=2)`
  returned non-null — rank 2 remained a valid NVLink destination from
  every peer's GPU perspective for the full 30-s fault window.
