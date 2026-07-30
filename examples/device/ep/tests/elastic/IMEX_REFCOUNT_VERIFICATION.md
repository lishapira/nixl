# IMEX cross-node VMM fabric refcount — empirical verification

## Question

When the *exporter* of a VMM fabric-handle allocation on **node A**
releases (either voluntarily via `cuMemRelease` or involuntarily via
`SIGKILL` → driver's `memoryfabricDestruct_IMPL`), and the *importer* on
**node B** still holds a live `cuMemImportFromShareableHandle`
reference, does the physical page on node A stay alive until node B
also releases?

Equivalently: does IMEX propagate the importer's refcount contribution
across nodes so node A's driver-side allocator knows that node B still
has a live reference to the page?

## Why this matters

Everything in `TEARDOWN_CALL_CHAIN.md`, `DRIVER_FINDINGS.md`, and
`WHY_NO_NVLINK_TRANSPORT_FAULT.md` was relying on the assumption that
this cross-node refcount is real. In particular, the elastic SIGTERM
teardown path and the `Buffer::inject_unmap_fault` code path both
depend on it: if the assumption is wrong, peer stores through the
imported PTE could hit reused/freed physical pages and corrupt data
silently, and the `unmap-mid-flight` fault-inject test would not be
producing the failure semantics we think it is.

Before these experiments the cross-node claim was **inferential**
(source-verified on the same-node path in `open-gpu-kernel-modules`,
but the IMEX daemon side of cross-node propagation is not open-source).

## Simplest verification — 2 processes, both death modes

Before running the full 3-process poisoning experiments below, there
is a much simpler smoke check that confirms the core claim ("cross-node
refcount holds when the survivor is the ONLY remaining ref-holder
anywhere") in about a minute.

### What it tests

- Only 2 processes total (1 victim on node A, 1 survivor on node B).
- Both death modes tested back-to-back in one run:
  - `release` — victim runs `cuMemUnmap + cuMemAddressFree +
    cuMemRelease` (same call chain as elastic's SIGTERM path
    `Buffer::destroy() → ~vmm_region()`).
  - `sigkill` — victim self-SIGKILLs; the driver runs involuntary
    `memoryfabricDestruct_IMPL` when the kernel closes
    `/dev/nvidia*` fds (same path elastic's
    `--fault-kill-signal=sigkill` fault-injector triggers).
- After signaling the victim, survivor waits **15 s** to give OS reap
  + driver kernel cleanup + IMEX destroy propagation ample time to
  settle.
- Then survivor does PROOF #1 (initial read), PROOF #2 (read still
  returns `SENTINEL_1`), PROOF #2b (write `SENTINEL_2`, read back).
- **No PROOF #3** (no poisoner process). This variant therefore
  cannot distinguish "refcount holds" from "physical page not yet
  recycled by the driver"; for the strongest claim run Experiment 1 /
  Experiment 2 below.

### Runtime evidence that survivor is the only ref-holder

CUDA VMM does **not** expose a public API to query the physical-page
refcount from user space. What we CAN verify at runtime:

1. **Victim is really dead** —
   - `release` mode: victim sends a `RELEASED` ack over TCP before
     exiting; survivor observes it.
   - `sigkill` mode: TCP connection breaks abruptly when the kernel
     tears down victim's sockets; survivor detects EOF.
2. **No other importer exists** — by construction of the test (only
   two processes total, only the survivor ever calls
   `cuMemImportFromShareableHandle`; no third process, no UCX/NIXL
   registration on top).

Given (1) + (2), the survivor is provably the only ref contributor
left after victim's teardown. PROOF #2 and PROOF #2b succeeding are
then the runtime empirical evidence that this single cross-node ref is
being honored by node A's driver.

### Files

| File | Purpose |
|---|---|
| `simple_refcount_test.py` | Roles `victim` and `survivor`. Reuses CUDA/TCP helpers from `imex_refcount_test.py` (same directory). |
| `run_simple_refcount_test.sh` | Runs BOTH death modes back-to-back on the same 2-node allocation, on separate TCP ports. |

### Result

Run `simple_refcount_20260730_022815` on `lyris[0151,0159]`, job 2534993
(same MNNVL clique, driver 580.173.02 / CUDA 13.0, AArch64):

```
=========================================================
VARIANT: death_mode=release  port=27184
=========================================================
[SURVIVOR] PROOF #1  PASS: reads SENTINEL_1=0xa1a1... (1,048,576 u64s)
[SURVIVOR] sending DIE to victim (mode=release)
[SURVIVOR] victim confirmed clean release complete
           (cuMemUnmap + cuMemAddressFree + cuMemRelease done)
[SURVIVOR] waiting 15s before probing
[SURVIVOR] PROOF #2  PASS: after victim RELEASE + 15s wait,
           survivor STILL reads SENTINEL_1 -> refcount HOLDS
[SURVIVOR] PROOF #2b PASS: survivor can WRITE + read-back SENTINEL_2
[release] PASS

=========================================================
VARIANT: death_mode=sigkill  port=27185
=========================================================
[SURVIVOR] PROOF #1  PASS: reads SENTINEL_1
[SURVIVOR] sending DIE to victim (mode=sigkill)
[SURVIVOR] victim SIGKILL confirmed: TCP connection lost as expected.
           Kernel has closed /dev/nvidia* fds and driver has run
           involuntary memoryfabricDestruct_IMPL for the victim's
           fabric allocation.
[SURVIVOR] waiting 15s before probing
[SURVIVOR] PROOF #2  PASS: after victim SIGKILL + 15s wait,
           survivor STILL reads SENTINEL_1 -> refcount HOLDS
[SURVIVOR] PROOF #2b PASS: survivor can WRITE + read-back SENTINEL_2
[sigkill] PASS

COMBINED RESULT
=========================================================
  release:PASS
  sigkill:PASS
[simple-refcount] OVERALL PASS: cross-node refcount HOLDS under all
                                tested death modes
```

Full logs:
`/lustre/fsw/network_research_advdev/lishapira/runs/simple_refcount_20260730_022815/`.

### How to run

```bash
NODE_A=<node1> NODE_B=<node2> SLURM_JOB_ID=<jobid> \
  bash nixl/examples/device/ep/tests/elastic/run_simple_refcount_test.sh <jobid>
```

Env knobs: `MODES` (default `"release sigkill"`), `PORT_BASE`
(default 27184), `SIZE_BYTES`, `PROBE_BYTES`, `DEATH_WAIT`
(default 15), `DEVICE`, `TCP_TIMEOUT`, `HARD_TIMEOUT`, `OUT`.

### Trade-off vs Experiment 1 / 2 below

| Aspect | Simple (this section) | Experiments 1 & 2 |
|---|---|---|
| Processes | 2 | 3 (adds a poisoner on node A) |
| Death modes | `release` + `sigkill` in one run | one mode per script |
| Proofs | #1, #2, #2b | #1, #2, #2b, **#3** (poisoning) |
| Can distinguish "refcount holds" vs "page not yet recycled"? | No | **Yes** (via PROOF #3) |
| Runtime | ~1 min (both modes) | ~30–60 s each |
| Use when | Quick smoke test | Strongest empirical claim |

## Reference model and the key invariant

For a VMM fabric allocation with one importer:

| Step | Local ref (node A) | Remote ref (node B via IMEX) | Total refcount |
|---|---|---|---|
| `cuMemCreate(FABRIC)` on A | 1 | 0 | 1 |
| `cuMemImportFromShareableHandle(FABRIC)` on B | 1 | 1 | 2 |
| **Exporter releases (or dies)** on A | 0 | 1 | **1**   ← page still alive **iff cross-node ref counts** |
| `cuMemRelease` on B (importer) | 0 | 0 | 0   ← page finally freed |

The row in bold is the whole point of these tests. **At that moment
there is exactly one reference contributor to the page, and it is on a
different node from where the physical memory lives.** If IMEX does
not propagate that reference into node A's driver refcount, node A's
allocator sees refcount = 0 and frees / recycles the page — which
would show up as either faults on the importer's next read, or as
silent data corruption when the recycled page gets used by some later
allocation on node A.

## Test artifacts

| File | Purpose |
|---|---|
| `imex_refcount_test.py` | Single Python script (`ctypes` + `libcuda.so.1`, no torch or NIXL dependency for the CUDA calls themselves). Supports 5 roles for Experiments 1 & 2 below. |
| `simple_refcount_test.py` | 2-role (`victim` / `survivor`) subset for the simple 2-process check above. Reuses helpers from `imex_refcount_test.py`. |
| `run_simple_refcount_test.sh` | Runs the simple 2-process check in BOTH death modes back-to-back. |
| `../../../../scripts/run_imex_refcount_test.sh` | 2-srun orchestrator for **Experiment 1** (voluntary release, with poisoning). |
| `../../../../scripts/run_imex_refcount_sigkill_test.sh` | 3-srun orchestrator for **Experiment 2** (involuntary SIGKILL, with poisoning). |

Roles supported by `imex_refcount_test.py`:

| Role | Where | What it does |
|---|---|---|
| `exporter` | node A | Alloc + write SENTINEL_1 + export + wait + `cuMemUnmap` + `cuMemAddressFree` + `cuMemRelease` + `--poison-count` new fabric allocs |
| `importer` | node B | Import + PROOFs #1/#2/#2b/#3 (exp. 1) |
| `victim` | node A | Alloc + write SENTINEL_1 + export + wait for DIE_NOW + `os.kill(getpid(), SIGKILL)` |
| `poisoner` | node A (SEPARATE process from victim) | After victim is dead, wait for signal, then `--poison-count` fabric allocs |
| `importer_sigkill` | node B | Import + coordinate victim + poisoner + PROOFs #1/#2/#2b/#3 (exp. 2) |

Both experiments use identical proofs; only the exporter-side teardown
mechanism differs.

## The 4 PROOFs (both experiments)

Each proof is a device-side read/write through the importer's mapping.
The importer writes to the region with a distinct u64 pattern and reads
it back after `cuCtxSynchronize()`, comparing against the expected
value across every u64 in the probed range.

| # | When | Expected | What it establishes |
|---|---|---|---|
| 1 | After importer's initial map | reads SENTINEL_1 | Cross-node fabric import path itself works |
| 2 | After exporter release / SIGKILL | reads STILL SENTINEL_1 | Refcount held through exporter teardown |
| 2b | After importer writes SENTINEL_2 | reads SENTINEL_2 | Mapping is fully live (not stale-cached) |
| 3 | After node-A allocates N (default 16) new fabric regions of the same size and fills each with a distinct CANARY pattern | reads STILL SENTINEL_2 | Physical page was NOT recycled by node A's allocator — refcount is genuinely holding, not just "driver hasn't gotten around to freeing" |

PROOF #3 is the discriminating one. Without it, "still SENTINEL_1" from
PROOF #2 could mean "refcount held" OR "page is freed but happens to
still contain the old bytes because nothing has overwritten them yet".
The poisoning specifically forces the node-A allocator to churn through
1+ GiB of new fabric allocations; if the victim's page were free, one
of those would almost certainly land on it and the importer would read
a `0xC0DE0000...` CANARY value instead of SENTINEL_2.

## Experiment 1 — voluntary release

### Setup

- 2 nodes: `lyris0013` (exporter), `lyris0014` (importer), same MNNVL
  clique, driver 580.173.02 / CUDA 13.0, AArch64.
- One process per node, both use CUDA device 0.
- `SIZE_BYTES` = 64 MiB (rounded to 2 MiB fabric granularity, so 64 MiB
  as-is), `PROBE_BYTES` = 8 MiB, `POISON_COUNT` = 16 (1 GiB of poison).

### Sequence

```
exporter (node A)                        importer (node B)
─────────────────                        ─────────────────
cuMemCreate(FABRIC)
cuMemAddressReserve
cuMemMap
cuMemSetAccess
write SENTINEL_1 = 0xA1A1...
cuMemExportToShareableHandle
listen TCP :27182                        connect TCP → node A:27182
send META + 64-byte handle  ─────────►
                                         cuMemImportFromShareableHandle
                                         cuMemAddressReserve + Map + setAccess
                                         PROOF #1: read == SENTINEL_1 ✓
BARRIER importer_ready
cuMemUnmap                               ← exporter drops entire ref
cuMemAddressFree                         chain (voluntary release path)
cuMemRelease
BARRIER exporter_released
                                         PROOF #2: read == SENTINEL_1 ✓
                                         write SENTINEL_2 = 0xB2B2...
                                         PROOF #2b: read == SENTINEL_2 ✓
BARRIER importer_proof2_done
POISONING: 16 × cuMemCreate(FABRIC)
  each written with 0xC0DE_XXXX
BARRIER poisoning_done
                                         PROOF #3: read == SENTINEL_2 ✓
BARRIER importer_proof3_done
release poisons                          cuMemUnmap + cuMemAddressFree
                                         + cuMemRelease
BARRIER exporter_cleanup_done
```

### Result

Run `imex_refcount_20260730_014004` on `lyris[0013,0014]`, job 2534718:

```
[EXPORTER] granularity=2,097,152  size=67,108,864  probe_bytes=8,388,608
[EXPORTER] primary: handle=0x94118f0 va=0x631200000 writing SENTINEL_1=0xa1a1a1a1a1a1a1a1
[EXPORTER] EXPORTER releasing primary (cuMemUnmap + cuMemAddressFree + cuMemRelease)
[EXPORTER] POISONING: allocating 16 new fabric regions ...
[EXPORTER] poison[0..15] handle=0x9413970..0x11457bb0

[IMPORTER] imported: handle=0x9411ec0 mapped at va=0x631200000
[IMPORTER] PROOF #1  PASS: importer reads SENTINEL_1=0xa1a1... (1,048,576 u64s)
[IMPORTER] PROOF #2  PASS: after exporter cuMemRelease, importer STILL reads SENTINEL_1 -> refcount held
[IMPORTER] PROOF #2b PASS: importer read-after-write == SENTINEL_2 -> mapping fully live
[IMPORTER] PROOF #3  PASS: after exporter's poisoning allocations, importer still reads SENTINEL_2
                          -> physical page was NOT reused -> cross-node refcount HOLDS

OVERALL: PASS - cross-node VMM fabric refcount HOLDS
```

Full logs: `/lustre/fsw/network_research_advdev/lishapira/runs/imex_refcount_20260730_014004/`.

## Experiment 2 — involuntary SIGKILL

### Setup

Same 2 nodes / driver / sizes as Experiment 1, but now **three**
processes: the exporter is split into `victim` and `poisoner`, both on
node A but as fully independent OS processes with independent CUDA
contexts (each does its own `cuInit` + `cuCtxCreate`). This is exactly
what makes the test valid — see "Refcount picture at the critical
moment" below.

### Sequence

```
victim (node A, TCP :27182)              importer (node B)
─────────────────────────                ─────────────────
cuMemCreate(FABRIC) + write SENTINEL_1
export handle
listen :27182                            connect → victim:27182
send META + HANDLE          ──────────►
                                         cuMemImportFromShareableHandle
                                         PROOF #1: read == SENTINEL_1 ✓
                             ◄─ DIE_NOW ─ send DIE_NOW
os.kill(getpid(), SIGKILL)               sleep 3s ← lets the kernel run
   ↳ kernel: exit_files() drops             exit_files() and IMEX process
     /dev/nvidia* fds → driver's            the destroy signal
     RmClient teardown runs:
       fabricvaspaceUnmapPhysMemdesc_IMPL
       memoryfabricDestruct_IMPL
       → IMEX destroy signal to node B
     (driver-forced version of
      cuMemUnmap+AddressFree+Release)
                                         PROOF #2: read == SENTINEL_1 (?)
                                         write SENTINEL_2
                                         PROOF #2b: read == SENTINEL_2 (?)

poisoner (node A, TCP :27183)
─────────────────────────
listen :27183                            connect → node A:27183
                             ◄────────── send META (size, probe_bytes)
                                         BARRIER start_poison ──►
cuInit + cuCtxCreate  ← fresh context
POISONING: 16 × cuMemCreate(FABRIC)
                                         ◄── BARRIER poison_done
                                         PROOF #3: read == SENTINEL_2 (?)
                                         BARRIER cleanup ──►
release poisons                          cuMemUnmap + AddressFree + Release
```

### Result

Run `imex_refcount_sigkill_20260730_014756` on `lyris[0013,0014]`,
job 2534718:

```
[VICTIM]   pid=1040826  primary: handle=0x94aebc0 va=0x631200000 writing SENTINEL_1
[VICTIM]   exported fabric handle, listening on :27182
[VICTIM]   waiting for DIE_NOW from importer
[VICTIM]   received DIE_NOW; killing self pid=1040826 with SIGKILL
srun: error: lyris0013: task 0: Killed         ← kernel confirms SIGKILL delivered

[POISONER] pid=1040827  listening for importer on 0.0.0.0:27183
[POISONER] barrier 'start_poison' released     ← waits until victim is dead
[POISONER] POISONING: allocating 16 new fabric regions ...
[POISONER] poison[0..15] handle=0x94ae960..0x10cf34e0
[POISONER] POISONING done: 16 allocations
[POISONER] POISONER exiting rc=0

[IMPORTER] imported: handle=0x94af150 mapped at va=0x631200000
[IMPORTER] PROOF #1  PASS: importer reads SENTINEL_1=0xa1a1... (1,048,576 u64s)
[IMPORTER] sending DIE_NOW to victim
[IMPORTER] waiting 3s for victim to be killed and reaped
           (driver runs memoryfabricDestruct_IMPL / IMEX destroy propagation)
[IMPORTER] PROOF #2  PASS: after victim SIGKILL, importer STILL reads SENTINEL_1
           -> refcount held through involuntary driver teardown
[IMPORTER] PROOF #2b PASS: importer read-after-write == SENTINEL_2
           -> mapping fully live after victim SIGKILL
[IMPORTER] PROOF #3  PASS: after poisoner allocations, importer still reads SENTINEL_2
           -> physical page NOT reused
           -> cross-node refcount HOLDS through victim's SIGKILL involuntary teardown

OVERALL: PASS - cross-node VMM fabric refcount HOLDS under exporter SIGKILL
```

Timing (from importer's log):
* `01:48:14` DIE_NOW sent → victim `os.kill(getpid(), SIGKILL)`.
* `01:48:14 → 01:48:17` 3-second grace period. Kernel completes
  `exit_files()` → `/dev/nvidia*` fd close → driver's `RmClient`
  teardown runs `memoryfabricDestruct_IMPL` /
  `fabricvaspaceUnmapPhysMemdesc_IMPL`, IMEX propagates destroy
  signal.
* `01:48:17` PROOF #2 succeeds: importer still reads `SENTINEL_1`. So
  the physical page was NOT freed during the involuntary teardown —
  IMEX must have told node A's allocator that node B still holds a
  ref.
* `01:48:20–01:48:21` Poisoner (separate process on node A, fresh
  CUDA context created only here) allocates 16 × 64 MiB new fabric
  regions with distinct `0xC0DE_00XX` canaries. All 16 succeed.
* `01:48:22` PROOF #3 succeeds: importer still reads `SENTINEL_2`.
  None of the 16 poison allocations landed on the victim's "released"
  physical page.

Full logs: `/lustre/fsw/network_research_advdev/lishapira/runs/imex_refcount_sigkill_20260730_014756/`.

## Refcount picture at the critical moment

This is the question the user asked directly: "after the exporter
dies, is there only one reference to the memory (the importer's), and
is it on a different node?"

**Yes — by construction of both experiments.**

### Who could hold a ref?

For a fabric allocation, a reference to the physical page can exist
only from:
1. The process that called `cuMemCreate(FABRIC)` (exporter/victim).
2. Any process anywhere that called `cuMemImportFromShareableHandle`
   for that specific handle. Local imports contribute to the local
   refcount; remote (cross-node) imports contribute via IMEX.

### Who does hold a ref in these tests?

| Experiment | Ref contributor | Node | Contribution type |
|---|---|---|---|
| Both | exporter/victim (before its release/death) | A | local (from `cuMemCreate`) |
| Both | importer | B | **remote, via IMEX-propagated `cuMemImportFromShareableHandle`** |

That is the complete list. There is no third holder because:

- **No UCX / NIXL registration** in this test. `imex_refcount_test.py`
  is pure `ctypes` on top of `libcuda.so.1`; it never touches
  `ucp_mem_map` or `nixl_agent->registerMem`. So UCX doesn't hold a
  parallel ref through some other TL.
- **No other importer on node A.** Even in the SIGKILL scenario, the
  poisoner is a fully independent process on node A that **never
  imports the victim's handle**. Poisoner only does its own new
  `cuMemCreate(FABRIC)` calls. Those return brand-new physical pages,
  each with a fresh refcount of 1 owned by the poisoner. Poisoner
  is *not* a reference to the victim's page.
- **No other importer on node B.** Just the one importer process.

### After the exporter dies

- Local (node A) ref chain contributed by the exporter → **gone**
  (either explicit `cuMemRelease` in Experiment 1, or driver-forced
  release via `RmClient` teardown in Experiment 2).
- Remote (node B) ref chain contributed by the importer → **still
  present**. This is the sole surviving reference and it lives on a
  different node from the physical memory.

If cross-node refcount is a real thing, node A's allocator must see
this remaining B-side ref and refuse to recycle the page. That's what
the PROOFs test. **Both experiments confirm: yes, it does.**

## What this establishes

Both experiments passed all 4 proofs on the same 2 nodes (`lyris0013`
+ `lyris0014`), same MNNVL clique, same driver 580.173.02 / CUDA
13.0, AArch64:

Positive (Experiment 1 — voluntary release):

- Cross-node VMM fabric refcount is real for the **voluntary release**
  path.
- The refcount is enforced strongly enough that node A's fabric
  allocator will NOT recycle the released page for 1 GiB of subsequent
  new fabric allocations, as long as node B still holds an import.
- Peers can not only read but **write** through their imported mapping
  after the exporter releases (PROOF #2b), which means the mapping
  isn't in some degraded read-only state either.

Positive (Experiment 2 — involuntary SIGKILL):

- Same guarantees hold under **involuntary** driver teardown. This
  covers `SIGKILL`, segfault, OOM-kill, and any other death where the
  application never got to run its own `cuMemRelease`. In all these
  cases the kernel closes the process's `/dev/nvidia*` fds, the
  driver's `RmClient` teardown runs
  `memoryfabricDestruct_IMPL` / `fabricvaspaceUnmapPhysMemdesc_IMPL`,
  and IMEX propagates a destroy signal to peer nodes. Our test proves
  this kernel-driver path **also** respects the cross-node refcount —
  node A does not physically free the page while node B still has an
  import.
- This is directly relevant to elastic's `sigkill` fault-injection
  mode and to any real crash of a peer in a multi-node run: the
  refcount contract holds identically to the polite-teardown case, so
  survivors are not exposed to silent data corruption through their
  still-mapped imports.

The two experiments together give us a **strong empirical basis** for
every claim in `TEARDOWN_CALL_CHAIN.md`, `DRIVER_FINDINGS.md`, and
`WHY_NO_NVLINK_TRANSPORT_FAULT.md` that depended on cross-node
refcount propagation. Claims that were previously marked "inferential"
in those docs can now be upgraded to "empirically verified on Lyris,
driver 580.173.02, CUDA 13.0".

## What this does NOT establish

- Behavior when there are **multiple** importers on multiple nodes
  (not tested; refcount contract predicts the page survives until all
  importers release, but this test only exercises the 1-importer case).
- Behavior when the exporter is on a **different clique / different
  MNNVL fabric** from the importer (this test requires same clique,
  verified by `mnnvl_probe.py`).
- Behavior under adversarial timing (e.g., exporter release
  interleaved with peer's in-flight NVLink stores) — this is a
  *before/after* refcount test, not an in-flight fault-injection test.
  That is what `Buffer::inject_unmap_fault` in the elastic runs tests
  separately.
- Anything about UCX or NIXL cleanup — this test deliberately bypasses
  both to isolate the VMM refcount question.

## How to reproduce

Get a 2-node allocation in the same MNNVL clique (verify with
`python3 mnnvl_probe.py` — should see identical `cluster_uuid` and
`clique_id` across both nodes), then from the login-node shell
(outside any container):

**Simple 2-process check (both death modes, ~1 min):**

```bash
NODE_A=<node1> NODE_B=<node2> SLURM_JOB_ID=<jobid> \
  bash nixl/examples/device/ep/tests/elastic/run_simple_refcount_test.sh <jobid>
```

**Experiment 1 (voluntary release, with poisoning):**

```bash
NODE_A=<node1> NODE_B=<node2> SLURM_JOB_ID=<jobid> \
  bash /lustre/fsw/network_research_advdev/lishapira/scripts/run_imex_refcount_test.sh <jobid>
```

**Experiment 2 (SIGKILL, with poisoning):**

```bash
NODE_A=<node1> NODE_B=<node2> SLURM_JOB_ID=<jobid> \
  bash /lustre/fsw/network_research_advdev/lishapira/scripts/run_imex_refcount_sigkill_test.sh <jobid>
```

Env knobs (Experiments 1 & 2): `PORT`, `POISONER_PORT` (exp. 2 only),
`SIZE_BYTES`, `PROBE_BYTES`, `POISON_COUNT`, `DEVICE`, `TCP_TIMEOUT`,
`VICTIM_DEATH_GRACE` (exp. 2 only), `HARD_TIMEOUT`, `OUT`.

Both runners write per-role logs under `OUT/` and print an overall
`PASS`/`FAIL` line at the end. Exit code = importer's exit code
(0 = all 4 proofs pass).

## Related docs

- `TEARDOWN_CALL_CHAIN.md` — the exact call chains for SIGTERM /
  SIGKILL / `cuMemUnmap` cleanup. This doc's Experiment 2 verifies
  the SIGKILL chain.
- `DRIVER_FINDINGS.md` — driver-source references for
  `_memoryFabricAttachMem`, `memoryfabricDestruct_IMPL`,
  `fabricvaspaceUnmapPhysMemdesc_IMPL` and the ref-counting invariant.
  This doc's experiments are the empirical complement to those
  source-based claims.
- `WHY_NO_NVLINK_TRANSPORT_FAULT.md` — the peer-side story for why
  we never observed XID 74/79/154 during unmap-mid-flight. The
  reasoning relies on the cross-node refcount holding, which is what
  these experiments verify directly.
- `EXPERIMENT_UNMAP_FAULT.md` — the in-flight fault-injection test
  suite; conceptually orthogonal to the refcount tests here but shares
  much of the mental model.
