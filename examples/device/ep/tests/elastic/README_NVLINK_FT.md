# NVLink Fault-Tolerance Test Guide

How to run and analyze the NVLink fault-tolerance (FT) tests for NIXL EP
elastic contraction. Each test kills one GPU rank (SIGKILL) at one of 15
distinct points — between ops, between send/receive within an op, or
**inside** the dispatch/combine kernel — and verifies the surviving ranks
all:

1. Detect the failure via the runtime mask buffer.
2. Stay correct (no bad partial transfers; per-peer mask checks pass).
3. Finish the run without crashing (all expected survivors print `WORKER DONE`).
4. Clean up — no leaked GPU memory, processes, shared-memory fragments,
   TCPStore/rank-server ports, or stuck NVSMI compute apps.
5. Leave the node reusable: a no-fault baseline immediately after each fault
   timing must still pass.

---

## 1. Files in this directory

| File | Role |
|---|---|
| `elastic.py` | Driver. Spawns N workers that join the elastic job, exchange traffic, and (in fault mode) self-kill at the selected timing. Validates the runtime mask and per-peer correctness on every iteration. |
| `nvlink_fault_tolerance.json` / `nvlink_fault_tolerance_baseline.json` | 1-node victim / no-kill plans. |
| `nvlink_fault_tolerance_2node.json` / `nvlink_fault_tolerance_2node_baseline.json` | 2-node analogues (4 ranks per node; rank 0 on master is the victim). |
| `run_nvlink_fault_tolerance_test.sh` | Single-run launcher for one node, one timing. Runs the 5 cleanup probes around `elastic.py` and emits a `CLEANUP REPORT`. Used by the 1-node sweep. |
| `run_nvlink_fault_tolerance_sweep.sh` | **1-node sweep**. Orchestrates 15 fault runs + a baseline before and after each. Writes `SUMMARY.md` + per-timing logs. |
| `run_nvlink_fault_tolerance_2node_step.sh` | Per-node body for the 2-node sweep. Runs locally in the container on each node and emits a tagged `CLEANUP REPORT [rank=N host=H role=R]`. |
| `run_nvlink_fault_tolerance_2node_sweep.sh` | **2-node sweep**. Stays on the **login node** (the container image lacks SLURM client tools); per timing launches a single `srun -N2 --ntasks-per-node=1` step. Aggregates per-rank logs and runs the same verdict checks as the 1-node sweep. |

The container image `nixl-hybrid-ep-cuda2.sqsh` lives at
`/lustre/fsw/network_research_advdev/lishapira/`, paired with a NIXL EP
install from the `nvlink_fault_tolerance` branch at
`/workspace/lishapira/nixl/install` inside the container (bind-mounted from
lustre; `setup_node.sh` exports the right `PYTHONPATH`/`LD_LIBRARY_PATH`).

---

## 2. The 15 timings

**CPU-level (7)** — host signals SIGKILL between API calls:
`before-dispatch`, `after-dispatch`, `between-dispatch-combine`,
`before-combine`, `after-combine`, `dispatch-between-send-receive`,
`combine-between-send-receive`.

**In-kernel (8)** — GPU marker fires while the kernel is in the marked
phase, host SIGKILLs during the spin window. `4 phases (dispatch-send,
dispatch-recv, combine-send, combine-recv) × 2 host launch modes`:

- `*-during-kernel-no-hook` (`return_recv_hook=False`) — single FUSED
  send+receive kernel.
- `*-during-kernel-hook-separated` (`return_recv_hook=True`) — host
  SPLITS dispatch/combine into two kernels (send then receive) with a
  `hook()` between them.

Both variants of a given (op, phase) cell hit the same kernel-side marker
target id; only the host-side arm location differs.

---

## 3. Prerequisites

### 3.1 SLURM allocation

```bash
salloc -N2 -p gb200 -A network_research_advdev -t 02:00:00 --exclusive
```

One node is enough for the 1-node sweep; two nodes are required for the
2-node sweep.

### 3.2 Container image + bind mount

```bash
export CONTAINER_IMAGE=/lustre/fsw/network_research_advdev/lishapira/nixl-hybrid-ep-cuda2.sqsh
export CONTAINER_MOUNTS=/lustre/fsw/network_research_advdev/lishapira:/workspace/lishapira
```

### 3.3 NIXL EP build

The sweep verifies this at startup; if missing it prints
`BUILD SANITY FAILED -- rebuild from nvlink_fault_tolerance branch.`:

```bash
cd /lustre/fsw/network_research_advdev/lishapira
NIXL_REF=nvlink_fault_tolerance NIXL_PREFIX=/workspace/lishapira/nixl/install \
    bash build_nixl_aarch64.sh
```

Build once per branch HEAD; the install on lustre is reused across nodes.

---

## 4. How to run

### 4.1 Single-node sweep

From inside the salloc shell, on a compute node:

```bash
cd nixl/examples/device/ep/tests/elastic
bash run_nvlink_fault_tolerance_sweep.sh
```

Default: all 15 timings + a baseline before and after each, on 8 ranks.
Output: `./results/<UTC>_<host>/SUMMARY.md` + per-timing logs.

### 4.2 Two-node sweep

Invoke from the **login node** with an active 2-node salloc:

```bash
export SLURM_JOB_ID=<your-job-id>
export SLURM_JOB_NODELIST=<your-nodelist e.g. lyris[0208-0209]>
# plus CONTAINER_IMAGE / CONTAINER_MOUNTS from §3.2
cd nixl/examples/device/ep/tests/elastic
bash run_nvlink_fault_tolerance_2node_sweep.sh
```

Default: 15 timings, 4 ranks per node (8 total), 1 iteration,
`SPIN_CYCLES=100000000`. Output:
`./results/2node_<UTC>_job<ID>_<nodes>/SUMMARY.md`, or override with a
named dir:

```bash
RUN_DIR="$(pwd)/results/15_timing_results_$(date -u +%Y%m%d_%H%M%S)" \
    bash run_nvlink_fault_tolerance_2node_sweep.sh
```

### 4.3 Useful overrides (env vars; apply to both sweeps)

| Env var | Default | When to change |
|---|---|---|
| `TIMINGS` | all 15 | Restrict to a subset for a smoke test or to bisect. |
| `ITERATIONS` | `1` | Repeat each timing 3–5x to look for flakiness. |
| `SPIN_CYCLES` | `100000000` | Raise if `MISSED_IN_KERNEL_TIMING` appears (widens the GPU spin window). |
| `SETTLE_SECONDS` | `5` | Raise if cleanup intermittently reports leftover processes / GPU memory deltas. |
| `MEM_LEAK_MIB` | `64` | GPU memory delta tolerance. |
| `RUN_DIR` | auto-named | Set a named dir to separate experimental runs. |
| `FAULT_KILL_SIGNAL` | `sigkill` | Switch to `sigterm` for graceful-termination tests. |
| `NUM_PROCESSES` | 8 (1-node) / 4 (per-node, 2-node) | Worker count. |

---

## 5. Output layout

```
SUMMARY.md                                            ← top-level verdict table (PASS/FAIL per timing + Overall)
build_probe.log                                       ← container-side nixl_ep import probe
initial_baseline.log                                  ← pre-sweep no-fault run
<timing>__iter<N>.log                                 ← combined fault-run log (both ranks' stdout)
<timing>__iter<N>__rank0_master_<host>.log            ← per-rank log (2-node only)
<timing>__iter<N>__rank1_worker_<host>.log            ← per-rank log (2-node only)
<timing>__iter<N>__post_baseline.log                  ← post-fault no-fault baseline
evidence/in_kernel_fault_rank<R>_<timing>_pid<P>.log  ← marker evidence (in-kernel timings only)
```

Alongside `RUN_DIR` you also get `RUN_DIR.console.log` (orchestrator stdout/stderr).

---

## 6. Reading SUMMARY: criteria, where each lives, and how failure looks

`SUMMARY.md` has a section per timing × iteration. The bottom-line
`Overall:` is `PASS` if every row was `PASS`. Each row's `Verdict` is
`PASS` or `FAIL(reason1, reason2, ...)`. The criteria, their log sources,
and the failure-tag form:

| Criterion | Where in logs | Failure tag | First thing to check |
|---|---|---|---|
| Fault exit code | last orchestrator line | `fault_rc=<nonzero>` | survivor crashed / run timed out — grep `Traceback`/`AssertionError` in `<timing>__iter<N>.log` |
| Survivors completed | `[rank R] WORKER DONE survived=true` count | `survivors=A/B` (A<B) | a survivor never finished — likely tracebacks, mask mismatch, or a hang |
| Survivor tracebacks | `Traceback`/`AssertionError`/`RuntimeError` in run log | `tracebacks=N` | read the first one |
| Mask never falsely flags an alive rank | `MASK CHECK SUMMARY passes=K calls=K` per survivor | `mask=A/B!` | first `MASK CHECK FAILED` line names the offending bit |
| Mask saw the kill | `[rank R] MASK DETECTED rank=K timestamp_ns=…` | (drives `Mask propagation latency` field; not a separate gate) | — |
| In-kernel marker hit the marked phase | `evidence/in_kernel_fault_rank<R>_<timing>_pid<P>.log` → `verdict=HIT_IN_KERNEL_WINDOW`, `target=<id>` | `hit=MISSED` | raise `SPIN_CYCLES` |
| | | `hit=TIMEOUT` | marker never fired — usually a build mismatch (rebuild from `nvlink_fault_tolerance`) |
| | | `hit=NO_EVIDENCE` | evidence-dir path bug; check `EVIDENCE_DIR_CONTAINER` |
| | | `target=MISMATCH(want=X,got=Y)` | marker fired in the wrong phase — bug |
| Cleanup (5 probes, per node) | `CLEANUP REPORT [rank=N host=H role=R]` → `cleanup_result: CLEAN`/`DIRTY (...)` | `cleanup=master:DIRTY(...)` / `cleanup=worker:DIRTY(...)` | the REPORT block lists which probe was dirty and the delta |
| Post-fault baseline | `<timing>__iter<N>__post_baseline.log` exits 0 with expected survivors | `post_baseline=rc=<x> survivors=<a/b>` | node is wedged — usually a cleanup miss or stuck process |
| Build sanity | `build_probe.log` at sweep startup | `BUILD SANITY FAILED` (fatal — sweep aborts) | rebuild from `nvlink_fault_tolerance` (§3.3) |

### The 5 cleanup probes (per node)

The step / launcher snapshots node state before and after `elastic.py`
and the REPORT compares them:

1. **GPU memory used (MiB)** — delta within `MEM_LEAK_MIB`.
2. **Leftover processes** matching `elastic.py` / `rank_server` / NIXL
   workers — must be 0 after the settle wait.
3. **`/dev/shm/torch_*` and `cuda.shm.*`** — must not have grown.
4. **TCP ports** for the rank server / TCPStore (9999, 10000) — must be released.
5. **NVSMI compute apps** on the GPUs — must be 0.

In the 2-node sweep, all 5 run on **both** master and worker; both must
report `CLEAN`.

### Two environmental failure modes that look like FAIL but aren't FT bugs

- **TCPStore connect refused** on the worker before any `[rank R]` line —
  stale port from a prior interrupted run; cleanup probe will report
  non-CLEAN ports; rerunning the timing resolves it.
- **Worker-startup death `exit=1` with no stdout** — unhealthy node;
  rerun on a fresh allocation.

---

## 7. Brief in-kernel kill mechanism

A pinned host-mapped 12-int marker buffer (`csrc/nixl_ep.hpp:53-67`)
couples the GPU and host:

- **Arm (host)** — `enable_in_kernel_fault_marker(target, sequence, spin_cycles)`
  zeroes all slots and writes `target` (1=dispatch-send, 2=dispatch-recv,
  3=combine-send, 4=combine-recv), a per-process `sequence > 0`, and
  `spin_cycles`.
- **Fire (kernel)** — at the start of each phase the kernel calls
  `maybe_mark_in_kernel_fault_enter`. If `marker[target] == call's
  hard-coded target`, it writes `marker[entered_idx]=sequence` and
  busy-waits `spin_cycles` GPU clocks. At phase end it writes
  `marker[exited_idx]=sequence`.
- **Kill (host helper thread)** — polls every 10 µs. First poll with
  `marker[entered_idx] >= sequence`: if `marker[exited_idx] < sequence`
  → kernel still in the phase → `HIT_IN_KERNEL_WINDOW` and
  `os.kill(pid, SIGKILL)`. Otherwise → `MISSED_IN_KERNEL_TIMING`.

`SPIN_CYCLES` must be large enough that one poll interval (~10 µs) +
signal delivery fits inside the window. The default `100000000` works on
B200/H100/GB200.
