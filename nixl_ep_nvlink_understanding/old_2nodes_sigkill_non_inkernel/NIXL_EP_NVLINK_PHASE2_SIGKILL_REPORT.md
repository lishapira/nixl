# NIXL EP NVLink SIGKILL Test Summary

## Goal

Validate NIXL EP fault tolerance on 2 NVLink/NVSwitch-connected nodes when one rank is killed with `SIGKILL`. Success means healthy ranks continue correct dispatch/combine communication, detect/mask the failed rank, finish with `done`, cleanup leaves no GPU/process leftovers, and a clean baseline passes afterward.

## Setup

- Repo: `git@github.com:lishapira/nixl.git`
- Branch/commit: `nvlink_understanding` / `de5e607`
- Nodes: `theia0153`, `theia0156`
- Run dir: `/lustre/fsw/network_research_advdev/lishapira/nixl/nixl_ep_nvlink_understanding/old_2nodes_sigkill_non_inkernel`
- Results dir: `/lustre/fsw/network_research_advdev/lishapira/nixl/nixl_ep_nvlink_understanding/old_2nodes_sigkill_non_inkernel/results`
- Container setup: every run sourced `/workspace/lishapira/setup_node.sh`
- Topology evidence: 4x GB300 per node, local `NV18`, fabric `Switch Connected`, `Completed / Success`, same `ClusterUUID`

## Plan

```json
[
  [0, 1, 2, 3],
  [0, 1, -2, 3, 4, 5, 6, 7],
  [0, 1, 3, 4]
]
```

Rank `2` is killed in the fault phase.

## Scripts

- `run_timing_sweep.sh`: runs all SIGKILL timings, cleanup, and post-fault baselines.
- `run_elastic_pair.sh`: launches `elastic.py` on both nodes with timeout handling.
- `cleanup_nodes.sh`: kills leftovers and records `nvidia-smi` process/memory state.
- `expansion_contraction_no_fault_rank2.json`: clean no-fault baseline plan.

## Command

```bash
RUN_DIR=/lustre/fsw/network_research_advdev/lishapira/nixl/nixl_ep_nvlink_understanding/old_2nodes_sigkill_non_inkernel
"$RUN_DIR/run_timing_sweep.sh" > "$RUN_DIR/results/timing_sweep.log" 2>&1
```

Fault runs used:

```bash
--fault-kill-signal sigkill --fault-kill-timing <timing>
```

## Results

| Timing | Fault rc | Post baseline | Result |
|---|---:|---:|---|
| `before-dispatch` | 124 | 0 | PASS |
| `after-dispatch` | 124 | 0 | PASS |
| `between-dispatch-combine` | 124 | 0 | PASS |
| `dispatch-between-send-receive` | 124 | 0 | PASS |
| `before-combine` | 124 | 0 | PASS |
| `combine-between-send-receive` | 124 | 0 | PASS |
| `after-combine` | 124 | 0 | PASS |

`124` is the external timeout wrapper result for SIGKILL fault runs, not a per-rank exit code.

## Evidence

- Healthy ranks produced `Dispatch + combine bandwidth`.
- All non-killed ranks logged `done`.
- Surviving ranks detected failed rank `{2}`.
- Hook timings logged `return_recv_hook=True`.
- No `AssertionError`, `Traceback`, `RuntimeError`, CUDA error, or Xid.
- Cleanup showed no leftover compute processes.
- Initial, all post-fault, and final no-fault baselines passed.

## Conclusion

**PASS** for 2-node NIXL EP elastic-level NVLink/NVSwitch SIGKILL boundary/hook coverage.

## Gaps / Next Steps

- This proves NVLink/NVSwitch fabric evidence, not NVL72 specifically.
- Existing timings are not true in-kernel send/receive kill coverage.
- Add GPU entered/exited markers and CPU helper SIGKILL for dispatch send/receive and combine send/receive.
- Run the same coverage on 4 nodes.

## Directory Note

`common/topology_examples/abandoned_topology_only_theia0193_theia0195_20260522_090939` is the earlier abandoned allocation. It has topology evidence only, not the completed SIGKILL sweep.
