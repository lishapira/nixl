# One-Node In-Kernel SIGKILL Smoke Summary

Scope: one-node GB200 smoke test only. This validates the new in-kernel marker/SIGKILL machinery on one node; it is not the final two-node NVLink/NVSwitch fault result.

Run directory:

`/lustre/fsw/network_research_advdev/lishapira/nixl/nixl_ep_nvlink_understanding/1node_sigkill_inkernel`

Setup:

- Node: `lyris0145`
- Allocation: `1995953`
- GPUs: 4x NVIDIA GB200
- Intra-node topology: GPUs connected by `NV18`
- Branch: `nvlink_understanding`
- Commit: `4087c88 nixl_ep tests: add in-kernel SIGKILL markers`
- Excluded commit: `c8d5254 nixl_ep tests: log elastic mask snapshots`
- Build: `build_rc=0`
- Spin cycles: `100000000`

Plan:

- Fault plan: `one_node_fault_rank2.json`
- Baseline plan: `one_node_baseline.json`
- Rank killed: `2`

Results:

- `dispatch-send-during-kernel`: `HIT_IN_KERNEL_WINDOW`, `exited_before_sigkill=0`, fault rc `0`, post baseline rc `0`
- `dispatch-receive-during-kernel`: `HIT_IN_KERNEL_WINDOW`, `exited_before_sigkill=0`, fault rc `0`, post baseline rc `0`
- `combine-send-during-kernel`: `HIT_IN_KERNEL_WINDOW`, `exited_before_sigkill=0`, fault rc `0`, post baseline rc `0`
- `combine-receive-during-kernel`: `HIT_IN_KERNEL_WINDOW`, `exited_before_sigkill=0`, fault rc `0`, post baseline rc `0`
- Initial baseline rc `0`
- Final baseline rc `0`

Validation notes:

- No `MISSED_IN_KERNEL_TIMING` or `IN_KERNEL_MARKER_TIMEOUT` found.
- No `Traceback`, `AssertionError`, `RuntimeError`, `CUDA error`, or `Xid` found in logs.
- Fault logs show healthy ranks detect failed rank `{2}`, continue communication, and finish with `done`.
- Cleanup logs show no leftover compute apps; GPU memory remained at low idle levels.

Conclusion:

The one-node GB200 smoke passed for all four in-kernel SIGKILL timing options. Next step is the same four-case run on two GB300 nodes for the real two-node NVLink/NVSwitch validation.
