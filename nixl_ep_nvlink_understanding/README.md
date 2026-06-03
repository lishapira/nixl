# NIXL EP NVLink Understanding Runs

This directory groups the NIXL EP SIGKILL/NVLink experiments by scope.

- `common/`: shared plans, planning notes, topology examples, and legacy helpers.
- `1node_sigkill_inkernel/`: completed one-node GB200/GB300 smoke for in-kernel send/receive SIGKILL timing.
- `old_2nodes_sigkill_non_inkernel/`: completed two-node elastic boundary/hook SIGKILL run.
- `2nodes_sigkill_inkernel/`: two-node in-kernel SIGKILL run area; not completed yet.

For each experiment bucket:

- `run_*.sh`: instruction/runner script.
- `plans/`: JSON plans used by the runner.
- `results/`: raw outputs and evidence logs.
- `*_SUMMARY.md` or report: human-readable result summary.

Pass criteria for in-kernel runs:

- `HIT_IN_KERNEL_WINDOW`
- `exited_before_sigkill=0`
- no `MISSED_IN_KERNEL_TIMING` or `IN_KERNEL_MARKER_TIMEOUT`
- no traceback/assert/runtime/CUDA/Xid errors
- surviving ranks detect failed rank `{2}` and continue bandwidth output
- non-killed ranks reach `done`
- cleanup shows no leftover compute apps
- post-fault baselines return `rc=0`
