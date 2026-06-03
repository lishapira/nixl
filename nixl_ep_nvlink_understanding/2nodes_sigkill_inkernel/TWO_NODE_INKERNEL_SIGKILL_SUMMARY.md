# Two-Node In-Kernel SIGKILL Summary

Status: not completed yet.

Goal: run the four in-kernel SIGKILL timings on two GB200/GB300 nodes:

- `dispatch-send-during-kernel`
- `dispatch-receive-during-kernel`
- `combine-send-during-kernel`
- `combine-receive-during-kernel`

Runner:

`run_2node_inkernel_smoke.sh`

Current result:

- A previous Slurm attempt created `results/failed_slurm_1937788/slurm-1937788.out`.
- That attempt failed before tests started due to `spank_sybil` credential setup.
- No two-node in-kernel build/test logs exist yet.

Expected pass criteria:

- evidence files contain `HIT_IN_KERNEL_WINDOW`
- `exited_before_sigkill=0`
- no `MISSED_IN_KERNEL_TIMING` or `IN_KERNEL_MARKER_TIMEOUT`
- healthy ranks detect failed rank `{2}`
- healthy ranks continue dispatch/combine bandwidth output
- non-killed ranks reach `done`
- cleanup shows no leftover compute apps
- post-fault baselines return `rc=0`
