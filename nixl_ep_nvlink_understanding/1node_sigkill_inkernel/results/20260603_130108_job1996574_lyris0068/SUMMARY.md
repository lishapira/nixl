# One-Node In-Kernel SIGKILL Run

- Started UTC: `2026-06-03T13:01:08Z`
- Job: `1996574`
- Node: `lyris0068`
- Branch: `nvlink_understanding`
- Commit: `4087c88 nixl_ep tests: add in-kernel SIGKILL markers`
- Spin cycles: `100000000`
- Results: `/lustre/fsw/network_research_advdev/lishapira/nixl_ep_nvlink_understanding/1node_sigkill_inkernel/results/20260603_130108_job1996574_lyris0068`

| Timing | Fault rc | Hit | Exited before SIGKILL | Cleanup | Post baseline rc | Result |
|---|---:|---|---:|---|---:|---|
- Build rc: `0`
| `dispatch-send-during-kernel` | 0 | yes | 0 | PASS | 0 | PASS |
| `dispatch-receive-during-kernel` | 0 | yes | 0 | PASS | 0 | PASS |
| `combine-send-during-kernel` | 0 | yes | 0 | PASS | 0 | PASS |
| `combine-receive-during-kernel` | 0 | yes | 0 | PASS | 0 | PASS |
