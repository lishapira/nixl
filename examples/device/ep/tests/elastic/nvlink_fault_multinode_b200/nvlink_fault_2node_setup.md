# NVLink fault on two B200 nodes - setup, how to run, run notes

What is needed to rerun the experiment, taken from the run folders
(`/swgwork/lishapira/nixl_ep_b200_2node/<run>/`). Results: `nvlink_fault_2node_results.md`;
conclusions: `nvlink_fault_2node_conclusion.md`.

## Environment

| | adv-dev-dgxb-201 | adv-dev-dgxb-200 |
|---|---|---|
| Role | master, ranks 0-3 | peer, ranks 4-7 (victim rank 4) |
| Driver | 615.71.09 | 580.173.02 |
| Rail addresses | 5.2.1.N | 5.2.0.N |

- 8x B200 per node, NV18 between every GPU pair. The runs use GPUs 2,3,5,6.
- Rails 3,4,6,7 = `mlx5_8`, `mlx5_9`, `mlx5_13`, `mlx5_14` (`enp79s0np0`, `enp94s0np0`,
  `enp192s0np0`, `enp206s0np0`), the NICs next to GPUs 2,3,5,6 (PXB). N = rail number.
- Victim: 200 GPU 2, PCI `0000:52:00.0`, UUID `GPU-c31cc41b-79e2-7031-5e01-eb6c762d4f65`.
- Image `nixl-ep:master`, ID `sha256:9b47eb7f...`, created 2026-09-03 (CUDA 13.0.2, UCX
  1.24.0). Copied from 201 to 200 with `docker save | zstd` and `zstd -dc | docker load`
  (`/swgwork/lishapira/nixl-ep-master-9b47eb7f.tar.zst`).
- `elastic.py` is the image's own copy. `patch_rdma.py` and `patch_fault_2node.py` patch it
  inside the container on every run.

Parameters in every run: 8192 tokens, 2 experts per rank, top-k 8, EP receive timeout
30000 ms, plan `three_phase_all_active_8rank.json` (3 phases, all 8 ranks active), fault in
phase 1 on rank 4, link 0. `during_send` used `NIXL_EP_FAULT_SEND_OFFSET_MS=12`.

## Before a run

- After every reboot of a node: `sudo ../nvlink_fault/rdma_restore.sh --test` on it. The
  runner's preflight fails without it.
- A real fault needs both nodes free of other users' GPU jobs.
- `sudo -v` on both hosts; the runner uses `sudo` for docker and dmesg.

## How to run

Start 201 first; use the same `RUN_ID` and settings on both hosts. `RUN_ID` must be new.
Results go to `/swgwork/lishapira/nixl_ep_b200_2node/<RUN_ID>/`. Run the real injection only
after its dry run printed `DRY RUN PASSED`.

On 201:

```bash
cd examples/device/ep/tests/elastic/nvlink_fault_multinode_b200
sudo -v
COMMON="NIXL_EP_FAULT=nvlink NIXL_EP_RAILS=3,4,6,7 NIXL_EP_GPUS=2,3,5,6 NIXL_EP_WAIT_SECONDS=1800"
env RUN_ID=dry-after_send-1 NIXL_EP_FAULT_TIMING=after_send NIXL_EP_FAULT_DRY_RUN=1 $COMMON bash run_2node.sh master
env RUN_ID=inject-after_send-1 NIXL_EP_FAULT_TIMING=after_send $COMMON bash run_2node.sh master
```

On 200: the same commands with `peer` instead of `master`.

- Other timings: set `NIXL_EP_FAULT_TIMING` (modes listed at the top of `run_2node.sh`).
- Baseline: leave out `NIXL_EP_FAULT` (default `none`).
- SIGTERM control: `NIXL_EP_FAULT=sigterm` (ranks 4-7 by default).
- One node (on 200 only): `RUN_ID=<id> NIXL_EP_FAULT_TIMING=after_send bash run_1node.sh`;
  results in `/swgwork/lishapira/nixl_ep_b200_1node/`.

## After a real fault

200 GPU 2 is left with 0 links and Recovery "Reset". Power-cycle 200 from 201:

```bash
ipmitool -I lanplus -H adv-dev-dgxb-200-bmc -U admin -P '<admin password>' chassis power cycle
```

When it is back, run `rdma_restore.sh --test` on 200 and check that every GPU has 18 links and
Recovery "None". The next run's preflight stops otherwise (as in `inject-after_recv-1`).

## Run folder contents

- `summary.log`: verdict.
- `timeline.log`: events in ms from the injector start.
- `ep_<host>.log`: EP output with `[t=<epoch>]` stamps.
- `dmesg_delta_<host>.log`: Xids.
- `gpu_health_*`, `nvlink_*`, `ib_*`: before/after state.
- `preflight_*`, `injector_probe_*`, `run_config.txt`, `fault_target.txt`.

`summarize_fault.py <dir> ...` and `fault_timeline.py <dir>` rebuild `summary.log` and
`timeline.log` from these files.

## Notes on specific runs

- `inject-during_send-1`, `inject-after_recv-1`: nothing was injected (200 waited at the sudo
  prompt; preflight stopped because GPU 2 was not reset yet).
- `dry-before_all_send-1`: the peer was started again by mistake after the run finished and
  overwrote 200's raw files; `summary.log` and `timeline.log` are the original ones.
- `inject-tnvlink-1`: kernel-log times are ~115 ms off (fixed afterwards in the runner).
- `dry-nvlink-1`, `dry-nvlink-2`, `baseline-4rank-1` to `-5`: earlier attempts while
  building the runner; `dry-nvlink-3` and `baseline-4rank-6` are the passing ones.
- `before_send_masked` (in the code, not in the results file): ranks 5-7 mask rank 4, then
  send after the drop. In `inject-before_send_masked-1` they survived and only rank 4 died.
  Its dry run cannot pass: rank 4 stays alive there, and masking it on ranks 5-7 only
  breaks the run.
