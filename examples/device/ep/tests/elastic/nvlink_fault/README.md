# B200 single-NIC RDMA-only NVLink fault experiment

## Goal

Compare the effect of the same live NVLink link-down fault on four NIXL EP
ranks using:

- RDMA-only: `--disable-ll-nvlink`, UCX `rc_gda`, all ranks pinned to
  `mlx5_4`.
- Default local transport: CUDA IPC over NVLink.

Both runs use `nixl-ep:master`, rebuilt on September 3, 2026 from the latest
NIXL and UCX master branches available that day, victim rank/GPU 2, link 0,
and `three_phase_all_active_4rank.json`:

```text
[[0,1,2,3],[0,1,2,3],[0,1,2,3]]
```

The plan does not expect rank 2 to fail. Phase 0 is a clean reference, the
fault is injected during phase 1, and phase 2 verifies whether the same
processes can continue after injection.

## Why one NIC

Each HCA passed approximately 350 Gb/s self-loopback, but that traffic stays
inside the HCA. Cross-NIC tests showed packets transmitted with no increase in
the destination's hardware receive counter. Forced-interface ping also failed
after the destination MAC was installed statically, bypassing ARP. Therefore
this deployment had no usable L2 path between the tested rails. LLDP or switch
inspection is still needed to distinguish cabling or isolated rails from
switch VLAN/port policy.

All ranks were therefore pinned to the validated `mlx5_4` path. This tests
whether avoiding CUDA IPC contains the NVLink fault, but it does not validate
separate HCAs, external wire traffic, routing, or cross-rail behavior.

## Injection

`nvlink_hwinject.c` opens the NVIDIA RM device hierarchy and sends:

```text
NV2080_CTRL_CMD_NVLINK_SET_HW_ERROR_INJECT (0x20803081)
linkMask = 1 << 0
errType = LINK_ERR
errSettings = FORCE_LINK_DOWN
```

The patch arms this command only in rank 2 and schedules it 100 microseconds
after phase 1 dispatch begins. Therefore injection occurs during phase 1 (the
second plan phase), followed by the post-fault phase 2. A successful call
reports `NV_OK`, generates a fatal link event, drops all NVLinks on GPU2, and
leaves GPU2 requiring reset. This is a destructive hardware injection;
power-cycle the node before another GPU experiment.

## Results

Recorded on a DGX B200 with four ranks and `mlx5_4`.

### RDMA-only

Result directory:
`/swgwork/lishapira/nixl_ep_nvlink_fault_results_B200_single_nic/rdma_3phase_20260903_154223`

- The three-phase baseline completed on all ranks.
- UCX selected `device(rc_gda/cuda0-mlx5_4:1)`, did not select `cuda_ipc`,
  increased the NIC counters, and produced zero NVLink counter movement.
- Injection returned `NV_OK`; GPU2 logged Xid 149 and Xid 154.
- Ranks 0, 1, 2, and 3 completed the post-fault phase 2 and exited normally.
- No CUDA error or peer-GPU Xid was recorded.
- GPU2 ended with zero NVLinks and `Recovery Action: Reset`; other GPUs stayed
  healthy.

This shows containment under the tested single-NIC `rc_gda` configuration:
the NVLink failure did not terminate the running ranks, and the same processes
continued through another communication phase after injection.

### NVLink/CUDA IPC comparison

Result directory:
`/swgwork/lishapira/nixl_ep_nvlink_fault_results_B200_single_nic/nvlink_3phase_20260903_155339`

- The three-phase baseline completed on all ranks with four `cuda_ipc`
  selections and 102,821,246 KiB of NVLink counter movement.
- Injection returned `NV_OK`.
- All four ranks reported CUDA `uncorrectable NVLink error` and exited with
  code 1 before completing phase 1.
- No rank entered the post-fault phase 2.
- GPU2 logged the originating Xid 149/154. Peer GPUs logged Xid 145 followed
  by Xid 45 channel kills.

CUDA IPC connected every process to peer GPU memory over NVLink, so the
injected fault poisoned all participating CUDA contexts. In RDMA-only mode,
those CUDA IPC/NVLink peer mappings were absent, containing the observed
failure.

## Reproduce

Prerequisites:

- `nixl-ep:master` built with the matching NIXL EP/UCX stack.
- All GPUs show 18 NVLinks and `Recovery Action: None`.
- No active GPU processes.
- `mlx5_4` is `PORT_ACTIVE`; on this node:

```bash
sudo ip link set enp24s0np0 up
```

First run the non-destructive all-active baseline:

```bash
NIXL_EP_DRY_RUN=1 \
NIXL_EP_IMAGE=nixl-ep:master \
./examples/device/ep/tests/elastic/nvlink_fault/run_rdma_only_inject_min.sh
```

Run the real RDMA-only injection:

```bash
NIXL_EP_DRY_RUN=0 \
NIXL_EP_IMAGE=nixl-ep:master \
./examples/device/ep/tests/elastic/nvlink_fault/run_rdma_only_inject_min.sh
```

Power-cycle the node, restore `enp24s0np0`, then run the NVLink comparison:

```bash
NIXL_EP_DRY_RUN=0 \
NIXL_EP_NVLINK_DEFAULT=1 \
NIXL_EP_IMAGE=nixl-ep:master \
./examples/device/ep/tests/elastic/nvlink_fault/run_rdma_only_inject_min.sh
```

New results are initially written under `/var/tmp/nixl_ep_nvlink/`. Relevant
comparison logs are archived under
`/swgwork/lishapira/nixl_ep_nvlink_fault_results_B200_single_nic/`. The script
aborts before injection if health or transport checks fail.

## Files

- `three_phase_all_active_4rank.json`: the single plan used by both paths.
- `run_rdma_only_inject_min.sh`: preflight, container runs, counter gates,
  injection, health capture, and verdict.
- `patch_fault.py`: patches the image's compatible `elastic.py` with the
  single-NIC pin and rank/phase-specific fault callback.
- `nvlink_hwinject.c`: NVIDIA RM hardware injector used by the callback.
- `README.md`: experiment scope, results, and reproduction steps.
