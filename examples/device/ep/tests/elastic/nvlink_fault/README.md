# B200: NVLink fault injection with NVLink and RDMA transports

## Goal

Inject the same real NVLink hardware link-down while NIXL EP is running, with
only the payload transport changed:

- Default local path: CUDA IPC over NVLink.
- RDMA-only path: cross-NIC GPU-initiated RDMA (`rc_gda`) with CUDA IPC
  disabled.

Both paths use the same eight ranks, workload parameters, plan, victim GPU,
link, and injection point. This makes the runs directly comparable.

## Tested configuration

Recorded on `adv-dev-dgxb-201`, a DGX B200 with eight GPUs and eight
rail-adjacent 400G RoCE HCAs. The tested image was `nixl-ep:master`; this branch
is based on NIXL commit `adf179a54a0951193bff0ab90201dfc605ed5d5c`.

The exact workload settings are the defaults in
`run_nvlink_fault_experiment.sh`:

```text
ranks                 8
tokens                8192
experts per rank      2
top-k                 8
timeout               30000 ms
victim                rank 2 / physical GPU 2
injected link         0
injection phase       1
```

NIXL EP allocates a 7633.633664 MB buffer per rank and runs its
dispatch/combine correctness and bandwidth workload. The plan keeps every rank
active in all three phases:

```json
[
  [0, 1, 2, 3, 4, 5, 6, 7],
  [0, 1, 2, 3, 4, 5, 6, 7],
  [0, 1, 2, 3, 4, 5, 6, 7]
]
```

Phase 0 is the clean reference. The fault fires during phase 1. Phase 2 tests
whether the same processes can continue after the fault. The plan does not
tell NIXL EP that rank 2 is expected to die.

## Cross-NIC RDMA topology

The RDMA run gives each rank its own GPU-adjacent HCA:

```text
rank/GPU 0 -> mlx5_4
rank/GPU 1 -> mlx5_7
rank/GPU 2 -> mlx5_8
rank/GPU 3 -> mlx5_9
rank/GPU 4 -> mlx5_10
rank/GPU 5 -> mlx5_13
rank/GPU 6 -> mlx5_14
rank/GPU 7 -> mlx5_15
```

The map comes from the `PXB` relationships in `nvidia-smi topo -m`. All HCAs
use RoCEv2/IPv4 GID index 3. Switch-side L2 forwarding between the rails must
already be configured.

After every reboot or power cycle, restore the NICs, IPs, multihoming sysctls,
and static any-to-any neighbors:

```bash
sudo ./examples/device/ep/tests/elastic/nvlink_fault/rdma_restore.sh --test
```

The self-test must report successful bandwidth for all seven peer HCAs.
Without the static neighbors, UCX wireup can fail with
`ibv_create_ah ... Connection timed out`.

## Injection method and timing

`nvlink_hwinject.c` opens `/dev/nvidiactl`, builds the NVIDIA Resource Manager
client/device/subdevice hierarchy, and sends:

```text
NV2080_CTRL_CMD_NVLINK_SET_HW_ERROR_INJECT (0x20803081)
linkMask    = 1 << 0
errType     = LINK_ERR
errSettings = FORCE_LINK_DOWN
```

This is a real hardware link teardown, not a simulated RAS event.

`patch_fault.py` patches the `elastic.py` contained in the tested image. It
arms the injector only in rank 2 and only in phase 1. At the first
`dispatch()` of that phase, the existing fault hook starts:

```python
threading.Timer(0.0001, FAULT_ACTION)
```

The ioctl therefore runs approximately 100 microseconds after dispatch begins,
while dispatch/combine kernels are active. In both tested transports all eight
ranks logged `start phase 1` before:

```text
[gpu 2] forcing NVLink link 0 down
SET_HW_ERROR_INJECT -> ioctl=0 status=0x0000 NV_OK
```

The driver generated fatal Xid 149 on GPU2, dropped all 18 of its NVLinks, and
left it at `Recovery Action: Reset`.

## Transport selection and proof

### RDMA-only

The runner passes `--disable-ll-nvlink`, which makes NIXL EP exclude
`cuda_ipc`. Each rank receives:

```text
UCX_NET_DEVICES=<rank HCA>:1,cuda0-<rank HCA>:1
UCX_TLS=^cuda_ipc
```

The pre-injection baseline must complete all eight ranks, select `rc_gda`,
select no `cuda_ipc`, increase the summed NIC hardware counters, and produce
zero NVLink data-counter movement. Otherwise the script aborts before
injection.

### Default NVLink

Set `NIXL_EP_NVLINK_DEFAULT=1`. The runner does not pass
`--disable-ll-nvlink` and does not expose the GDA device. The baseline must
complete all eight ranks, select `cuda_ipc`, and increase NVLink data
counters. Otherwise the script aborts before injection.

The rail HCAs remain pinned in this mode so the UCX backend can initialize, but
the payload device selected by UCX is `cuda_ipc/cuda`, not `rc_gda`.

## Reproduce

Warning: a real injection is destructive. Prepare BMC power-cycle access
before starting.

Verify first:

- Every GPU has 18 active links and `Recovery Action: None`.
- No GPU compute process is running.
- Every mapped HCA is `PORT_ACTIVE`.
- `rdma_restore.sh --test` reports successful cross-NIC RDMA.

Run the non-destructive eight-rank RDMA validation:

```bash
NIXL_EP_DRY_RUN=1 \
  ./examples/device/ep/tests/elastic/nvlink_fault/run_nvlink_fault_experiment.sh
```

Run the real eight-rank RDMA injection:

```bash
NIXL_EP_DRY_RUN=0 \
  ./examples/device/ep/tests/elastic/nvlink_fault/run_nvlink_fault_experiment.sh
```

Power-cycle the node and run `rdma_restore.sh --test` again. Then run the
matched default-NVLink injection:

```bash
NIXL_EP_DRY_RUN=0 NIXL_EP_NVLINK_DEFAULT=1 \
  ./examples/device/ep/tests/elastic/nvlink_fault/run_nvlink_fault_experiment.sh
```

Each invocation first runs a complete no-fault baseline. New results are
written under `/var/tmp/nixl_ep_nvlink/inject_8rank_<timestamp>/`.

The runner exits 0 when all required ranks complete. A default-NVLink
full-blast run exits 11 because workers fail and the runner's survivor verdict
correctly fails; this is the expected experimental outcome, not a harness
startup failure.

## Results

Persistent copies are stored on the test system under:

```text
/swgwork/lishapira/nixl_ep_nvlink_fault_final_8rank/
```

### RDMA dry run

Result: `rdma_8rank_3phase_dryrun_20260916_150114`

- Eight distinct HCAs and `rc_gda` were selected.
- `cuda_ipc` usage was zero.
- NIC counters increased; NVLink data counters stayed at zero.
- All eight ranks completed all three phases.
- No hardware fault was injected.

### RDMA real injection

Result: `rdma_8rank_3phase_inject_20260916_150435`

- Baseline: 8/8 ranks completed, 64 UCX configuration lines showed `rc_gda`,
  none showed `cuda_ipc`, NIC counters increased, and NVLink counters did not.
- Injection returned `NV_OK` after all eight ranks entered phase 1.
- All eight ranks ended phase 1, completed phase 2, and exited cleanly.
- There were no uncorrectable NVLink errors, CUDA illegal-memory-access
  errors, UCX errors, peer Xids, or peer channel kills.
- Only GPU2 logged Xid 149/154 and became degraded.
- Mean dispatch/combine bandwidth was 40.86 GB/s in phase 0, 41.13 GB/s in
  the injected phase, and 41.87 GB/s post-fault.

This proves that the RDMA payload continued normally while GPU2's NVLink
hardware was degraded.

### Default NVLink real injection: full blast radius

Result: `nvlink_8rank_3phase_inject_repeat_20260916_160144`

- Baseline: 8/8 ranks completed, 64 UCX configuration lines showed
  `cuda_ipc`, none showed `rc_gda`, and NVLink counters increased.
- Injection returned `NV_OK` after all eight ranks entered phase 1.
- No rank completed phase 1 or entered phase 2.
- All eight workers exited code 1.
- All eight worker PIDs independently reported
  `uncorrectable NVLink error detected during the execution`.
- Fatal Xid 149 occurred on GPU2. Xid 145 and channel-kill Xid 45 events
  occurred on every GPU.
- GPU2 became degraded; every peer retained 18 links and
  `Recovery Action: None`.

### Default NVLink timing variation

Result: `nvlink_8rank_3phase_inject_20260916_151855`

The same command and parameters produced a partial blast radius once:

- Ranks 1-5 failed and independently reported the uncorrectable error.
- Ranks 0, 6, and 7 remained healthy, detected/masked the failed ranks, and
  completed phase 2.
- Xid 145/45 appeared on GPUs 1-5, while GPUs 0, 6, and 7 received no Xids.

This run is retained because it is valid evidence of timing-dependent fault
propagation. It does not show reliable containment: the default-NVLink repeat,
the earlier B200 experiment, and the mapped ranks in this run demonstrate that
a CUDA IPC peer context can be poisoned before NIXL EP masking can help.

## Explanation and conclusion

CUDA IPC maps peer GPU memory over NVLink into each participating rank's CUDA
context. When GPU2 leaves the fabric, ranks that consume the invalid peer
mapping receive CUDA error 220 (`cudaErrorNvlinkUncorrectable`). Their own CUDA
contexts are poisoned, so process-level NIXL EP masking cannot save them.

RDMA-only communication does not install CUDA IPC/NVLink peer mappings in the
payload path. Each rank reaches peer memory through its HCA. The same physical
NVLink failure therefore leaves peer CUDA contexts healthy, and all ranks can
continue communicating over RDMA.

The evidence supports these conclusions:

- One targeted link-down removes the victim GPU from the NVLink fabric: all 18
  victim links drop and the victim requires reset.
- Hardware degradation is local to the victim GPU.
- With default CUDA IPC/NVLink, process and CUDA-context damage can spread to
  every mapped rank; a full eight-rank job failure is reproducible.
- With cross-NIC RDMA, the same injection is contained from the workload's
  perspective: all eight ranks continue at unchanged throughput and peers
  receive no Xids.
- NIXL EP masking can recover only ranks whose CUDA contexts remain healthy.
  It cannot repair a context already poisoned by an uncorrectable NVLink
  error.

## Recovery

After every real injection, GPU2 ends at zero links with
`Recovery Action: Reset`. An in-band `nvidia-smi` reset was insufficient in
this setup. Cold power-cycle the node, then restore the RDMA host state:

```bash
sudo ./examples/device/ep/tests/elastic/nvlink_fault/rdma_restore.sh --test
```

Before another experiment, confirm all eight GPUs have 18 links and
`Recovery Action: None`.

## Files

- `nvlink_hwinject.c`: R570 NVIDIA RM hardware link-down injector.
- `patch_fault.py`: adds the phase/rank-specific callback and per-rank HCA
  selection to the image's compatible `elastic.py`.
- `run_nvlink_fault_experiment.sh`: preflight, path selection, baseline gates,
  hardware counters, injector probe, real injection, evidence capture, and
  verdict.
- `three_phase_all_active_8rank.json`: common plan for both transports.
- `rdma_restore.sh`: post-boot RoCE setup, static neighbors, and cross-NIC
  validation.
- `README.md`: experiment method, exact reproduction, results, and conclusion.
