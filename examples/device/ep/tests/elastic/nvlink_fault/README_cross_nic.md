# NVLink fault injection against NIXL EP on an RDMA-only transport (DGX B200)

## Goal

Inject a real, fatal NVLink link-down into one GPU while NIXL EP kernels are
running, with the workload configured so that **all peer-to-peer data movement
goes over RDMA** (no CUDA IPC / NVLink in the data path), and observe the blast
radius.

Question: does losing an NVLink on one GPU still terminate the whole job?

Answer: **no.** With an RDMA-only transport the fault stays local to the
affected GPU's NVLink hardware. Every rank, including the victim, kept
communicating and completed the post-fault phase at full throughput.

## Setup

- Node: `adv-dev-dgxb-201` (DGX B200, 8x B200, 8x 400G ConnectX RoCE HCAs).
- Image: `nixl-ep:master`.
- 4 ranks, one per GPU. Victim: rank/GPU 2, NVLink link 0.
- Transport: RDMA only. `--disable-ll-nvlink` makes NIXL EP set
  `UCX_TLS=^cuda_ipc`, excluding CUDA IPC (and therefore NVLink) from the data
  path. UCX then selects GPU-initiated RDMA (`rc_gda`).
- Each rank is pinned to its own rail-adjacent HCA, so traffic physically
  leaves the host and crosses the switch. The map comes from the `PXB` entries
  of `nvidia-smi topo -m`:

| rank / GPU | HCA | netdev | IP |
| --- | --- | --- | --- |
| 0 | `mlx5_4` | `enp24s0np0` | 5.2.1.1 |
| 1 | `mlx5_7` | `enp64s0np0` | 5.2.1.2 |
| 2 (victim) | `mlx5_8` | `enp79s0np0` | 5.2.1.3 |
| 3 | `mlx5_9` | `enp94s0np0` | 5.2.1.4 |

GPUs 4-7 map to `mlx5_10`, `mlx5_13`, `mlx5_14`, `mlx5_15` if the rank count is
raised. Containers run with host networking, `UCX_IB_GID_INDEX=3`
(RoCEv2/IPv4), and the GDA device `cuda0-<hca>` enabled per rank.

## What NIXL EP runs during the injection

The workload is `examples/device/ep/tests/elastic/elastic.py`, a Mixture-of-
Experts expert-parallel dispatch/combine test:

```bash
python3 -u examples/device/ep/tests/elastic/elastic.py \
    --plan /tools/plan_3phase.json --num-processes 4 \
    --num-experts-per-rank 32 --num-topk 8 --num-tokens 256 \
    --timeout-ms 10000 --disable-ll-nvlink
```

That is 4 ranks, 128 experts total, top-8 routing, 256 tokens, and a
1908.41 MB NIXL EP buffer per rank.

Each rank walks a phase plan. In every phase it connects to its peers and then
runs a correctness-and-bandwidth sweep of `buffer.dispatch()` and
`buffer.combine()` all-to-all collectives, iterating over `return_recv_hook`,
FP8 dispatch, `round_scale`, and `use_ue8m0` variants. Results are validated
against a reference (tolerance 1e-5, or 9e-4 for FP8), checked for NaNs, hashed,
and then benchmarked. So at injection time the GPUs are actively executing
expert-parallel all-to-all communication kernels with live correctness checks,
not sitting idle.

The plan is `three_phase_all_active_4rank.json`:

```text
[[0,1,2,3],[0,1,2,3],[0,1,2,3]]
```

All three phases keep all four ranks active. The plan deliberately does **not**
mark rank 2 as expected-to-fail, so nothing in the job is told that a fault is
coming. Phase 0 is a clean reference, the fault is injected during phase 1, and
phase 2 measures whether the same processes can keep communicating afterwards.

## Injection method

`nvlink_hwinject.c` talks directly to the NVIDIA Resource Manager. It opens
`/dev/nvidiactl`, builds the RM object hierarchy for the target GPU, and issues
one control ioctl:

```text
NV2080_CTRL_CMD_NVLINK_SET_HW_ERROR_INJECT (0x20803081)
linkMask    = 1 << 0
errType     = LINK_ERR
errSettings = FORCE_LINK_DOWN
```

This is a hardware-level forced link-down, not a software error simulation. The
injector runs inside the victim rank's container (`--cap-add SYS_ADMIN`), with
`CUDA_VISIBLE_DEVICES` cleared for the subprocess because the RM addresses GPUs
by physical index. Observed output:

```text
GPU minor 2: NVLink v8, enabled links 0x3ffff
*** FORCE_LINK_DOWN on GPU 2 link 0 ***
  SET_HW_ERROR_INJECT -> ioctl=0 status=0x0000 NV_OK
  ioctl took 16.439 ms
```

The harness first runs a non-destructive probe of the same injector (reporting
the link topology without changing it) and aborts if that fails.

## Injection timing

`patch_fault.py` patches the image's own `elastic.py` so the fault callback is
armed **only** in rank 2 and **only** in the configured phase
(`NIXL_EP_FAULT_PHASE=1`). It reuses the existing fault-tolerance hook: at the
beginning of the first `dispatch()` of that phase, a
`threading.Timer(0.0001, ...)` is started, so the RM ioctl lands roughly
100 microseconds later, while dispatch/combine kernels are in flight.

The log shows this ordering precisely - all four ranks enter phase 1, then the
injection fires mid-collective:

```text
43-50  global_rank=0..3 -> start phase 1
52     [gpu 2] forcing NVLink link 0 down
56     SET_HW_ERROR_INJECT -> ioctl=0 status=0x0000 NV_OK
58-63  all four ranks report phase-1 dispatch+combine bandwidth
61-69  all four ranks end phase 1 and start phase 2
```

## Evidence the RDMA path actually carried the traffic

Transport selection alone is not proof, so the harness gates on hardware
counters as well. From the injection run:

| Check | Value |
| --- | --- |
| Distinct HCAs pinned | `mlx5_4`, `mlx5_7`, `mlx5_8`, `mlx5_9` (one per rank) |
| `rc_gda` device selections | 4 on each of `cuda0-mlx5_4/7/8/9` |
| `cuda_ipc` selections | 0 |
| NIC counter delta, 4 HCAs summed | +27,965,732,192 (`port_xmit_data` + `port_rcv_data`) |
| NVLink data counter delta | 0 KiB |

The NVLink zero was confirmed to be real rather than a parsing artifact: the
capture holds 288 `Data Tx`/`Data Rx` lines and every one reads `0 KiB`. The
baseline gate in the same run independently required NIC counters to rise and
NVLink counters to stay flat, and it passed (16 `rc_gda` selections, 0
`cuda_ipc`, NIC delta +27,971,814,172, NVLink delta 0).

## Results

Archived under
`/swgwork/lishapira/nixl_ep_nvlink_fault_results_B200_cross_nic/`,
directory `crossnic_rdma_3phase_inject_20260915_175644`.

**Every rank survived and finished.** Verdict: `started=[0,1,2,3]`,
`ended=[0,1,2,3]`, `done=[0,1,2,3]`, `failed_worker_exits={}`.

Dispatch + combine bandwidth per rank per phase:

| phase | rank 0 | rank 1 | rank 2 (victim) | rank 3 |
| --- | --- | --- | --- | --- |
| 0 (clean) | 41.43 | 41.42 | 41.42 | 41.55 |
| 1 (fault injected) | 40.81 | 40.84 | 40.85 | 40.83 |
| 2 (post-fault) | 44.87 | 44.86 | 44.82 | 44.83 |

Units are GB/s. Throughput was essentially unchanged through the fault and
recovered fully afterwards; the victim rank is indistinguishable from its peers.

What the driver recorded on GPU2:

```text
Xid 149  NETIR_LINK_EVT  Fatal  XC0 i1 Link 00
Xid 154  recovery action changed None -> Drain and Reset
knvlinkSetDegradedMode_IMPL: GPU2 marked Degraded. Error originated on linkId 0!
Xid 154  recovery action changed Drain and Reset -> GPU Reset Required
```

Other observations:

- `fault.log` contains zero `uncorrectable NVLink error`, zero CUDA illegal
  memory access, and zero `UCX ERROR` lines.
- Final health: GPU2 at 0 NVLinks and `Recovery Action: Reset`; GPUs 0, 1, and
  3-7 unaffected at 18 links and `None`.
- A non-destructive three-phase dry run
  (`crossnic_rdma_3phase_dryrun_20260915_175402`) was executed first and also
  completed on all ranks, establishing the RDMA baseline independently of the
  fault.

## Explanation

On the default local transport, CUDA IPC maps peer GPU memory over NVLink into
every participating CUDA context. A fatal link event therefore invalidates
mappings that all ranks hold, every context becomes poisoned, and the failure is
global rather than local.

With `--disable-ll-nvlink` there are no CUDA IPC peer mappings in the data path.
Each rank reaches its peers through its own HCA, so GPU2's NVLink hardware is
not part of any peer's addressing. Taking that link down removes something no
other rank depends on, and the collectives continue over the NICs unaffected.

For reference, a control run on the same node, image, plan, victim rank, and
injection point but with the default CUDA IPC/NVLink transport behaves very
differently. There the baseline selects `cuda_ipc` and moves 102,821,246 KiB
over NVLink, and the same injection produces 38 `uncorrectable NVLink error`
reports, after which **no rank enters phase 2 and all exit with code 1**.
Control logs are archived alongside this experiment in
`control_nvlink_cuda_ipc_3phase_20260903_155339`.

### Why the victim also finished

GPU2 completing is expected rather than suspicious. Its payload travels over
`mlx5_8`, not NVLink, and the `Drain and Reset` recovery action lets already
running work finish while blocking creation of *new* CUDA contexts. Within this
short job GPU2 therefore ran to completion on its existing context. A longer job,
or any attempt to start new work on GPU2, would require the reset first - which
is why GPU2 ends the run in `Recovery Action: Reset`.

This distinction matters when reading the result: it demonstrates fault
containment and continued communication, not that the GPU is healthy.

## Reproduce

Prerequisites:

- All GPUs report 18 NVLinks and `Recovery Action: None`, with no active GPU
  processes. The harness aborts if not.
- Switch-side L2 forwarding between the host's rail ports. On this fabric the
  host ports had to be members of the switch default bridge as access VLAN 1
  ports; without that, cross-HCA RDMA fails at the data phase with
  `RETRY_EXC_ERR` (packets transmitted, zero received).
- Host-side RoCE configuration, which does **not** persist across a boot:

```bash
sudo ./examples/device/ep/tests/elastic/nvlink_fault/rdma_restore.sh --test
```

`--test` is required, not optional. All eight HCAs share `5.0.0.0/8`, so dynamic
ARP resolves unreliably between them; `--test` installs static permanent
any-to-any neighbors. Confirm its self-test prints Gb/s (not `FAILED`) for each
rail. Then run the non-destructive baseline:

```bash
NIXL_EP_CROSS_NIC=1 NIXL_EP_DRY_RUN=1 \
  ./examples/device/ep/tests/elastic/nvlink_fault/run_rdma_only_inject_min.sh
```

Then the real injection:

```bash
NIXL_EP_CROSS_NIC=1 NIXL_EP_DRY_RUN=0 \
  ./examples/device/ep/tests/elastic/nvlink_fault/run_rdma_only_inject_min.sh
```

Add `NIXL_EP_NVLINK_DEFAULT=1` to run the default CUDA IPC/NVLink control
instead. Results are written under `/var/tmp/nixl_ep_nvlink/`.

## Recovery after an injection

The injection is destructive. GPU2 ends in `Recovery Action: Reset`, and an
`nvidia-smi` reset is not sufficient. Per run:

1. BMC power cycle, e.g.
   `ipmitool -I lanplus -H <bmc> -U <user> -P <pass> chassis power cycle`.
2. Run `rdma_restore.sh --test` from this directory - after a power cycle the
   HCAs come back admin-down with no IPs and no static neighbors.
3. Verify all GPUs report 18 links and `Recovery Action: None`.

## Additional notes

- Skipping `--test` leaves `INCOMPLETE` neighbor entries and the run fails in
  UCX's UD wireup with
  `ibv_create_ah(dgid=::ffff:5.2.1.x sgid_index=3) ... Connection timed out`.
  `/proc/net/arp` will show `00:00:00:00:00:00` for exactly the peers that
  time out.
- An aborted previous run can leave `python3` workers holding GPU memory, after
  which preflight stops with `ABORT: a GPU workload is active.` Kill the
  container with `sudo docker kill` before retrying.
- `UCX DIAG invalid gid[3] on mlx5_0..3` is harmless. Those are the IB-link
  HCAs, which have no RoCEv2/IPv4 GID; the pinned rail HCAs are unaffected.
- The harness patches the container image's own `elastic.py` rather than
  mounting one from this branch, because the branch copy's API does not match
  the compiled `nixl_ep` in the image.
- A single NVLink link is injected (`linkMask = 1 << 0`), but the driver
  degrades the GPU and drops all 18 NVLinks in response.

## Files

- `run_rdma_only_inject_min.sh`: preflight, container runs, transport and
  hardware-counter gates, injector probe, injection, health capture, dmesg
  delta, and verdict. `NIXL_EP_CROSS_NIC=1` selects the rail map, defaults the
  GID index to 3, checks every HCA in use, and sums NIC counters across them.
- `patch_fault.py`: patches the image's `elastic.py`. Reads `NIXL_EP_NIC_MAP`
  to give each rank its own HCA by `local_rank` (falling back to a single
  `NIXL_EP_NIC`), and arms the rank/phase-specific fault callback.
- `nvlink_hwinject.c`: NVIDIA RM hardware injector.
- `three_phase_all_active_4rank.json`: the phase plan.
- `README_cross_nic.md`: this document.
