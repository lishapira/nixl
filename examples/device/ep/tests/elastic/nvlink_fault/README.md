# NVLink link-down fault injection for NIXL EP

Takes a real NVLink down on one GPU while EP traffic is in flight, to test whether EP's
fault tolerance survives losing a peer's interconnect.

**Result: it does not. A single GPU losing NVLink kills the entire 8-rank job.**

---

## How the error is injected

The NVIDIA driver exposes an internal Resource Manager (RM) API through `/dev/nvidiactl`.
`nvlink_hwinject.c` talks to it directly with `ioctl()`, because no user-facing tool exposes
the control we need. It builds the object chain RM requires (client -> device -> subdevice)
and issues:

```
NV2080_CTRL_CMD_NVLINK_SET_HW_ERROR_INJECT   (0x20803081)
    linkMask   = 1 << <link>
    errCfg[<link>] = { errType = LINK_ERR, errSettings = FORCE_LINK_DOWN }
```

This is a real hardware link teardown, not a simulated RAS event. Struct layouts are
transcribed from NVIDIA's open kernel modules at tag **570.158.01** and must be re-checked
if the driver changes. Requires root / `CAP_SYS_ADMIN`.

The same mechanism backs NVIDIA's RAS validation suite (NVRASTool) NVLink tests. We call the
control directly because the injector binary shipped with that suite is built for driver
branch R595 and our R570 driver rejects it at the RM version handshake.

**Trigger point.** `elastic.py` already had a fault hook that sends `SIGTERM` to a victim
rank at the first dispatch of its phase. `--fault-nvlink` swaps only the callback:

```python
timer = threading.Timer(0.0001, FAULT_ACTION or self_kill)
```

Without the flag the SIGTERM path is byte-for-byte unchanged.

## Running it

The capture runner performs health checks, builds and probes the injector, runs the original
8-rank experiment, and saves the application log, `dmesg`, PID-to-GPU snapshots, and pre/post
GPU health:

```bash
sudo -v
bash examples/device/ep/tests/elastic/nvlink_fault/run_final_8rank_repro.sh
```

By default, results are saved in
`/var/tmp/nixl_ep_nvlink/final_8rank_<timestamp>/`. To choose another persistent location:

```bash
NIXL_EP_NVLINK_RESULTS_DIR=/shared/my_run \
  bash examples/device/ep/tests/elastic/nvlink_fault/run_final_8rank_repro.sh
```

At completion it prints `=== error summary ===`, followed by per-PID counts for
`uncorrectable NVLink error detected`, `all_8_pids_uncorrectable=true|false`, and the worker
exit line. A successful fault experiment returns the workload's nonzero status because all
eight workers are expected to fail. GPU 2 then requires a BMC power cycle.

The equivalent manual steps are:

```bash
# 1. build the injector on the DUT
gcc -O1 -Wall -o nvlink_hwinject nvlink_hwinject.c

# 2. sanity probe - targets no link, changes nothing.
#    NV_ERR_INVALID_PARAMETER (0x3b) = control is available.
#    NV_ERR_NOT_SUPPORTED (0x56) or NV_ERR_GENERIC (0xffff) = unusable on this driver.
sudo ./nvlink_hwinject 2

# 3. run EP with the fault
sudo docker run --rm --gpus all --ipc=host --cap-add=SYS_ADMIN \
  -v <repo>/examples/device/ep/tests/elastic:/workspace/nixl/examples/device/ep/tests/elastic \
  -v <path>/nvlink_fault:/tools \
  -w /workspace/nixl nixl-ep:local \
  python3 -u examples/device/ep/tests/elastic/elastic.py \
    --plan examples/device/ep/tests/elastic/nvlink_fault_rank2.json \
    --num-processes 8 --num-tokens 8192 \
    --fault-nvlink --fault-nvlink-link 0
```

Two flags are not optional:

- **`--cap-add=SYS_ADMIN`** — without it the ioctl returns `NV_ERR_INSUFFICIENT_PERMISSIONS`
  (`0x1b`) and injects nothing. The run then looks like a clean pass.
- **`--num-tokens 8192`** — purely timing. The fault needs ~325 ms of live traffic to reach
  the data path; at the default 128 tokens a phase is only ~71 ms and the job finishes
  before the link dies (measured: 128 -> 71 ms, 2048 -> 343 ms, 8192 -> ~1.2 s).

## What happens

Kernel, relative to the ioctl:

```
+0.6 ms    Xid 149  NETIR_LINK_EVT Fatal Link 00 (0x016415c6)
+283 ms    Xid 45   CUDA channels killed, "caused by previous Xid 149"
+404 ms    GPU marked Degraded -> Xid 154 "GPU Reset Required"
```

The ioctl itself blocks ~16 ms idle, ~274 ms under live traffic. That is the driver tearing
the link down against in-flight data, not a hang.

Note the fault is **not contained to the targeted link**: a fatal link event degrades the
whole GPU, so all 18 of its links drop. It is effectively "remove this GPU from the fabric".

## Result

**All 8 workers exit 1.** Reproduced four times, including two fully archived runs on
2026-08-18.

```
RuntimeError: Worker processes failed: worker 0 (exit code 1), worker 1 (exit code 1),
worker 2 (exit code 1), ... worker 7 (exit code 1)
```

Processes die with CUDA error 220, `cudaErrorNvlinkUncorrectable`, reported through failing
CUDA IPC teardown — `cuIpcCloseMemHandle` and `cuMemFree_v2(device_ep)`. Those are the peer
IPC mappings between ranks coming apart, which is why the blast radius is the whole job: EP
ranks map each other's memory over NVLink, so when one GPU leaves the fabric every rank that
mapped it takes uncorrectable errors on its own GPU.

XIDs land on **all 8 GPUs**, not just the victim (final archived run):

```
victim GPU:  Xid 149 x1, 145 x1, 154 x1-2, 45 x64
each peer:   Xid 145 x1, 45 x32
```

The runner archives these kernel-side events in `dmesg.log` and the matching host process
ownership in `pidmap.log`.

How to read them. `Xid 45` names the GPU that hosted a channel, the pid that owned it, the
channel id, and which earlier XID caused the teardown — the only XID that ties a GPU to a
process, and the one that determines who dies. Counts are per process per GPU, so a GPU shared
by several processes accumulates more.

**`45 x64` is 64 teardown events, not 64 distinct channels.** Measured on the isolated run:
64 `Xid 45` lines covering only **9** distinct channel ids (`Ch 1..9`), each reported 2–12
times, split exactly 32 attributed to the fatal `149` and 32 to the nonfatal `145`. So read
these as event counts; the number of distinct channels is far smaller.

The other three are device-level: `Xid 149` is the link event
itself, attributed to the **injector** rather than to any process harmed by it; `154` is the
recovery-action change (1 or 2 lines, depending on whether the intermediate `Drain and Reset`
is logged); `145` is a nonfatal receive-side error, one per affected GPU.

So XID location tells you where channels died, **not who dies**. Both 2026-08-18 validations
record `uncorrectable NVLink error detected` from all eight worker PIDs: 22 times per PID in
the first and 21–22 times per PID when validating the committed runner. Run 2 also records
the error from all eight PIDs; run 1 records it from seven identifiable PIDs, with the
remaining worker reporting the generic `unspecified launch failure`. All four runs end with
workers 0–7 exiting 1.

EP's rank-masking never saves the job. In run 1 one rank did report
`detected rank failures: {1,...,7}` before dying; in run 2 no rank got that far. That
difference is incidental timing, **not** a property of EP — do not read the run-1 masking as
partial success.

## Blast radius: mapping, not job membership

Measured 2026-08-11 with four cohorts around one injection. **You die if and only if you hold
a CUDA context or mapping involving the faulted GPU.** Being in the victim's job is neither
necessary nor sufficient:

| Cohort | GPUs | Context on victim? | Same job? | Outcome |
|---|---|---|---|---|
| EP ranks | 0,1,2,3 | yes, own + IPC peers | yes | all exit 1 |
| separate container, touches victim | 2 ↔ 3 | **yes** | **no** | **died**, CUDA 220 |
| separate container, disjoint | 6 ↔ 7 | no | no | survived, 0 stalls |
| **same container, disjoint** | 4 ↔ 5 | **no** | **yes** | **survived, 0 stalls** |

A process sharing the victim's container, cgroup, PID namespace, process group and
`--ipc=host` — pushing continuous NVLink traffic the whole time — finished 405,455 iterations
untouched with zero XIDs on its GPUs, while a process in a *different* container that merely
touched the faulted GPU died. Consistent with the harness: `elastic.py` uses
`spawn(join=False)` and joins raw processes, bypassing the `ProcessContext.join()` path that
would otherwise terminate siblings, so there is no automatic job-kill.

**`--gpus all` is itself a blast-radius decision.** A bystander that only ever names `cuda:4`
and `cuda:5` still died under `--gpus all`, because the CUDA runtime creates a ~616 MiB
primary context on *every visible device* during peer-access setup — so it held a mapping on
the victim after all and took 160 channel kills. Restricting device visibility is what
actually contains the damage.

Stable across the four 8-rank runs: all workers exit 1, only the victim GPU degrades, and
the `Xid 149` payload is identical. Both final archived runs resolve the earlier ambiguity:
all eight workers independently reported the explicit uncorrectable NVLink error.

## Cost and recovery

Each injection leaves the victim GPU at 0 links, `GPU Recovery Action: Reset`. Peer GPUs are
**not** degraded — they keep all 18 links and need nothing.

**In-band recovery does not work.** `nvidia-smi -r` refuses with "In use by another client"
even after stopping Fabric Manager, `nvidia-persistenced`, DCGM and NVSM. Recovery requires
a **BMC power cycle**, which clears all degraded GPUs at once:

```bash
ipmitool -I lanplus -H <bmc> -U <user> -P <pass> chassis power cycle   # wait ~7 min
```

So budget one power cycle per run, and verify all 8 GPUs are back at 18 links before the
next one — with any GPU degraded, `--gpus all` fails CUDA init with
`cudaErrorDevicesUnavailable`.

## Environment

Single node, Blackwell HGX 8-GPU (8x B200), NVLink 5, 18 links/GPU at 53.125 GB/s.
Driver `570.158.01` open kernel module and matching Fabric Manager, both **stock** — no
driver patch, firmware change, or module rebuild.