# GB300 MNNVL: real NVLink link-down during NIXL EP

Run 600 on 2026-09-14 used two GB300 nodes (`gb-wlake-17` and
`gb-wlake-18`), four GPUs/ranks per node, driver 580.173.02, and one MNNVL
clique. NIXL was built from main commit
`adf179a54a0951193bff0ab90201dfc605ed5d5c` plus this branch's `elastic.py`
fault hook.

**Result: forcing one link down on one GPU killed all eight ranks across both
nodes. Only the victim GPU was degraded.**

## Injection method

`nvlink_hwinject.c` opens `/dev/nvidiactl`, builds the RM
client/device/subdevice hierarchy, and issues:

```text
NV2080_CTRL_CMD_NVLINK_SET_HW_ERROR_INJECT (0x20803081)
link 0: errType=LINK_ERR, errSettings=FORCE_LINK_DOWN
```

This is a real hardware link teardown, not a simulated RAS event. GB300/R580
requires the 1048-byte parameter layout containing both the legacy `linkMask`
and `NV2080_CTRL_NVLINK_LINK_MASK`. Driver-version parsing is architecture
independent for Grace/aarch64.

The destructive ioctl requires real host root. NIXL EP remains unprivileged:
rank 6 atomically writes a validated request on node-local `/raid`; an
explicitly armed root controller on `gb-wlake-18` validates that request and
performs the ioctl. A second root helper captures peer-node kernel evidence.

## Timing relative to NIXL EP

The plan expands from ranks 0-3 to 0-7, then marks rank 6 for failure. Rank 6
is `gb-wlake-18` local rank/GPU 2.

`--fault-nvlink` changes only the existing fault callback:

```python
timer = threading.Timer(0.0001, FAULT_ACTION or self_kill)
```

The callback ran at rank 6's first dispatch in phase 2. All eight ranks had
entered phase 2 and none had completed it, so communication was in flight.

- Injection began 88.857 seconds after the node-18 application start.
- Request-to-injection delay: 43.4 ms.
- RM ioctl: 2.157 ms, `NV_OK`.
- First timestamped CUDA errors: +431.7 ms on node 17 and +913.4 ms on node 18.
- Both four-worker groups failed about 2.83 seconds after injection.

## Results and explanation

All eight workers exited 1 with `uncorrectable NVLink error detected`.
No rank logged `detected rank failures`, completed phase 2, or reached the
planned seven-rank contraction.

The victim GPU logged Xid 149 (`NETIR_LINK_DOWN`), Xid 154 (Recovery Action
became `Reset`), Xid 145, and Xid 45 channel kills. Although only link 0 was
targeted, all 18 links on that GPU became inactive.

Every non-victim GPU—three local and four on the remote node—logged the same
peer signature: Xid 145 once and Xid 45 56 times across 28 distinct channels.
All seven peers retained 18 links and Recovery Action `None`.

The blast radius follows CUDA IPC/NVLink mappings, not the server boundary.
When the victim left the MNNVL fabric, every rank holding its peer mapping
received an uncorrectable CUDA error. EP masking can recover a terminated
process, but not this event: the surviving processes' own CUDA contexts were
also poisoned. This matches the B200 CUDA-IPC result and extends it across a
physical node boundary.

After a cold BMC power cycle, all four victim-node GPUs returned to 18 links,
Recovery Action `None`, fabric `Completed`, and zero link error counters.
The peer node required no recovery.

## Reproduce

**Destructive: obtain cluster-owner approval and prepare victim-node cold
power-cycle recovery first. One run requires real root on both nodes.**

1. Check out this branch on shared storage. It is based on the exact tested
   main commit. Build NIXL EP for `sm_103` as described in
   `examples/device/ep/README.md`, and make the same aarch64 Pyxis/Enroot
   image available at the same `/raid` path on both nodes. Export:

   ```bash
   export NIXL_REPO_ROOT=$PWD
   export NIXL_INSTALL=/shared/path/nixl-install
   export NIXL_EP_IMAGE=/raid/users/$USER/img/pytorch-26.06-py3-arm64.sqsh
   ```

2. Build the injector on a GB300 host:

   ```bash
   cd examples/device/ep/tests/elastic/nvlink_fault_mnnvl
   gcc -O2 -Wall -Wextra -std=c11 -o nvlink_hwinject nvlink_hwinject.c
   ```

3. Submit with node 18 as Slurm node index 1:

   ```bash
   sbatch --nodelist=gb-wlake-17,gb-wlake-18 \
     examples/device/ep/tests/elastic/nvlink_fault_mnnvl/run_2node_mnnvl.sbatch
   ```

4. Enter one root shell on each allocated node and run the two exact helper
   commands printed in the job log. Injection cannot occur before both helpers
   report armed.
5. Review the result directory. Then run
   `sync && ipmitool -I open chassis power cycle` as root on the victim and
   verify all four victim GPUs have 18 links and Recovery Action `None`.
