"""In-container patch: add NVLink fault injection + single-NIC RDMA (rc_gda)
pinning to the image's OWN elastic.py, so it stays compatible with the compiled
nixl_ep in nixl-ep:master (do NOT mount the fault-branch elastic.py - its API
differs). Idempotent.

NIXL_EP_NIC selects the one HCA used by every rank (default: mlx5_4).
NIXL_EP_GDA=1 adds its GPUDirect device ("cuda0-<nic>") for the rc_gda path.

Injection is env-driven so no argparse changes are needed:
  NIXL_EP_FAULT_NVLINK=1, NIXL_EP_FAULT_RANK, NIXL_EP_FAULT_PHASE,
  NIXL_EP_FAULT_LINK, NIXL_EP_FAULT_TOOL.
The rank/phase trigger is independent of negative-rank plan entries, allowing an
all-active plan to test whether an RDMA-only workload survives the link fault.
The victim rank runs nvlink_hwinject on its OWN physical GPU (local_rank % 8);
CUDA_VISIBLE_DEVICES is cleared for that subprocess because nvlink_hwinject
addresses GPUs by physical RM index.
"""
import sys

PATH = "examples/device/ep/tests/elastic/elastic.py"
src = open(PATH).read()

if "def nvlink_link_down" in src:
    print("patch_fault: elastic.py already patched", flush=True)
    sys.exit(0)

# 1) helper + FAULT_ACTION right after self_kill's body
anchor_selfkill = "    os.kill(os.getpid(), signal.SIGTERM)\n"
helper = (
    "\n\nFAULT_ACTION = None\n\n\n"
    "def nvlink_link_down(gpu, link, tool):\n"
    "    import os as _os, subprocess as _sp\n"
    "    print(f'[gpu {gpu}] forcing NVLink link {link} down', flush=True)\n"
    "    _env = dict(_os.environ)\n"
    "    _env.pop('CUDA_VISIBLE_DEVICES', None)\n"
    "    _sp.run([tool, str(gpu), 'down', str(link)], env=_env, check=False)\n"
)
if anchor_selfkill not in src:
    print("patch_fault: ERROR self_kill anchor not found", flush=True)
    sys.exit(2)
src = src.replace(anchor_selfkill, anchor_selfkill + helper, 1)

# 2) swap the kill timer to prefer the injector
anchor_timer = "threading.Timer(0.0001, self_kill)"
if anchor_timer not in src:
    print("patch_fault: ERROR timer anchor not found", flush=True)
    sys.exit(3)
src = src.replace(anchor_timer, "threading.Timer(0.0001, FAULT_ACTION or self_kill)", 1)

# 3) after set_device(0): pin per-rank adjacent-NIC rc_gda and arm FAULT_ACTION
# only in the explicitly selected victim process.
anchor_setdev = "    torch.cuda.set_device(0)\n"
if anchor_setdev not in src:
    print("patch_fault: ERROR set_device anchor not found", flush=True)
    sys.exit(4)
inject = (
    '    _nic = os.environ.get("NIXL_EP_NIC", "mlx5_4")\n'
    '    # Always pin a valid NIC: NIXL EP creates a UCX backend even on the NVLink/cuda_ipc\n'
    '    # path. GDA device (cuda0-<nic>) only for the rc_gda data path; cuda_ipc is unaffected.\n'
    '    if os.environ.get("NIXL_EP_GDA") == "1" and os.environ.get("NIXL_EP_NVLINK_DEFAULT") != "1":\n'
    '        os.environ["UCX_NET_DEVICES"] = f"{_nic}:1,cuda0-{_nic}:1"\n'
    '    else:\n'
    '        os.environ["UCX_NET_DEVICES"] = f"{_nic}:1"\n'
    '    if (os.environ.get("NIXL_EP_FAULT_NVLINK") == "1"\n'
    '            and global_rank == int(os.environ.get("NIXL_EP_FAULT_RANK", "2"))):\n'
    '        global FAULT_ACTION\n'
    '        import functools\n'
    '        FAULT_ACTION = functools.partial(\n'
    '            nvlink_link_down,\n'
    '            local_rank % 8,\n'
    '            int(os.environ.get("NIXL_EP_FAULT_LINK", "0")),\n'
    '            os.environ.get("NIXL_EP_FAULT_TOOL", "/tools/nvlink_hwinject"),\n'
    '        )\n'
)
src = src.replace(anchor_setdev, anchor_setdev + inject, 1)

# 4) Trigger the injector in its requested phase even when the plan does not
# mark the victim negative. The original negative-rank/SIGTERM behavior remains.
anchor_fault_arg = "            fault_tolerance_test=kill_rank,\n"
if anchor_fault_arg not in src:
    print("patch_fault: ERROR fault trigger anchor not found", flush=True)
    sys.exit(5)
replacement_fault_arg = (
    "            fault_tolerance_test=kill_rank or (\n"
    "                FAULT_ACTION is not None\n"
    "                and plan.get_phase() == int(os.environ.get(\"NIXL_EP_FAULT_PHASE\", \"1\"))\n"
    "            ),\n"
)
src = src.replace(anchor_fault_arg, replacement_fault_arg, 1)

open(PATH, "w").write(src)
print("patch_fault: patched independent fault trigger + adjacent-NIC rc_gda pinning into elastic.py", flush=True)
