"""Add an environment-driven fault to the container's elastic.py.

Run it after patch_rdma.py, which pins CUDA_VISIBLE_DEVICES to the rank's GPU.
Without NIXL_EP_FAULT elastic.py behaves as before. Otherwise each global rank
in NIXL_EP_FAULT_RANKS fires the fault at its first dispatch of phase
NIXL_EP_FAULT_PHASE, from the timer elastic.py uses for plan-driven kills:
  sigterm  the rank sends itself SIGTERM, the plan-driven kill path
  nvlink   the rank forces NVLink link NIXL_EP_FAULT_LINK of its own GPU down
           with NIXL_EP_FAULT_TOOL, which needs CAP_SYS_ADMIN
The plan keeps the faulted ranks active, so the other ranks keep sending to
them and have to detect the loss through their own timeouts.

NIXL_EP_FAULT_TIMING orders the nvlink fault against the first dispatch of the
phase on NIXL_EP_FAULT_NEIGHBOURS, the other ranks on the victim's node. The
ranks of one node share the container's /tmp, which carries the handshake:
  timer   the victim's timer starts the injector and the victim keeps running
  nvlink  the neighbours launch their first dispatch only after the injector
          has returned, so its NVLink sends go to the downed GPU
  rdma    the victim waits until the neighbours have launched their first
          dispatch, whose NVLink sends to it take about a millisecond, then
          injects before it launches its own; the neighbours wait for its data
In the nvlink and rdma timings the victim launches its own first dispatch only
after the injector returns.

The split timings act on the first dispatch of the phase that elastic.py runs
with return_recv_hook, which launches the send part as its own kernel and the
receive part from hook(). The victim holds that dispatch until the injector
returns. Each neighbour drains its GPU, launches the send part, waits for it
on the GPU and prints its wall-clock span, and calls hook() only after the
injector returns:
  before_send  the neighbours launch the send part after the injector returns
  after_send   the victim injects once every neighbour's send part has finished
  during_send  the neighbours launch the send part NIXL_EP_FAULT_SEND_OFFSET_MS
               after the victim starts the injector, aiming the drop into it
  before_all_send  before_send, and the other node's ranks also hold their send
               part until the injector returns, which they learn from a key the
               victim sets in elastic.py's TCP store
  before_send_masked  before_send, but the neighbours mask the victim first, so
               their send part writes to every rank except the victim
In two more split timings nobody holds the send part, and each rank of the
victim's node prints when its send and receive parts finish on the GPU, from
a thread that polls a CUDA event, so nothing waits for them:
  after_victim_send  the victim injects once its own send part has finished,
                     and holds its receive part until the injector returns
  after_recv         the victim injects once it and every neighbour have
                     finished the receive part; they all hold the next op
                     until the injector returns
NIXL_EP_FAULT_DRY_RUN=1 runs the injector's probe, which touches no link,
instead of the link-down.

For the timeline, every elastic.py print gets a " [t=<epoch seconds>]" suffix,
and each rank prints a line before every dispatch and combine of the
correctness loop and before the benchmark.
"""

import sys

path = "examples/device/ep/tests/elastic/elastic.py"
source = open(path).read()
marker = "# B200_TWO_NODE_FAULT"

if marker in source:
    print("patch_fault_2node: elastic.py already patched", flush=True)
    sys.exit(0)

self_kill_anchor = "    os.kill(os.getpid(), signal.SIGTERM)\n"
timer_anchor = "threading.Timer(0.0001, self_kill)"
timer_start_anchor = "                                timer.start()\n"
device_anchor = "    torch.cuda.set_device(0)\n"
test_main_anchor = "        test_main(\n"
fault_arg_anchor = "            fault_tolerance_test=kill_rank,\n"
dispatch_anchor = "                            packed_recv_x, packed_recv_count, handle, event, hook = (\n"
dispatch_end_anchor = (
    "                                )\n"
    "                            )\n"
    "                            hook() if return_recv_hook else event.current_stream_wait()\n"
)
combine_anchor = "                            combined_x, event, hook = buffer.combine(\n"
bench_anchor = "    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))\n"
rdma_marker = "# B200_TWO_NODE_RDMA_PIN"
anchors = (
    self_kill_anchor,
    timer_anchor,
    timer_start_anchor,
    device_anchor,
    test_main_anchor,
    fault_arg_anchor,
    dispatch_anchor,
    dispatch_end_anchor,
    combine_anchor,
    bench_anchor,
    rdma_marker,
)
missing = [anchor.strip() for anchor in anchors if source.count(anchor) != 1]
if missing:
    print(f"patch_fault_2node: ERROR anchors not found exactly once: {missing}", flush=True)
    sys.exit(2)

helpers = (
    "\n\n"
    f"{marker}\n"
    "import builtins\n"
    "\n"
    "FAULT_ACTION = None\n"
    'FAULT_TIMING = os.environ.get("NIXL_EP_FAULT_TIMING", "timer")\n'
    'FAULT_NEIGHBOURS = {int(r) for r in os.environ.get("NIXL_EP_FAULT_NEIGHBOURS", "").split(",") if r}\n'
    'FAULT_SYNC_DIR = "/tmp/nixl_ep_fault_sync"\n'
    'FAULT_STATE = {"active": False, "held": set()}\n'
    'SPLIT_TIMINGS = (\n'
    '    "before_send", "after_send", "during_send", "before_all_send", "before_send_masked", "after_victim_send", "after_recv"\n'
    ")\n"
    'FREE_TIMINGS = ("after_victim_send", "after_recv")\n'
    "\n\n"
    "def print(*args, **kwargs):\n"
    '    builtins.print(*args, f"[t={time.time():.6f}]", **kwargs)\n'
    "\n\n"
    "def fault_flag(name):\n"
    "    os.makedirs(FAULT_SYNC_DIR, exist_ok=True)\n"
    '    open(os.path.join(FAULT_SYNC_DIR, name), "w").close()\n'
    "\n\n"
    "def fault_wait(rank, name, what):\n"
    "    flag = os.path.join(FAULT_SYNC_DIR, name)\n"
    "    deadline = time.time() + 120\n"
    "    while not os.path.exists(flag):\n"
    "        if time.time() > deadline:\n"
    '            print(f"[rank {rank}] gave up waiting for {what}", flush=True)\n'
    "            return\n"
    "        time.sleep(0.0002)\n"
    "\n\n"
    "def split_wait(rank, name):\n"
    '    store = FAULT_STATE.get("store")\n'
    "    if store is None or rank in FAULT_NEIGHBOURS:\n"
    '        fault_wait(rank, name, f"the {name} flag")\n'
    "        return\n"
    "    deadline = time.time() + 120\n"
    '    while not store.check([f"nixl_ep_fault_{name}"]):\n'
    "        if time.time() > deadline:\n"
    '            print(f"[rank {rank}] gave up waiting for the {name} key", flush=True)\n'
    "            return\n"
    "        time.sleep(0.0002)\n"
    "\n\n"
    "def fault_hold(rank, when, hook):\n"
    '    if not FAULT_STATE["active"] or when in FAULT_STATE["held"]:\n'
    "        return\n"
    "    if FAULT_TIMING in SPLIT_TIMINGS:\n"
    "        if hook:\n"
    "            split_hold(rank, when)\n"
    "    elif rank in FAULT_NEIGHBOURS:\n"
    "        neighbour_hold(rank, when)\n"
    "\n\n"
    "def neighbour_hold(rank, when):\n"
    '    FAULT_STATE["held"].add(when)\n'
    '    if FAULT_TIMING == "nvlink" and when == "before":\n'
    '        print(f"[rank {rank}] holding the first dispatch until the injector returns", flush=True)\n'
    '        fault_wait(rank, "injected", "the injector")\n'
    '        print(f"[rank {rank}] injector returned; launching the first dispatch", flush=True)\n'
    '    elif FAULT_TIMING == "rdma" and when == "after":\n'
    '        fault_flag(f"launched_{rank}")\n'
    '        print(f"[rank {rank}] launched the first dispatch", flush=True)\n'
    "\n\n"
    "def split_hold(rank, when):\n"
    "    victim = FAULT_ACTION is not None\n"
    '    if not victim and rank not in FAULT_NEIGHBOURS and FAULT_TIMING != "before_all_send":\n'
    "        return\n"
    '    FAULT_STATE["held"].add(when)\n'
    "    if FAULT_TIMING in FREE_TIMINGS:\n"
    "        free_hold(rank, when, victim)\n"
    "        return\n"
    '    if when == "received":\n'
    "        return\n"
    "    if victim:\n"
    '        if when == "before":\n'
    '            fault_flag(f"ready_{rank}")\n'
    '            print(f"[rank {rank}] holding the send part until the injector returns", flush=True)\n'
    '            fault_wait(rank, "injected", "the injector")\n'
    '            print(f"[rank {rank}] injector returned; launching the send part", flush=True)\n'
    "        return\n"
    '    if when == "before":\n'
    '        if FAULT_TIMING == "before_send_masked":\n'
    '            for victim_rank in sorted(int(r) for r in os.environ["NIXL_EP_FAULT_RANKS"].split(",")):\n'
    '                FAULT_STATE["buffer"].update_mask_buffer(victim_rank, True)\n'
    '                print(f"[rank {rank}] masked rank {victim_rank} before the send part", flush=True)\n'
    "        torch.cuda.synchronize()\n"
    '        if FAULT_TIMING != "after_send":\n'
    '            release = "injected" if FAULT_TIMING in ("before_send", "before_all_send", "before_send_masked") else "go"\n'
    '            fault_flag(f"ready_{rank}")\n'
    '            print(f"[rank {rank}] holding the send part until the {release} flag", flush=True)\n'
    "            split_wait(rank, release)\n"
    '            print(f"[rank {rank}] released; launching the send part", flush=True)\n'
    "        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)\n"
    "        start.record()\n"
    '        FAULT_STATE["send"] = (start, end, time.time())\n'
    "        return\n"
    '    start, end, launched = FAULT_STATE["send"]\n'
    "    end.record()\n"
    "    torch.cuda.synchronize()\n"
    "    print(\n"
    '        f"[rank {rank}] send part: launched {launched:.6f}, finished by {time.time():.6f}, "\n'
    '        f"GPU time {start.elapsed_time(end):.3f} ms",\n'
    "        flush=True,\n"
    "    )\n"
    '    if FAULT_TIMING == "after_send":\n'
    '        fault_flag(f"sent_{rank}")\n'
    '    split_wait(rank, "injected")\n'
    '    print(f"[rank {rank}] launching the receive part", flush=True)\n'
    "\n\n"
    "def gpu_mark():\n"
    "    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)\n"
    "    start.record()\n"
    "    return start, end, time.time()\n"
    "\n\n"
    "def gpu_watch(rank, name, start, end, launched):\n"
    "    end.record()\n"
    "\n"
    "    def poll():\n"
    "        deadline = time.time() + 90\n"
    "        while not end.query():\n"
    "            if time.time() > deadline:\n"
    '                print(f"[rank {rank}] {name} not finished on the GPU after 90 s", flush=True)\n'
    "                return\n"
    "            time.sleep(0.00005)\n"
    "        print(\n"
    '            f"[rank {rank}] {name}: launched {launched:.6f}, finished by {time.time():.6f}, "\n'
    '            f"GPU time {start.elapsed_time(end):.3f} ms",\n'
    "            flush=True,\n"
    "        )\n"
    "\n"
    "    threading.Thread(target=poll, daemon=True).start()\n"
    "\n\n"
    "def free_hold(rank, when, victim):\n"
    '    if when == "before":\n'
    "        torch.cuda.synchronize()\n"
    '        print(f"[rank {rank}] launching the send part", flush=True)\n'
    '        FAULT_STATE["send"] = gpu_mark()\n'
    "        return\n"
    '    if when == "after":\n'
    '        gpu_watch(rank, "send part", *FAULT_STATE["send"])\n'
    '        if victim and FAULT_TIMING == "after_victim_send":\n'
    "            torch.cuda.synchronize()\n"
    '            fault_flag(f"sent_{rank}")\n'
    '            print(f"[rank {rank}] own send part done; holding the receive part until the injector returns", flush=True)\n'
    '            fault_wait(rank, "injected", "the injector")\n'
    '        print(f"[rank {rank}] launching the receive part", flush=True)\n'
    '        FAULT_STATE["recv"] = gpu_mark()\n'
    "        return\n"
    '    gpu_watch(rank, "receive part", *FAULT_STATE["recv"])\n'
    '    if FAULT_TIMING == "after_recv":\n'
    "        torch.cuda.synchronize()\n"
    '        fault_flag(f"received_{rank}")\n'
    '        print(f"[rank {rank}] receive part done; holding the next op until the injector returns", flush=True)\n'
    '        fault_wait(rank, "injected", "the injector")\n'
    '        print(f"[rank {rank}] injector returned; continuing", flush=True)\n'
    "\n\n"
    "def timed_fault(rank, inject):\n"
    '    if FAULT_TIMING == "rdma":\n'
    "        for neighbour in sorted(FAULT_NEIGHBOURS):\n"
    '            fault_wait(rank, f"launched_{neighbour}", f"rank {neighbour} to launch its first dispatch")\n'
    "        time.sleep(0.005)\n"
    "    elif FAULT_TIMING in SPLIT_TIMINGS:\n"
    '        if FAULT_TIMING == "after_victim_send":\n'
    '            flags = [f"sent_{rank}"]\n'
    '        elif FAULT_TIMING == "after_recv":\n'
    '            flags = [f"received_{r}" for r in sorted(FAULT_NEIGHBOURS | {rank})]\n'
    "        else:\n"
    '            sent = "sent" if FAULT_TIMING == "after_send" else "ready"\n'
    '            flags = [f"ready_{rank}"] + [f"{sent}_{neighbour}" for neighbour in sorted(FAULT_NEIGHBOURS)]\n'
    "        for name in flags:\n"
    '            fault_wait(rank, name, f"the {name} flag")\n'
    '        if FAULT_TIMING == "during_send":\n'
    '            offset_ms = float(os.environ.get("NIXL_EP_FAULT_SEND_OFFSET_MS", "10"))\n'
    '            print(f"[rank {rank}] releasing the neighbours\' send part {offset_ms} ms from now", flush=True)\n'
    '            threading.Timer(offset_ms / 1000, fault_flag, ("go",)).start()\n'
    "    try:\n"
    "        inject()\n"
    "    finally:\n"
    '        fault_flag("injected")\n'
    '        if FAULT_STATE.get("store") is not None:\n'
    '            FAULT_STATE["store"].set("nixl_ep_fault_injected", "1")\n'
    "\n\n"
    "def nvlink_link_down(rank, uuid, link, tool):\n"
    "    import subprocess\n"
    "\n"
    '    if os.environ.get("NIXL_EP_FAULT_DRY_RUN") == "1":\n'
    '        print(f"[rank {rank}] DRY RUN: probing {uuid}; no link is touched", flush=True)\n'
    "        command = [tool, uuid]\n"
    "    else:\n"
    '        print(f"[rank {rank}] forcing NVLink link {link} down on {uuid}", flush=True)\n'
    '        command = [tool, uuid, "down", str(link)]\n'
    "    env = dict(os.environ)\n"
    '    env.pop("CUDA_VISIBLE_DEVICES", None)\n'
    "    returncode = subprocess.run(command, env=env).returncode\n"
    '    print(f"[rank {rank}] injector exit code {returncode}", flush=True)\n'
)
arming = (
    '    if FAULT_TIMING == "before_all_send":\n'
    '        FAULT_STATE["store"] = store_group.create_client_store(master_addr=server_addr, port=TCP_STORE_PORT)\n'
    '    _fault = os.environ.get("NIXL_EP_FAULT", "none")\n'
    '    _fault_ranks = {int(r) for r in os.environ.get("NIXL_EP_FAULT_RANKS", "").split(",") if r}\n'
    '    if _fault != "none" and global_rank in _fault_ranks:\n'
    "        global FAULT_ACTION\n"
    '        _fault_gpu = os.environ["CUDA_VISIBLE_DEVICES"]\n'
    '        _fault_target = f"GPU {_fault_gpu}"\n'
    '        if _fault == "sigterm":\n'
    "            FAULT_ACTION = self_kill\n"
    '        elif _fault == "nvlink":\n'
    "            import subprocess\n"
    "\n"
    "            _fault_uuid = subprocess.check_output(\n"
    '                ["nvidia-smi", "-i", _fault_gpu, "--query-gpu=uuid", "--format=csv,noheader"],\n'
    "                text=True,\n"
    "            ).strip()\n"
    "            FAULT_ACTION = partial(\n"
    "                nvlink_link_down,\n"
    "                global_rank,\n"
    "                _fault_uuid,\n"
    '                int(os.environ["NIXL_EP_FAULT_LINK"]),\n'
    '                os.environ["NIXL_EP_FAULT_TOOL"],\n'
    "            )\n"
    '            if FAULT_TIMING != "timer":\n'
    "                FAULT_ACTION = partial(timed_fault, global_rank, FAULT_ACTION)\n"
    '            _fault_target += f" ({_fault_uuid})"\n'
    "        else:\n"
    '            raise ValueError(f"unknown NIXL_EP_FAULT={_fault}")\n'
    "        print(\n"
    '            f"[rank {global_rank}] armed {_fault} fault on {_fault_target} "\n'
    '            f"for phase {os.environ[\'NIXL_EP_FAULT_PHASE\']}, timing {FAULT_TIMING}",\n'
    "            flush=True,\n"
    "        )\n"
    '    if global_rank in FAULT_NEIGHBOURS and FAULT_TIMING != "timer":\n'
    "        print(\n"
    '            f"[rank {global_rank}] neighbour of the fault rank, timing {FAULT_TIMING} "\n'
    '            f"for phase {os.environ[\'NIXL_EP_FAULT_PHASE\']}",\n'
    "            flush=True,\n"
    "        )\n"
)
fault_arg = (
    "            fault_tolerance_test=kill_rank\n"
    "            or (\n"
    "                FAULT_ACTION is not None\n"
    '                and plan.get_phase() == int(os.environ["NIXL_EP_FAULT_PHASE"])\n'
    "            ),\n"
)
dispatch_end = (
    "                                )\n"
    "                            )\n"
    '                            fault_hold(rank, "after", return_recv_hook)\n'
    "                            hook() if return_recv_hook else event.current_stream_wait()\n"
    '                            fault_hold(rank, "received", return_recv_hook)\n'
)

source = source.replace(self_kill_anchor, self_kill_anchor + helpers, 1)
source = source.replace(timer_anchor, "threading.Timer(0.0001, FAULT_ACTION or self_kill)", 1)
source = source.replace(
    timer_start_anchor,
    timer_start_anchor
    + '                                if FAULT_TIMING in ("nvlink", "rdma"):\n'
    + "                                    timer.join()\n",
    1,
)
source = source.replace(device_anchor, device_anchor + arming, 1)
source = source.replace(
    test_main_anchor,
    '        FAULT_STATE["active"] = plan.get_phase() == int(os.environ.get("NIXL_EP_FAULT_PHASE", "-1"))\n'
    '        FAULT_STATE["buffer"] = buffer\n'
    + test_main_anchor,
    1,
)
source = source.replace(fault_arg_anchor, fault_arg, 1)
source = source.replace(
    dispatch_anchor,
    '                            fault_hold(rank, "before", return_recv_hook)\n'
    '                            print(f"[rank {rank}] dispatch {num_times}.{i}", flush=True)\n'
    + dispatch_anchor,
    1,
)
source = source.replace(dispatch_end_anchor, dispatch_end, 1)
source = source.replace(
    combine_anchor,
    '                            print(f"[rank {rank}] combine {num_times}.{int(zero_copy)}", flush=True)\n'
    + combine_anchor,
    1,
)
source = source.replace(bench_anchor, '    print(f"[rank {rank}] benchmark", flush=True)\n' + bench_anchor, 1)
open(path, "w").write(source)
print("patch_fault_2node: added the NIXL_EP_FAULT trigger and timeline prints to elastic.py", flush=True)
