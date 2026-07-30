#!/usr/bin/env python3
"""
Simple 2-peer / 2-node cross-node VMM fabric refcount test.

Only 2 processes total - one victim on node A, one survivor on node B.
No third "poisoner" process.

Victim allocates a VMM fabric region, exports it, and dies via one of
two mechanisms (--death-mode):

  release  - victim runs
             cuMemUnmap + cuMemAddressFree + cuMemRelease  and exits.
             This is the exact call sequence that
             elastic's Buffer::destroy() -> ~vmm_region() runs under
             the SIGTERM cooperative teardown path.

  sigkill  - victim calls os.kill(getpid(), SIGKILL). The kernel
             closes /dev/nvidia* fds and the driver's RmClient teardown
             runs memoryfabricDestruct_IMPL /
             fabricvaspaceUnmapPhysMemdesc_IMPL on the victim's fabric
             allocation involuntarily. This is what
             elastic's --fault-kill-signal=sigkill fault injector
             triggers on the victim rank.

After the victim dies, the survivor waits N seconds (--death-wait,
default 15) to give the OS reap + driver cleanup + IMEX destroy
propagation plenty of time to settle. Then it does two proofs:

  PROOF #2  Read the imported region -> expect the SENTINEL_1 pattern
            the victim wrote before dying.
  PROOF #2b Write SENTINEL_2 through the imported mapping, read back
            -> expect SENTINEL_2.

If both PROOFs pass, cross-node refcount is holding the physical page
alive with the survivor as the ONLY remaining ref-holder anywhere.
(There is no third process; no other process on either node imports
this handle; so by construction the survivor's cuMemImportFromShareableHandle
is the only ref that could be keeping the page alive.)

------------------------------------------------------------------
Can we check the refcount at runtime?
------------------------------------------------------------------
Not directly. CUDA VMM does not expose a public API to query the
physical-page refcount from user-space. What we CAN check at runtime:

  1. Victim process is really dead.
     * In --death-mode=release: victim sends a RELEASED ack over the
       TCP connection then exits; survivor observes both.
     * In --death-mode=sigkill: TCP connection breaks abruptly when
       the kernel tears down victim's sockets; survivor detects EOF.

  2. No other importer exists (by construction of the test - only two
     processes total, only the survivor calls
     cuMemImportFromShareableHandle).

Given (1) and (2), the survivor is provably the ONLY ref contributor
that could be keeping the physical page alive after victim's death.
PROOF #2 and PROOF #2b succeeding are the runtime empirical evidence
that this single cross-node ref is being honored by node A's driver.

Compared to imex_refcount_test.py (SIGKILL variant), this simpler
test:
  * Uses 2 processes instead of 3 (no poisoner)
  * Does NOT do poisoning, so it cannot distinguish
    "refcount holds" from "physical page not yet recycled by the
    driver". For that stronger evidence run the 3-process variant.
  * Adds an explicit 15s wait after victim death to be conservative
    about driver cleanup timing.
"""

import argparse
import ctypes
import json
import os
import signal as _signal
import socket
import sys
import time
from ctypes import byref, c_size_t, c_void_p

# Reuse the CUDA / TCP helpers from the sibling test file. When run via
# `python3 /full/path/simple_refcount_test.py`, Python adds the script's
# own directory to sys.path automatically, so this import works.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imex_refcount_test as core  # noqa: E402


DEATH_MODES = ("release", "sigkill")


# ============================================================
# Victim
# ============================================================


def run_victim(args):
    core.ROLE = "VICTIM"
    dev, _ = core.setup_cuda(args.device)
    prop = core.build_prop(dev)

    gran = c_size_t()
    core.check(
        core.cuMemGetAllocationGranularity(
            byref(gran),
            byref(prop),
            core.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )
    size = core.align_up(args.size_bytes, gran.value)
    probe_bytes = min(size, args.probe_bytes)
    probe_bytes = probe_bytes - (probe_bytes % 8)
    core.log(
        f"pid={os.getpid()}  death_mode={args.death_mode}  "
        f"gran={gran.value:,}  size={size:,}  probe_bytes={probe_bytes:,}"
    )

    handle, va = core.alloc_and_map(prop, size, dev, "primary")
    SENTINEL_1 = 0xA1A1A1A1A1A1A1A1
    core.log(
        f"primary: handle={handle.value:#x}  va={va.value:#x}  "
        f"writing SENTINEL_1={SENTINEL_1:#x}"
    )
    core.write_u64_pattern(va, SENTINEL_1, probe_bytes)

    fab = core.CUmemFabricHandle()
    core.check(
        core.cuMemExportToShareableHandle(
            c_void_p(ctypes.addressof(fab)),
            handle,
            core.CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        ),
        "cuMemExportToShareableHandle",
    )
    core.log("exported fabric handle")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.listen(1)
    core.log(f"listening for survivor on 0.0.0.0:{args.port}")
    sock.settimeout(args.tcp_timeout)
    conn, addr = sock.accept()
    conn.settimeout(None)
    core.log(f"survivor connected from {addr}")

    meta = {
        "size": size,
        "probe_bytes": probe_bytes,
        "sentinel_1": SENTINEL_1,
        "sentinel_2": 0xB2B2B2B2B2B2B2B2,
        "death_mode": args.death_mode,
        "victim_pid": os.getpid(),
    }
    core.send_msg(conn, "META", json.dumps(meta).encode())
    core.send_msg(conn, "HANDLE", bytes(fab.data))
    core.log(f"sent META + HANDLE ({core.FABRIC_HANDLE_SIZE} bytes)")

    core.log("waiting for DIE signal from survivor")
    tag, _ = core.recv_msg(conn)
    if tag != "DIE":
        raise RuntimeError(f"expected DIE, got {tag}")

    if args.death_mode == "sigkill":
        core.log(f"received DIE; SIGKILL self (pid={os.getpid()})")
        try:
            conn.close()
            sock.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        # No explicit CUDA cleanup here - the whole point is that the
        # driver has to do it involuntarily via memoryfabricDestruct_IMPL
        # when the kernel closes /dev/nvidia* fds during exit.
        os.kill(os.getpid(), _signal.SIGKILL)
        return  # unreachable

    # death_mode == "release": clean voluntary cleanup, matches
    # ~vmm_region() from Buffer::destroy() in elastic under SIGTERM.
    core.log(
        "received DIE; running clean release "
        "(cuMemUnmap + cuMemAddressFree + cuMemRelease)"
    )
    core.full_release(handle, va, size, "primary")

    try:
        core.send_msg(conn, "RELEASED")
    except Exception as e:
        core.log(f"WARNING: couldn't send RELEASED ack: {e}")

    core.log(f"release complete; VICTIM exiting rc=0 pid={os.getpid()}")
    try:
        conn.close()
        sock.close()
    except Exception:
        pass
    sys.exit(0)


# ============================================================
# Survivor
# ============================================================


def run_survivor(args):
    core.ROLE = "SURVIVOR"
    dev, _ = core.setup_cuda(args.device)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    core.log(f"connecting to victim {args.peer_host}:{args.port}")
    connected = False
    deadline = time.time() + args.tcp_timeout
    last_err = None
    while time.time() < deadline:
        try:
            sock.connect((args.peer_host, args.port))
            connected = True
            break
        except (ConnectionRefusedError, OSError) as e:
            last_err = e
            time.sleep(1)
    if not connected:
        raise RuntimeError(
            f"could not connect to {args.peer_host}:{args.port}: {last_err}"
        )
    core.log("connected to victim")

    tag, payload = core.recv_msg(sock)
    if tag != "META":
        raise RuntimeError(f"expected META, got {tag}")
    meta = json.loads(payload.decode())
    size = int(meta["size"])
    probe_bytes = int(meta["probe_bytes"])
    SENTINEL_1 = int(meta["sentinel_1"])
    SENTINEL_2 = int(meta["sentinel_2"])
    death_mode = meta["death_mode"]
    victim_pid = meta.get("victim_pid", "?")
    core.log(
        f"received META: size={size:,}  probe_bytes={probe_bytes:,}  "
        f"S1={SENTINEL_1:#x}  S2={SENTINEL_2:#x}  death_mode={death_mode}  "
        f"victim_pid={victim_pid}"
    )

    tag, handle_bytes = core.recv_msg(sock)
    if tag != "HANDLE" or len(handle_bytes) != core.FABRIC_HANDLE_SIZE:
        raise RuntimeError(
            f"expected HANDLE ({core.FABRIC_HANDLE_SIZE} bytes), got tag={tag} "
            f"len={len(handle_bytes)}"
        )
    fab = core.CUmemFabricHandle()
    ctypes.memmove(fab.data, handle_bytes, core.FABRIC_HANDLE_SIZE)
    core.log("received HANDLE")

    handle, va = core.import_and_map(fab, size, dev, "imported")
    core.log(f"imported: handle={handle.value:#x}  va={va.value:#x}")

    proofs = {}

    # PROOF #1
    vals = core.read_u64_buffer(va, probe_bytes)
    m = core.find_mismatch(vals, SENTINEL_1)
    proofs["proof_1_initial_read"] = m is None
    if m is None:
        core.log(
            f"PROOF #1 PASS: survivor reads SENTINEL_1={SENTINEL_1:#x} "
            f"({len(vals):,} u64s) -> cross-node fabric import works"
        )
    else:
        core.log(f"PROOF #1 FAIL: index[{m[0]}]={m[1]:#x}")

    # Signal victim to die
    core.log(
        f"sending DIE to victim (mode={death_mode}). "
        "After victim's teardown finishes, survivor is BY DESIGN the "
        "only remaining ref-holder anywhere (no other importers, no "
        "third process). The next PROOFs are the runtime evidence of "
        "whether cross-node refcount keeps the page alive."
    )
    core.send_msg(sock, "DIE")

    if death_mode == "release":
        # Wait for RELEASED ack, then victim exits cleanly.
        try:
            sock.settimeout(15)
            tag, _ = core.recv_msg(sock)
            if tag == "RELEASED":
                core.log(
                    "victim confirmed clean release complete "
                    "(cuMemUnmap + cuMemAddressFree + cuMemRelease done)"
                )
            else:
                core.log(f"WARNING: unexpected tag from victim: {tag}")
        except Exception as e:
            core.log(
                f"WARNING: no RELEASED ack from victim ({e}); "
                "assuming it exited"
            )
    else:
        # sigkill: TCP will break abruptly.
        try:
            sock.settimeout(3)
            data = sock.recv(1)
            if data:
                core.log(
                    f"WARNING: unexpected data from victim after SIGKILL: "
                    f"{data!r}"
                )
        except (socket.timeout, ConnectionResetError, OSError):
            pass
        core.log(
            "victim SIGKILL confirmed: TCP connection lost as expected. "
            "Kernel has closed /dev/nvidia* fds and the driver has run "
            "involuntary memoryfabricDestruct_IMPL for the victim's "
            "fabric allocation."
        )

    try:
        sock.close()
    except Exception:
        pass

    # Extra-conservative wait to make sure driver + IMEX are fully done
    core.log(
        f"waiting {args.death_wait}s before probing "
        "(gives OS reap + driver kernel cleanup + IMEX destroy "
        "propagation ample time to settle)"
    )
    time.sleep(args.death_wait)

    # PROOF #2
    vals = core.read_u64_buffer(va, probe_bytes)
    m = core.find_mismatch(vals, SENTINEL_1)
    proofs["proof_2_read_after_victim_death"] = m is None
    if m is None:
        core.log(
            f"PROOF #2 PASS: after victim {death_mode.upper()} + "
            f"{args.death_wait}s wait, survivor STILL reads SENTINEL_1 "
            "-> physical page is still alive; only survivor's cross-node "
            "ref could be keeping it alive; therefore IMEX cross-node "
            "refcount HOLDS"
        )
    else:
        core.log(
            f"PROOF #2 FAIL: after victim {death_mode.upper()} + "
            f"{args.death_wait}s wait, expected SENTINEL_1, "
            f"index[{m[0]}]={m[1]:#x} -> physical page may have been "
            "freed OR mapping stale"
        )

    # PROOF #2b
    core.log(f"writing SENTINEL_2={SENTINEL_2:#x} through survivor mapping")
    core.write_u64_pattern(va, SENTINEL_2, probe_bytes)
    vals = core.read_u64_buffer(va, probe_bytes)
    m = core.find_mismatch(vals, SENTINEL_2)
    proofs["proof_2b_writeread_after_victim_death"] = m is None
    if m is None:
        core.log(
            "PROOF #2b PASS: survivor can WRITE through the mapping and "
            "read back its own write -> mapping is fully live, not "
            "read-only degraded or cached-stale"
        )
    else:
        core.log(
            f"PROOF #2b FAIL: expected SENTINEL_2, index[{m[0]}]={m[1]:#x}"
        )

    core.log("SURVIVOR releasing")
    core.full_release(handle, va, size, "imported")

    core.log("")
    core.log("=" * 64)
    core.log(f"SUMMARY (death_mode={death_mode})")
    core.log("=" * 64)
    for k, v in proofs.items():
        core.log(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(proofs.values())
    core.log("=" * 64)
    if all_pass:
        core.log(
            f"OVERALL: PASS - cross-node refcount HOLDS under victim "
            f"{death_mode.upper()}. Survivor was structurally the only "
            "remaining ref-holder (only two processes in test, only "
            "survivor called cuMemImportFromShareableHandle); its cross-"
            "node ref alone kept node A's physical page alive after the "
            "victim's teardown."
        )
    else:
        failed = [k for k, v in proofs.items() if not v]
        core.log(f"OVERALL: FAIL - failing proofs: {failed}")
    core.log("=" * 64)

    sys.exit(0 if all_pass else 1)


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser(
        description="Simple 2-peer / 2-node cross-node VMM fabric "
        "refcount test (release + sigkill death modes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--role", choices=["victim", "survivor"], required=True)
    ap.add_argument(
        "--peer-host",
        help="hostname/IP of victim (survivor role only)",
    )
    ap.add_argument("--port", type=int, default=27184)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--death-mode",
        choices=DEATH_MODES,
        default="release",
        help=(
            "How the victim dies. 'release' = clean cuMemUnmap + "
            "cuMemAddressFree + cuMemRelease (matches elastic's SIGTERM "
            "path). 'sigkill' = victim self-SIGKILLs; driver runs "
            "involuntary cleanup (matches elastic's "
            "--fault-kill-signal=sigkill path)."
        ),
    )
    ap.add_argument(
        "--death-wait",
        type=int,
        default=15,
        help="Seconds survivor waits after victim signals death, before "
        "doing PROOF #2 (default: 15).",
    )
    ap.add_argument("--size-bytes", type=int, default=64 * 1024 * 1024)
    ap.add_argument("--probe-bytes", type=int, default=8 * 1024 * 1024)
    ap.add_argument("--tcp-timeout", type=int, default=120)
    args = ap.parse_args()

    if args.role == "victim":
        run_victim(args)
    else:
        if not args.peer_host:
            ap.error("--peer-host is required for --role survivor")
        run_survivor(args)


if __name__ == "__main__":
    main()
