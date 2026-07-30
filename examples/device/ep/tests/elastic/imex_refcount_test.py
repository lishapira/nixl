#!/usr/bin/env python3
"""
Cross-node VMM fabric refcount test.

Purpose
-------
Empirically verify that when the *exporter* of a VMM fabric-handle
allocation releases (cuMemUnmap + cuMemAddressFree + cuMemRelease),
but an *importer* on a different node still holds a live
cuMemImportFromShareableHandle reference, the physical page stays
alive. This is the cross-node analog of the intranode VMM refcount
guarantee, and it depends on IMEX propagating destroy-refcount
information between nodes.

Requires
--------
* Two ranks on two different nodes in the same MNNVL clique.
  Verify with `mnnvl_probe.py` (same ClusterUUID + CliqueId).
* CUDA driver with FABRIC handle-type support
  (CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED == 1).
* IMEX daemon running on both nodes.

Design (both ranks execute concurrently)
----------------------------------------
Rank 0 (EXPORTER, node A):
  E1. cuMemCreate(FABRIC) + reserve VA + map + setAccess
  E2. Write SENTINEL_1 to the region
  E3. cuMemExportToShareableHandle -> raw bytes -> TCP -> rank 1
  E4. BARRIER importer_ready
  E5. cuMemUnmap + cuMemAddressFree + cuMemRelease  ← EXPORTER DROPS
  E6. BARRIER exporter_released
  E7. BARRIER importer_proof2_done
  E8. POISONING: allocate `--poison-count` new fabric regions of the
      same size, each filled with a distinct CANARY pattern, to try
      to force the driver to reuse the freshly-released physical
      page if the cross-node refcount did NOT protect it.
  E9. BARRIER poisoning_done
  E10. BARRIER importer_proof3_done
  E11. Release poison allocations
  E12. BARRIER exporter_cleanup_done, exit

Rank 1 (IMPORTER, node B):
  I1. Receive shareable handle over TCP
  I2. cuMemImportFromShareableHandle(FABRIC) + reserve VA + map + setAccess
  I3. Read the region -> expect SENTINEL_1  [PROOF #1: import works]
  I4. BARRIER importer_ready
  I5. BARRIER exporter_released
  I6. Read the region -> expect SENTINEL_1  [PROOF #2:
      refcount holds through exporter release]
  I7. Write SENTINEL_2, read back, expect SENTINEL_2
      [PROOF #2b: mapping is fully usable, not just cached-stale]
  I8. BARRIER importer_proof2_done
  I9. BARRIER poisoning_done
  I10. Read the region -> expect SENTINEL_2  [PROOF #3: physical page
       was NOT reused despite exporter's poisoning allocations]
  I11. BARRIER importer_proof3_done
  I12. cuMemUnmap + cuMemAddressFree + cuMemRelease
  I13. BARRIER exporter_cleanup_done, print SUMMARY, exit(rc)

Interpretation
--------------
* PROOF #1 fail   -> import path broken (probably not-in-clique or
                     IMEX misconfigured); test is inconclusive for
                     refcount.
* PROOF #2 fail   -> refcount likely BROKEN: page freed by exporter's
                     release even though importer still held a ref.
* PROOF #2b fail  -> write-through path broken; unusual.
* PROOF #3 fail   -> refcount BROKEN: exporter's new allocations
                     reused the "freed" physical page and clobbered
                     importer's still-mapped view.
* All PROOFs pass -> cross-node refcount HOLDS for this cluster/driver
                     under exporter-drop-first semantics.

Notes on sensitivity
--------------------
PROOF #3 depends on the driver actually reusing the freed physical
page during poisoning. On a mostly-idle GPU with many free pages, the
driver may pick a different page — in that case, the test can be a
false-negative (i.e., can't distinguish "refcount holds" from
"driver just didn't pick this page"). Bumping --poison-count helps.
The negative control for this test is to run the same flow on a
single node (importer on same node) and confirm PROOF #3 also passes;
if it doesn't, the sensitivity for the cross-node variant is
questionable.

Usage
-----
Rank 0 (exporter, on node A):
    python3 imex_refcount_test.py --role exporter \
        --port 27182 --size-bytes $((64*1024*1024)) --poison-count 16

Rank 1 (importer, on node B):
    python3 imex_refcount_test.py --role importer \
        --peer-host <node-A-hostname> --port 27182 \
        --size-bytes $((64*1024*1024)) --poison-count 16

Both ranks must be given the same --size-bytes and --poison-count.
Importer exits 0 on all-proofs-pass, non-zero otherwise.
"""

import argparse
import ctypes
import json
import os
import signal
import socket
import struct
import sys
import time
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_int,
    c_size_t,
    c_ubyte,
    c_uint,
    c_ulonglong,
    c_ushort,
    c_void_p,
    cast,
)


# ============================================================
# CUDA driver enums (values from cuda.h / driver_types.h)
# ============================================================

CUDA_SUCCESS = 0
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_HANDLE_TYPE_FABRIC = 8
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 132

FABRIC_HANDLE_SIZE = 64  # NV_MEM_HANDLE_TYPE_FABRIC_HANDLE_SIZE


# ============================================================
# CUDA structs
# ============================================================


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", c_int), ("id", c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", c_ubyte),
        ("gpuDirectRDMACapable", c_ubyte),
        ("usage", c_ushort),
        ("reserved", c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", c_int)]


class CUmemFabricHandle(ctypes.Structure):
    _fields_ = [("data", c_ubyte * FABRIC_HANDLE_SIZE)]


# ============================================================
# libcuda binding (lazy: only loaded when we actually run the test,
# so --help works even on the login node without libcuda installed)
# ============================================================

_cuda = None  # ctypes.CDLL, loaded by _load_libcuda()

# ctypes function handles, all populated by _load_libcuda()
cuInit = None
cuDeviceGet = None
cuDeviceGetAttribute = None
cuCtxCreate = None
cuCtxSynchronize = None
cuGetErrorName = None
cuGetErrorString = None
cuMemGetAllocationGranularity = None
cuMemCreate = None
cuMemAddressReserve = None
cuMemMap = None
cuMemSetAccess = None
cuMemUnmap = None
cuMemAddressFree = None
cuMemRelease = None
cuMemExportToShareableHandle = None
cuMemImportFromShareableHandle = None
cuMemcpyHtoD = None
cuMemcpyDtoH = None


def _load_libcuda():
    """Load libcuda.so.1 and bind all needed functions. Idempotent.

    Tries hard to load the *driver-injected* libcuda (matches the host's
    kernel driver), not any older stub bundled in the container that would
    trigger CUDA_ERROR_SYSTEM_DRIVER_MISMATCH (rc=803).

    Strategy:
      1. If torch is importable, initialize it (torch resolves the correct
         driver-injected libcuda internally via its CUDA runtime probing).
         After torch's init, ctypes.CDLL("libcuda.so.1") will find the same
         one already loaded in the process.
      2. Otherwise, try a list of known driver-injection paths in order.
    """
    global _cuda
    global cuInit, cuDeviceGet, cuDeviceGetAttribute, cuCtxCreate
    global cuCtxSynchronize, cuGetErrorName, cuGetErrorString
    global cuMemGetAllocationGranularity, cuMemCreate, cuMemAddressReserve
    global cuMemMap, cuMemSetAccess, cuMemUnmap, cuMemAddressFree, cuMemRelease
    global cuMemExportToShareableHandle, cuMemImportFromShareableHandle
    global cuMemcpyHtoD, cuMemcpyDtoH

    if _cuda is not None:
        return

    # Step 1: try to prime the right libcuda via torch.
    torch_primed = False
    try:
        import torch  # noqa: F401

        # torch.cuda.init() loads the driver-injected libcuda and calls cuInit
        # under the hood. Any subsequent ctypes.CDLL("libcuda.so.1") picks up
        # the same handle that was resolved by torch.
        if torch.cuda.is_available():
            torch.cuda.init()
            torch_primed = True
    except Exception as e:
        # torch import/init is best-effort; we'll fall through to path search.
        print(
            f"WARN: torch prime failed ({type(e).__name__}: {e}); "
            "will try direct libcuda paths",
            file=sys.stderr,
        )

    # Step 2: try to load libcuda.so.1, preferring driver-injected paths.
    # Driver-injected paths come FIRST so we don't accidentally load an
    # older bundled libcuda from /usr/local/cuda/compat/lib.real (that stub
    # returns CUDA_ERROR_SYSTEM_DRIVER_MISMATCH / rc=803 when the host
    # driver is newer). The CORRECT fix, though, is to run inside a shell
    # where /workspace/lishapira/setup_node.sh has been sourced -- it
    # strips /cuda/compat from LD_LIBRARY_PATH so the loader picks the
    # right one via the default path resolution.
    candidate_paths = [
        # GB200 / Grace-Hopper / other AArch64 CUDA on Ubuntu (host-injected)
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        # x86_64 CUDA on Ubuntu / Debian (host-injected)
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        # RHEL / CentOS
        "/usr/lib64/libcuda.so.1",
        # enroot / some Docker configs
        "/usr/local/nvidia/lib64/libcuda.so.1",
        # loader default (uses LD_LIBRARY_PATH / ldconfig -- last, because
        # LD_LIBRARY_PATH may still point at the /cuda/compat stub if the
        # caller forgot to source setup_node.sh)
        "libcuda.so.1",
    ]

    last_err = None
    for path in candidate_paths:
        try:
            _cuda = ctypes.CDLL(path)
            # bind cuInit + cuGetErrorString early so we can probe
            _cuda.cuInit.argtypes = [c_uint]
            _cuda.cuInit.restype = c_int
            _cuda.cuGetErrorString.argtypes = [c_int, POINTER(c_char_p)]
            _cuda.cuGetErrorString.restype = c_int

            # Quick probe: cuInit(0). Skip if torch already inited (calling it
            # twice is harmless per CUDA docs).
            rc = _cuda.cuInit(0)
            if rc == 0:
                print(
                    f"INFO: libcuda loaded from '{path}'"
                    + (" (torch-primed)" if torch_primed else ""),
                    file=sys.stderr,
                )
                break
            else:
                # get name of error
                desc = c_char_p()
                _cuda.cuGetErrorString(rc, byref(desc))
                d = desc.value.decode() if desc.value else "unknown"
                last_err = f"cuInit rc={rc} ({d}) from '{path}'"
                print(f"WARN: {last_err}; trying next", file=sys.stderr)
                _cuda = None
        except OSError as e:
            last_err = f"{path}: {e}"
            continue

    if _cuda is None:
        raise RuntimeError(
            "cannot load a working libcuda.so.1 from any of: "
            f"{candidate_paths}. Last error: {last_err}. "
            "Likely a driver/runtime mismatch inside the container; try "
            "running `nvidia-smi` in the container to check driver visibility."
        )

    def _bind(name, argtypes, restype=c_int):
        fn = getattr(_cuda, name)
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    cuInit = _bind("cuInit", [c_uint])
    cuDeviceGet = _bind("cuDeviceGet", [POINTER(c_int), c_int])
    cuDeviceGetAttribute = _bind(
        "cuDeviceGetAttribute", [POINTER(c_int), c_int, c_int]
    )
    cuCtxCreate = _bind("cuCtxCreate_v2", [POINTER(c_void_p), c_uint, c_int])
    cuCtxSynchronize = _bind("cuCtxSynchronize", [])
    cuGetErrorName = _bind("cuGetErrorName", [c_int, POINTER(c_char_p)])
    cuGetErrorString = _bind("cuGetErrorString", [c_int, POINTER(c_char_p)])

    cuMemGetAllocationGranularity = _bind(
        "cuMemGetAllocationGranularity",
        [POINTER(c_size_t), POINTER(CUmemAllocationProp), c_int],
    )
    cuMemCreate = _bind(
        "cuMemCreate",
        [POINTER(c_ulonglong), c_size_t, POINTER(CUmemAllocationProp), c_ulonglong],
    )
    cuMemAddressReserve = _bind(
        "cuMemAddressReserve",
        [POINTER(c_ulonglong), c_size_t, c_size_t, c_ulonglong, c_ulonglong],
    )
    cuMemMap = _bind(
        "cuMemMap", [c_ulonglong, c_size_t, c_size_t, c_ulonglong, c_ulonglong]
    )
    cuMemSetAccess = _bind(
        "cuMemSetAccess",
        [c_ulonglong, c_size_t, POINTER(CUmemAccessDesc), c_size_t],
    )
    cuMemUnmap = _bind("cuMemUnmap", [c_ulonglong, c_size_t])
    cuMemAddressFree = _bind("cuMemAddressFree", [c_ulonglong, c_size_t])
    cuMemRelease = _bind("cuMemRelease", [c_ulonglong])
    cuMemExportToShareableHandle = _bind(
        "cuMemExportToShareableHandle",
        [c_void_p, c_ulonglong, c_int, c_ulonglong],
    )
    cuMemImportFromShareableHandle = _bind(
        "cuMemImportFromShareableHandle",
        [POINTER(c_ulonglong), c_void_p, c_int],
    )
    cuMemcpyHtoD = _bind("cuMemcpyHtoD_v2", [c_ulonglong, c_void_p, c_size_t])
    cuMemcpyDtoH = _bind("cuMemcpyDtoH_v2", [c_void_p, c_ulonglong, c_size_t])


def cuda_err_str(err):
    name = c_char_p()
    desc = c_char_p()
    cuGetErrorName(err, byref(name))
    cuGetErrorString(err, byref(desc))
    n = name.value.decode() if name.value else "UNKNOWN"
    d = desc.value.decode() if desc.value else "no description"
    return f"{n}: {d}"


def check(err, op):
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"CUDA {op} failed (rc={err}): {cuda_err_str(err)}")


# ============================================================
# TCP sync helpers (framed messages)
# ============================================================


def _send_all(sock, buf):
    view = memoryview(buf)
    while view:
        n = sock.send(view)
        if n == 0:
            raise ConnectionError("send returned 0")
        view = view[n:]


def _recv_all(sock, n):
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise ConnectionError("peer closed connection")
        out.extend(chunk)
    return bytes(out)


# Tag field: 32 bytes fixed. Long enough to hold every "BAR:<name>" tag
# used by the barriers below (longest is "BAR:exporter_cleanup_done" = 25).
# Header = 32-byte zero-padded tag + 4-byte big-endian payload length.
_TAG_LEN = 32
_HDR_LEN = _TAG_LEN + 4
_HDR_FMT = f"!{_TAG_LEN}sI"


def send_msg(sock, tag, payload=b""):
    tag_bytes = tag.encode()
    if len(tag_bytes) > _TAG_LEN:
        raise ValueError(
            f"tag '{tag}' is {len(tag_bytes)} bytes, exceeds "
            f"tag field size {_TAG_LEN}"
        )
    tag_b = tag_bytes.ljust(_TAG_LEN, b"\x00")
    hdr = struct.pack(_HDR_FMT, tag_b, len(payload))
    _send_all(sock, hdr + payload)


def recv_msg(sock):
    hdr = _recv_all(sock, _HDR_LEN)
    tag_raw, length = struct.unpack(_HDR_FMT, hdr)
    tag = tag_raw.rstrip(b"\x00").decode()
    payload = _recv_all(sock, length) if length > 0 else b""
    return tag, payload


def wait_barrier(sock, name):
    log(f"barrier '{name}' waiting")
    send_msg(sock, f"BAR:{name}")
    tag, _ = recv_msg(sock)
    if tag != f"BAR:{name}":
        raise RuntimeError(f"expected BAR:{name}, got {tag}")
    log(f"barrier '{name}' released")


# ============================================================
# CUDA helpers
# ============================================================


def setup_cuda(device_id):
    _load_libcuda()
    check(cuInit(0), "cuInit")
    dev = c_int()
    check(cuDeviceGet(byref(dev), device_id), "cuDeviceGet")
    fab = c_int()
    check(
        cuDeviceGetAttribute(
            byref(fab), CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev
        ),
        "cuDeviceGetAttribute(FABRIC_SUPPORTED)",
    )
    if not fab.value:
        raise RuntimeError(
            f"CUDA device {device_id} reports no fabric-handle support "
            "(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED == 0). "
            "This test requires MNNVL fabric memory."
        )
    ctx = c_void_p()
    check(cuCtxCreate(byref(ctx), 0, dev), "cuCtxCreate")
    return dev, ctx


def build_prop(dev):
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev.value
    prop.allocFlags.gpuDirectRDMACapable = 1
    return prop


def align_up(x, gran):
    return ((x + gran - 1) // gran) * gran


def alloc_and_map(prop, size, dev, label):
    handle = c_ulonglong()
    check(cuMemCreate(byref(handle), size, byref(prop), 0), f"cuMemCreate({label})")
    va = c_ulonglong()
    check(
        cuMemAddressReserve(byref(va), size, 0, 0, 0),
        f"cuMemAddressReserve({label})",
    )
    check(cuMemMap(va, size, 0, handle, 0), f"cuMemMap({label})")
    access = CUmemAccessDesc()
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = dev.value
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    check(
        cuMemSetAccess(va, size, byref(access), 1), f"cuMemSetAccess({label})"
    )
    return handle, va


def import_and_map(fab, size, dev, label):
    handle = c_ulonglong()
    check(
        cuMemImportFromShareableHandle(
            byref(handle),
            c_void_p(ctypes.addressof(fab)),
            CU_MEM_HANDLE_TYPE_FABRIC,
        ),
        f"cuMemImportFromShareableHandle({label})",
    )
    va = c_ulonglong()
    check(
        cuMemAddressReserve(byref(va), size, 0, 0, 0),
        f"cuMemAddressReserve({label})",
    )
    check(cuMemMap(va, size, 0, handle, 0), f"cuMemMap({label})")
    access = CUmemAccessDesc()
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = dev.value
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    check(
        cuMemSetAccess(va, size, byref(access), 1), f"cuMemSetAccess({label})"
    )
    return handle, va


def full_release(handle, va, size, label):
    check(cuMemUnmap(va, size), f"cuMemUnmap({label})")
    check(cuMemAddressFree(va, size), f"cuMemAddressFree({label})")
    check(cuMemRelease(handle), f"cuMemRelease({label})")


def write_u64_pattern(va, pattern_u64, byte_len):
    """Write byte_len bytes of the u64 pattern (little-endian) to va."""
    assert byte_len % 8 == 0
    n = byte_len // 8
    buf = (c_ulonglong * n)(*([pattern_u64] * n))
    check(cuMemcpyHtoD(va, cast(buf, c_void_p), byte_len), "cuMemcpyHtoD")
    check(cuCtxSynchronize(), "cuCtxSynchronize(after write)")


def read_u64_buffer(va, byte_len):
    """Read byte_len bytes from va and return list of u64s."""
    assert byte_len % 8 == 0
    n = byte_len // 8
    buf = (c_ulonglong * n)()
    check(cuMemcpyDtoH(cast(buf, c_void_p), va, byte_len), "cuMemcpyDtoH")
    check(cuCtxSynchronize(), "cuCtxSynchronize(after read)")
    return list(buf)


def find_mismatch(vals, expected):
    """Return (index, value) of the first non-matching u64, or None."""
    for i, v in enumerate(vals):
        if v != expected:
            return (i, v)
    return None


# ============================================================
# Logging
# ============================================================

ROLE = "?"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{ROLE}] {msg}", flush=True)


# ============================================================
# Exporter (rank 0, node A)
# ============================================================


def run_exporter(args):
    global ROLE
    ROLE = "EXPORTER"

    dev, _ = setup_cuda(args.device)
    prop = build_prop(dev)

    gran = c_size_t()
    check(
        cuMemGetAllocationGranularity(
            byref(gran), byref(prop), CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        ),
        "cuMemGetAllocationGranularity",
    )
    size = align_up(args.size_bytes, gran.value)
    probe_bytes = min(size, args.probe_bytes)
    probe_bytes = probe_bytes - (probe_bytes % 8)
    log(
        f"granularity={gran.value:,}  size={size:,}  probe_bytes={probe_bytes:,}  "
        f"poison_count={args.poison_count}"
    )

    # E1-E2: allocate + map + write SENTINEL_1
    handle, va = alloc_and_map(prop, size, dev, "primary")
    SENTINEL_1 = 0xA1A1A1A1A1A1A1A1
    log(
        f"primary: handle={handle.value:#x}  va={va.value:#x}  "
        f"writing SENTINEL_1={SENTINEL_1:#x}"
    )
    write_u64_pattern(va, SENTINEL_1, probe_bytes)

    # E3: export
    fab = CUmemFabricHandle()
    check(
        cuMemExportToShareableHandle(
            c_void_p(ctypes.addressof(fab)),
            handle,
            CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        ),
        "cuMemExportToShareableHandle",
    )
    log("exported fabric handle")

    # accept importer TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.listen(1)
    log(f"listening for importer on 0.0.0.0:{args.port}")
    sock.settimeout(args.tcp_timeout)
    conn, addr = sock.accept()
    conn.settimeout(None)
    log(f"importer connected from {addr}")

    # send meta + handle
    meta = {
        "size": size,
        "probe_bytes": probe_bytes,
        "sentinel_1": SENTINEL_1,
        "sentinel_2": 0xB2B2B2B2B2B2B2B2,
    }
    send_msg(conn, "META", json.dumps(meta).encode())
    send_msg(conn, "HANDLE", bytes(fab.data))
    log(f"sent META + HANDLE ({FABRIC_HANDLE_SIZE} bytes)")

    wait_barrier(conn, "importer_ready")

    # E5: EXPORTER RELEASES
    log("EXPORTER releasing primary (cuMemUnmap + cuMemAddressFree + cuMemRelease)")
    full_release(handle, va, size, "primary")
    log("EXPORTER release complete")

    wait_barrier(conn, "exporter_released")
    wait_barrier(conn, "importer_proof2_done")

    # E8: POISONING
    log(
        f"POISONING: allocating {args.poison_count} new fabric regions of "
        f"size {size:,} to try to force reuse of the freed physical page"
    )
    poison_allocs = []
    for i in range(args.poison_count):
        canary = 0xC0DE000000000000 | (i & 0xFFFFFFFFFFFFFFFF)
        try:
            h, v = alloc_and_map(prop, size, dev, f"poison[{i}]")
        except Exception as e:
            log(
                f"poison alloc {i} failed ({e}); continuing with "
                f"{len(poison_allocs)} allocations"
            )
            break
        write_u64_pattern(v, canary, probe_bytes)
        poison_allocs.append((h, v, canary))
        log(f"poison[{i}]: handle={h.value:#x}  va={v.value:#x}  canary={canary:#x}")
    log(f"POISONING done: {len(poison_allocs)} allocations")

    wait_barrier(conn, "poisoning_done")
    wait_barrier(conn, "importer_proof3_done")

    # E11: release poisons
    for i, (h, v, _) in enumerate(poison_allocs):
        try:
            full_release(h, v, size, f"poison[{i}]")
        except Exception as e:
            log(f"WARNING: failed to release poison[{i}]: {e}")
    log("poison allocations released")

    wait_barrier(conn, "exporter_cleanup_done")

    conn.close()
    sock.close()
    log("EXPORTER exiting rc=0")
    sys.exit(0)


# ============================================================
# Importer (rank 1, node B)
# ============================================================


def run_importer(args):
    global ROLE
    ROLE = "IMPORTER"

    dev, _ = setup_cuda(args.device)

    # connect to exporter
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    log(f"connecting to exporter {args.peer_host}:{args.port}")
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
            f"could not connect to {args.peer_host}:{args.port} within "
            f"{args.tcp_timeout}s: {last_err}"
        )
    log("connected to exporter")

    tag, payload = recv_msg(sock)
    if tag != "META":
        raise RuntimeError(f"expected META, got {tag}")
    meta = json.loads(payload.decode())
    size = int(meta["size"])
    probe_bytes = int(meta["probe_bytes"])
    SENTINEL_1 = int(meta["sentinel_1"])
    SENTINEL_2 = int(meta["sentinel_2"])
    log(
        f"received META: size={size:,}  probe_bytes={probe_bytes:,}  "
        f"S1={SENTINEL_1:#x}  S2={SENTINEL_2:#x}"
    )

    tag, handle_bytes = recv_msg(sock)
    if tag != "HANDLE" or len(handle_bytes) != FABRIC_HANDLE_SIZE:
        raise RuntimeError(
            f"expected HANDLE ({FABRIC_HANDLE_SIZE} bytes), got tag={tag} "
            f"len={len(handle_bytes)}"
        )
    fab = CUmemFabricHandle()
    ctypes.memmove(fab.data, handle_bytes, FABRIC_HANDLE_SIZE)
    log("received HANDLE")

    # I2: import + map
    handle, va = import_and_map(fab, size, dev, "imported")
    log(f"imported: handle={handle.value:#x}  mapped at va={va.value:#x}")

    proofs = {}

    # PROOF #1
    vals = read_u64_buffer(va, probe_bytes)
    m = find_mismatch(vals, SENTINEL_1)
    proofs["proof_1_initial_read"] = m is None
    if m is None:
        log(
            f"PROOF #1 PASS: importer reads SENTINEL_1={SENTINEL_1:#x} "
            f"({len(vals):,} u64s)"
        )
    else:
        log(
            f"PROOF #1 FAIL: expected SENTINEL_1={SENTINEL_1:#x}, "
            f"index[{m[0]}]={m[1]:#x}"
        )

    wait_barrier(sock, "importer_ready")
    wait_barrier(sock, "exporter_released")

    # PROOF #2
    vals = read_u64_buffer(va, probe_bytes)
    m = find_mismatch(vals, SENTINEL_1)
    proofs["proof_2_read_after_exporter_release"] = m is None
    if m is None:
        log(
            f"PROOF #2 PASS: after exporter cuMemRelease, importer STILL "
            f"reads SENTINEL_1 ({len(vals):,} u64s) -> refcount held"
        )
    else:
        is_canary = (m[1] >> 48) == 0xC0DE
        log(
            f"PROOF #2 FAIL: after exporter release, expected SENTINEL_1, "
            f"index[{m[0]}]={m[1]:#x} (canary={is_canary})  "
            "-> refcount BROKEN"
        )

    # PROOF #2b: write-through
    log(f"writing SENTINEL_2={SENTINEL_2:#x} through importer mapping")
    write_u64_pattern(va, SENTINEL_2, probe_bytes)
    vals = read_u64_buffer(va, probe_bytes)
    m = find_mismatch(vals, SENTINEL_2)
    proofs["proof_2b_writeread_after_exporter_release"] = m is None
    if m is None:
        log(
            f"PROOF #2b PASS: importer read-after-write == SENTINEL_2 "
            "-> importer mapping is fully live"
        )
    else:
        log(
            f"PROOF #2b FAIL: expected SENTINEL_2, index[{m[0]}]={m[1]:#x}"
        )

    wait_barrier(sock, "importer_proof2_done")
    wait_barrier(sock, "poisoning_done")

    # PROOF #3
    vals = read_u64_buffer(va, probe_bytes)
    m = find_mismatch(vals, SENTINEL_2)
    proofs["proof_3_no_reuse_after_poisoning"] = m is None
    if m is None:
        log(
            f"PROOF #3 PASS: after exporter's poisoning allocations, "
            f"importer still reads SENTINEL_2 -> physical page was NOT "
            "reused -> cross-node refcount HOLDS"
        )
    else:
        is_canary = (m[1] >> 48) == 0xC0DE
        canary_idx = m[1] & 0xFFFF if is_canary else None
        log(
            f"PROOF #3 FAIL: after poisoning, expected SENTINEL_2, "
            f"index[{m[0]}]={m[1]:#x}"
            + (f" (canary#{canary_idx})" if is_canary else "")
            + "  -> cross-node refcount BROKEN: exporter reused freed page"
        )

    wait_barrier(sock, "importer_proof3_done")

    # I12: importer releases
    log("IMPORTER releasing")
    full_release(handle, va, size, "imported")

    wait_barrier(sock, "exporter_cleanup_done")

    # SUMMARY
    log("")
    log("=" * 64)
    log("SUMMARY")
    log("=" * 64)
    for name, ok in proofs.items():
        log(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_pass = all(proofs.values())
    log("=" * 64)
    if all_pass:
        log("OVERALL: PASS - cross-node VMM fabric refcount HOLDS")
        log(
            "         (exporter's cuMemRelease did NOT free the physical page "
            "while importer held a live import; poisoning did not clobber it)"
        )
    else:
        failed = [k for k, v in proofs.items() if not v]
        log(f"OVERALL: FAIL - failing proofs: {failed}")
    log("=" * 64)

    sock.close()
    sys.exit(0 if all_pass else 1)


# ============================================================
# SIGKILL scenario roles (three-srun orchestration)
#
# Same 4 proofs as the voluntary-release scenario, but the exporter is
# replaced by two independent processes on node A:
#
#   * VICTIM   - allocates fabric mem, exports handle, waits for
#                importer's DIE_NOW, then os.kill(getpid(), SIGKILL).
#   * POISONER - a completely separate process on node A. Waits for
#                importer's start_poison signal (which importer sends
#                only AFTER victim has been killed and reaped), then
#                does cuMemCreate poisoning to try to force reuse of
#                the physical page the victim's death "freed".
#
# The importer (importer_sigkill role) coordinates: connects to victim
# first, does PROOF #1, signals DIE_NOW, waits a grace period for driver
# cleanup, does PROOFs #2 / #2b, then connects to poisoner to drive
# PROOF #3.
#
# The critical semantic being tested: does the driver's INVOLUNTARY
# teardown path (RmClient release on fd close - > memoryfabricDestruct_IMPL)
# still respect the IMEX-mediated cross-node refcount held by the
# importer? Positive result would strengthen the voluntary-release test
# by covering the SIGKILL / crash / OOM-kill code path too.
# ============================================================


def run_victim(args):
    """
    SIGKILL scenario, on node A. Allocate fabric mem, export handle, wait
    for DIE_NOW from importer, then SIGKILL self.
    """
    global ROLE
    ROLE = "VICTIM"

    dev, _ = setup_cuda(args.device)
    prop = build_prop(dev)

    gran = c_size_t()
    check(
        cuMemGetAllocationGranularity(
            byref(gran), byref(prop), CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        ),
        "cuMemGetAllocationGranularity",
    )
    size = align_up(args.size_bytes, gran.value)
    probe_bytes = min(size, args.probe_bytes)
    probe_bytes = probe_bytes - (probe_bytes % 8)
    log(
        f"pid={os.getpid()}  granularity={gran.value:,}  size={size:,}  "
        f"probe_bytes={probe_bytes:,}"
    )

    # Allocate + map + write SENTINEL_1
    handle, va = alloc_and_map(prop, size, dev, "primary")
    SENTINEL_1 = 0xA1A1A1A1A1A1A1A1
    log(
        f"primary: handle={handle.value:#x}  va={va.value:#x}  "
        f"writing SENTINEL_1={SENTINEL_1:#x}"
    )
    write_u64_pattern(va, SENTINEL_1, probe_bytes)

    # Export
    fab = CUmemFabricHandle()
    check(
        cuMemExportToShareableHandle(
            c_void_p(ctypes.addressof(fab)),
            handle,
            CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        ),
        "cuMemExportToShareableHandle",
    )
    log("exported fabric handle")

    # Accept importer
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.listen(1)
    log(f"listening for importer on 0.0.0.0:{args.port}")
    sock.settimeout(args.tcp_timeout)
    conn, addr = sock.accept()
    conn.settimeout(None)
    log(f"importer connected from {addr}")

    meta = {
        "size": size,
        "probe_bytes": probe_bytes,
        "sentinel_1": SENTINEL_1,
        "sentinel_2": 0xB2B2B2B2B2B2B2B2,
    }
    send_msg(conn, "META", json.dumps(meta).encode())
    send_msg(conn, "HANDLE", bytes(fab.data))
    log(f"sent META + HANDLE ({FABRIC_HANDLE_SIZE} bytes)")

    log("waiting for DIE_NOW from importer")
    tag, _ = recv_msg(conn)
    if tag != "DIE_NOW":
        raise RuntimeError(f"expected DIE_NOW, got {tag}")

    log(f"received DIE_NOW; killing self pid={os.getpid()} with SIGKILL")
    try:
        conn.close()
        sock.close()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    # Note: NO cuMemUnmap / cuMemRelease / no Buffer::destroy analog. All
    # cleanup here is involuntary - the kernel will close /dev/nvidia* fds
    # on exit and the driver's RmClient teardown will run
    # memoryfabricDestruct_IMPL / fabricvaspaceUnmapPhysMemdesc_IMPL for
    # our fabric allocation. This is exactly what SIGKILL means in this
    # test. The refcount question: does IMEX propagate our destroy signal
    # to the importer's node such that the importer's still-live ref
    # keeps the physical page alive?
    os.kill(os.getpid(), signal.SIGKILL)
    # unreachable


def run_poisoner(args):
    """
    SIGKILL scenario, on node A (SAME node as victim, different process).
    Waits for importer to signal (which importer only does after victim
    is dead), then does cuMemCreate poisoning.
    """
    global ROLE
    ROLE = "POISONER"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.listen(1)
    log(
        f"pid={os.getpid()}  listening for importer on 0.0.0.0:{args.port}"
    )
    sock.settimeout(args.tcp_timeout)
    conn, addr = sock.accept()
    conn.settimeout(None)
    log(f"importer connected from {addr}")

    tag, payload = recv_msg(conn)
    if tag != "META":
        raise RuntimeError(f"expected META, got {tag}")
    meta = json.loads(payload.decode())
    size = int(meta["size"])
    probe_bytes = int(meta["probe_bytes"])
    log(f"received META: size={size:,}  probe_bytes={probe_bytes:,}")

    wait_barrier(conn, "start_poison")

    # NOW init CUDA in this fresh process. Doing it here (rather than at
    # the top of the role) means we start with a clean CUDA context AFTER
    # victim has died and been reaped, which is closer to what a real
    # "surviving neighbor" process on node A would look like.
    dev, _ = setup_cuda(args.device)
    prop = build_prop(dev)

    log(
        f"POISONING: allocating {args.poison_count} new fabric regions of "
        f"size {size:,} to force reuse of the victim's freed physical page "
        "if the cross-node refcount is broken under SIGKILL"
    )
    poison_allocs = []
    for i in range(args.poison_count):
        canary = 0xC0DE000000000000 | (i & 0xFFFFFFFFFFFFFFFF)
        try:
            h, v = alloc_and_map(prop, size, dev, f"poison[{i}]")
        except Exception as e:
            log(
                f"poison alloc {i} failed ({e}); continuing with "
                f"{len(poison_allocs)} allocations"
            )
            break
        write_u64_pattern(v, canary, probe_bytes)
        poison_allocs.append((h, v, canary))
        log(
            f"poison[{i}]: handle={h.value:#x}  va={v.value:#x}  "
            f"canary={canary:#x}"
        )
    log(f"POISONING done: {len(poison_allocs)} allocations")

    wait_barrier(conn, "poison_done")
    wait_barrier(conn, "cleanup")

    for i, (h, v, _) in enumerate(poison_allocs):
        try:
            full_release(h, v, size, f"poison[{i}]")
        except Exception as e:
            log(f"WARNING: failed to release poison[{i}]: {e}")
    log("poison allocations released")

    conn.close()
    sock.close()
    log("POISONER exiting rc=0")
    sys.exit(0)


def run_importer_sigkill(args):
    """
    SIGKILL scenario, on node B. Coordinates victim and poisoner via TCP.
    """
    global ROLE
    ROLE = "IMPORTER"

    dev, _ = setup_cuda(args.device)

    # ---------------- Connect to VICTIM ----------------
    vsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    log(f"connecting to victim {args.peer_host}:{args.port}")
    connected = False
    deadline = time.time() + args.tcp_timeout
    last_err = None
    while time.time() < deadline:
        try:
            vsock.connect((args.peer_host, args.port))
            connected = True
            break
        except (ConnectionRefusedError, OSError) as e:
            last_err = e
            time.sleep(1)
    if not connected:
        raise RuntimeError(
            f"could not connect to victim {args.peer_host}:{args.port}: "
            f"{last_err}"
        )
    log("connected to victim")

    tag, payload = recv_msg(vsock)
    if tag != "META":
        raise RuntimeError(f"expected META from victim, got {tag}")
    meta = json.loads(payload.decode())
    size = int(meta["size"])
    probe_bytes = int(meta["probe_bytes"])
    SENTINEL_1 = int(meta["sentinel_1"])
    SENTINEL_2 = int(meta["sentinel_2"])
    log(
        f"received META: size={size:,}  probe_bytes={probe_bytes:,}  "
        f"S1={SENTINEL_1:#x}  S2={SENTINEL_2:#x}"
    )

    tag, handle_bytes = recv_msg(vsock)
    if tag != "HANDLE" or len(handle_bytes) != FABRIC_HANDLE_SIZE:
        raise RuntimeError(
            f"expected HANDLE ({FABRIC_HANDLE_SIZE} bytes), got tag={tag} "
            f"len={len(handle_bytes)}"
        )
    fab = CUmemFabricHandle()
    ctypes.memmove(fab.data, handle_bytes, FABRIC_HANDLE_SIZE)
    log("received HANDLE")

    # Import + map
    handle, va = import_and_map(fab, size, dev, "imported")
    log(f"imported: handle={handle.value:#x}  mapped at va={va.value:#x}")

    proofs = {}

    # PROOF #1
    vals = read_u64_buffer(va, probe_bytes)
    m = find_mismatch(vals, SENTINEL_1)
    proofs["proof_1_initial_read"] = m is None
    if m is None:
        log(
            f"PROOF #1 PASS: importer reads SENTINEL_1={SENTINEL_1:#x} "
            f"({len(vals):,} u64s)"
        )
    else:
        log(
            f"PROOF #1 FAIL: expected SENTINEL_1, index[{m[0]}]={m[1]:#x}"
        )

    # ---------------- Kill victim ----------------
    log("sending DIE_NOW to victim")
    send_msg(vsock, "DIE_NOW")
    try:
        vsock.close()
    except Exception:
        pass

    log(
        f"waiting {args.victim_death_grace}s for victim to be killed and "
        "reaped (driver runs memoryfabricDestruct_IMPL / IMEX destroy "
        "propagation during this window)"
    )
    time.sleep(args.victim_death_grace)

    # PROOF #2 - after victim SIGKILL
    vals = read_u64_buffer(va, probe_bytes)
    m = find_mismatch(vals, SENTINEL_1)
    proofs["proof_2_read_after_victim_sigkill"] = m is None
    if m is None:
        log(
            "PROOF #2 PASS: after victim SIGKILL, importer STILL reads "
            f"SENTINEL_1 ({len(vals):,} u64s) -> refcount held through "
            "involuntary driver teardown"
        )
    else:
        is_canary = (m[1] >> 48) == 0xC0DE
        log(
            f"PROOF #2 FAIL: after victim SIGKILL, expected SENTINEL_1, "
            f"index[{m[0]}]={m[1]:#x} (canary={is_canary})  "
            "-> refcount BROKEN under SIGKILL"
        )

    # PROOF #2b - write-through
    log(f"writing SENTINEL_2={SENTINEL_2:#x} through importer mapping")
    write_u64_pattern(va, SENTINEL_2, probe_bytes)
    vals = read_u64_buffer(va, probe_bytes)
    m = find_mismatch(vals, SENTINEL_2)
    proofs["proof_2b_writeread_after_victim_sigkill"] = m is None
    if m is None:
        log(
            "PROOF #2b PASS: importer read-after-write == SENTINEL_2 "
            "-> mapping fully live after victim SIGKILL"
        )
    else:
        log(
            f"PROOF #2b FAIL: expected SENTINEL_2, "
            f"index[{m[0]}]={m[1]:#x}"
        )

    # ---------------- Connect to POISONER ----------------
    psock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    log(f"connecting to poisoner {args.peer_host}:{args.poisoner_port}")
    connected = False
    deadline = time.time() + args.tcp_timeout
    last_err = None
    while time.time() < deadline:
        try:
            psock.connect((args.peer_host, args.poisoner_port))
            connected = True
            break
        except (ConnectionRefusedError, OSError) as e:
            last_err = e
            time.sleep(1)
    if not connected:
        raise RuntimeError(
            f"could not connect to poisoner "
            f"{args.peer_host}:{args.poisoner_port}: {last_err}"
        )
    log("connected to poisoner")

    # Poisoner needs size + probe_bytes to allocate matching regions
    send_msg(
        psock,
        "META",
        json.dumps({"size": size, "probe_bytes": probe_bytes}).encode(),
    )

    wait_barrier(psock, "start_poison")
    wait_barrier(psock, "poison_done")

    # PROOF #3
    vals = read_u64_buffer(va, probe_bytes)
    m = find_mismatch(vals, SENTINEL_2)
    proofs["proof_3_no_reuse_after_poisoning"] = m is None
    if m is None:
        log(
            "PROOF #3 PASS: after poisoner allocations, importer still "
            "reads SENTINEL_2 -> physical page NOT reused -> cross-node "
            "refcount HOLDS through victim's SIGKILL involuntary teardown"
        )
    else:
        is_canary = (m[1] >> 48) == 0xC0DE
        canary_idx = m[1] & 0xFFFF if is_canary else None
        log(
            f"PROOF #3 FAIL: expected SENTINEL_2, "
            f"index[{m[0]}]={m[1]:#x}"
            + (f" (canary#{canary_idx})" if is_canary else "")
            + "  -> refcount BROKEN under SIGKILL: victim's freed page "
            "was reused by poisoner"
        )

    wait_barrier(psock, "cleanup")

    log("IMPORTER releasing")
    full_release(handle, va, size, "imported")

    # SUMMARY
    log("")
    log("=" * 64)
    log("SUMMARY (SIGKILL scenario)")
    log("=" * 64)
    for name, ok in proofs.items():
        log(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_pass = all(proofs.values())
    log("=" * 64)
    if all_pass:
        log(
            "OVERALL: PASS - cross-node VMM fabric refcount HOLDS under "
            "exporter SIGKILL"
        )
        log(
            "         (driver's involuntary teardown on victim's SIGKILL "
            "did NOT free the physical page while importer held a live "
            "cross-node import; poisoner's new allocations did not "
            "clobber the importer's mapping)"
        )
    else:
        failed = [k for k, v in proofs.items() if not v]
        log(f"OVERALL: FAIL - failing proofs: {failed}")
    log("=" * 64)

    psock.close()
    sys.exit(0 if all_pass else 1)


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser(
        description="Cross-node VMM fabric refcount test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--role",
        choices=["exporter", "importer", "victim", "poisoner", "importer_sigkill"],
        required=True,
        help=(
            "release scenario: exporter (node A) + importer (node B); "
            "sigkill scenario: victim + poisoner (both on node A) + "
            "importer_sigkill (node B)"
        ),
    )
    ap.add_argument(
        "--peer-host",
        help="hostname/IP of exporter/victim (importer* roles only)",
    )
    ap.add_argument("--port", type=int, default=27182)
    ap.add_argument(
        "--poisoner-port",
        type=int,
        default=27183,
        help="port for importer_sigkill to reach poisoner (default 27183)",
    )
    ap.add_argument(
        "--victim-death-grace",
        type=int,
        default=3,
        help="seconds importer_sigkill waits after DIE_NOW before doing "
        "PROOF #2, to let the OS reap the victim and the driver run "
        "memoryfabricDestruct_IMPL (default 3)",
    )
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    ap.add_argument(
        "--size-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="requested allocation size in bytes; rounded up to fabric "
        "granularity (default: 64 MiB)",
    )
    ap.add_argument(
        "--probe-bytes",
        type=int,
        default=8 * 1024 * 1024,
        help="how many bytes of the allocation to fill with the pattern "
        "and read back on each PROOF (default: 8 MiB). Capped at "
        "--size-bytes.",
    )
    ap.add_argument(
        "--poison-count",
        type=int,
        default=16,
        help="number of new fabric allocations the exporter makes after "
        "releasing, to try to force reuse of the freed physical page "
        "(default: 16). Each is --size-bytes big.",
    )
    ap.add_argument(
        "--tcp-timeout",
        type=int,
        default=120,
        help="seconds to wait for the peer to connect / accept (default: 120)",
    )
    args = ap.parse_args()

    if args.role == "exporter":
        run_exporter(args)
    elif args.role == "importer":
        if not args.peer_host:
            ap.error("--peer-host is required for --role importer")
        run_importer(args)
    elif args.role == "victim":
        run_victim(args)
    elif args.role == "poisoner":
        run_poisoner(args)
    elif args.role == "importer_sigkill":
        if not args.peer_host:
            ap.error("--peer-host is required for --role importer_sigkill")
        run_importer_sigkill(args)
    else:
        ap.error(f"unknown role: {args.role}")


if __name__ == "__main__":
    main()
