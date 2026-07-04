# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Artifact capture for NVLink fault-injection experiments.

Per-rank background sampler + end-of-run diagnostics + summary.json. Gated
off by default; enable per run with NIXL_FAULT_CAPTURE=1. No sudo required.

Layout under ${FAULT_EVIDENCE_DIR:-cwd}:

    artifacts_rank{R}/
      nvlink_counters.csv   # nvidia-smi nvlink --errorcounters, ~500 ms cadence
      dmesg.log             # dmesg -T --since '5 minutes ago' at close
                            # NOTE: on Lyris gb200-backfill dmesg_restrict=1
                            # and the user is not in adm/systemd-journal, so
                            # dmesg is expected to be permission-denied. We
                            # write the error text as evidence anyway.
      imex.log              # /var/log/nvidia-imex-verbose.log fragment
                            # bounded by the sampler start/stop timestamps.
                            # World-readable on gb200-backfill and captures
                            # MNNVL disconnect / import-error events that
                            # substitute for kernel-XID visibility.
      summary.json          # {rank, role, inject_event_ts, xid_seen,
                            #  imex_error_count, ...}
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# NVLink counter sampling cadence (seconds). Cheap enough at 500 ms; peers
# only need better-than-timeout visibility.
_SAMPLE_INTERVAL_SEC = 0.5
# Window for the dmesg snapshot (best-effort; usually denied on Lyris).
_DMESG_WINDOW = "5 minutes ago"
# World-readable IMEX log on gb200-backfill. Any [ERROR]/[WARNING] line here
# in the run window is our XID substitute for MNNVL fabric events.
# The container has its own /var/log; runner scripts bind-mount the host
# /var/log at /host/var/log:ro so we can still see the real imex log. We
# search both, preferring the container-native path if it exists.
_IMEX_LOG_PATHS = (
    "/var/log/nvidia-imex-verbose.log",
    "/host/var/log/nvidia-imex-verbose.log",
)
# Field names emitted per (gpu, link) row in nvlink_counters.csv. Names match
# the actual `nvidia-smi nvlink --errorcounters` output on GB200 (Blackwell).
_NVLINK_FIELDS: Tuple[str, ...] = (
    "malformed_pkt_err",
    "buffer_overrun_err",
    "rx_err",
    "rx_remote_err",
    "rx_general_err",
    "local_link_integrity_err",
    "tx_discards",
    "link_recovery_ok",
    "link_recovery_fail",
    "total_link_recovery",
    "effective_err",
    "symbol_err",
)
# Regexes anchored to the exact strings emitted by nvidia-smi.
_NVLINK_FIELD_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(rf"^Link\s+(\d+):\s+{pat}\s*:\s*(-?\d+)\s*$", re.IGNORECASE))
    for name, pat in (
        ("malformed_pkt_err",         r"Malformed\s+packet\s+Errors"),
        ("buffer_overrun_err",        r"Buffer\s+overrun\s+Errors"),
        ("rx_err",                    r"Rx\s+Errors"),
        ("rx_remote_err",             r"Rx\s+remote\s+Errors"),
        ("rx_general_err",            r"Rx\s+General\s+Errors"),
        ("local_link_integrity_err",  r"Local\s+link\s+integrity\s+Errors"),
        ("tx_discards",               r"Tx\s+discards"),
        ("link_recovery_ok",          r"Link\s+recovery\s+successful\s+events"),
        ("link_recovery_fail",        r"Link\s+recovery\s+failed\s+events"),
        ("total_link_recovery",       r"Total\s+link\s+recovery\s+events"),
        ("effective_err",             r"Effective\s+Errors"),
        ("symbol_err",                r"Symbol\s+Errors"),
    )
)
# Two header shapes seen in the wild:
#   `nvlink --errorcounters`: "GPU 0: NVIDIA GB200 (UUID: GPU-...)"  → id "0"
#   `-q -d PAGE_RETIREMENT`  : "GPU 00000008:06:00.0"                → id "00000008:06:00.0"
# Regex-only can't disambiguate cleanly (BDF contains colons); we split in
# Python instead.
_GPU_HEADER_PREFIX = "GPU "


def _parse_gpu_header(line: str) -> Optional[str]:
    if not line.startswith(_GPU_HEADER_PREFIX):
        return None
    rest = line[len(_GPU_HEADER_PREFIX):].strip()
    # ": " (colon-space) is the reliable separator in the nvlink-style header
    # because BDF PCI addresses contain colons but no colon-space pairs.
    if ": " in rest:
        return rest.split(": ", 1)[0].strip()
    return rest


def enabled() -> bool:
    """True when NIXL_FAULT_CAPTURE=1 in env."""
    return os.environ.get("NIXL_FAULT_CAPTURE", "0") == "1"


@dataclass
class Summary:
    rank: int
    role: str = "peer"                          # 'victim' on the injecting rank
    inject_mode: str = "sigkill"                # copied from --fault-inject-mode
    inject_event_ts: Optional[int] = None       # time.time_ns() at injection
    observed_first_error_ts: Optional[int] = None
    observed_error_type: Optional[str] = None
    xid_seen: bool = False                      # kernel XID seen in dmesg (usually N/A on Lyris)
    xid_count: int = 0
    dmesg_readable: Optional[bool] = None       # False on Lyris user shells (env constraint)
    imex_log_captured: bool = False             # True if we managed to copy the imex fragment
    imex_error_count: int = 0                   # #[ERROR] lines in window (fabric-event proxy for XID)
    imex_warning_count: int = 0                 # #[WARNING] lines in window
    recovered: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ArtifactCapture:
    """Per-rank artifact capture. Start in worker() startup, stop before exit.

    Safe to construct even when capture is not needed - all methods are no-ops
    on a stopped/never-started instance.
    """

    def __init__(self, rank: int, evidence_dir: str, local_rank: Optional[int] = None):
        self.rank = rank
        base = evidence_dir or os.getcwd()
        self.dir = os.path.join(base, f"artifacts_rank{rank}")
        os.makedirs(self.dir, exist_ok=True)
        self.counters_csv = os.path.join(self.dir, "nvlink_counters.csv")
        self.dmesg_log = os.path.join(self.dir, "dmesg.log")
        self.imex_log = os.path.join(self.dir, "imex.log")
        self.summary_path = os.path.join(self.dir, "summary.json")
        self.summary = Summary(rank=rank)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start_ns: Optional[int] = None
        self._t_stop_ns: Optional[int] = None
        # nvidia-smi -i wants the driver's original device index. When
        # CUDA_VISIBLE_DEVICES has been narrowed to a single GPU (elastic.py
        # sets it to str(local_rank % 8) BEFORE we construct), the driver
        # renumbers that GPU to 0 for us, so we must ALSO pass 0 to
        # nvidia-smi. Otherwise we'd be sampling a peer's counters or hit
        # "Invalid device id".
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if cvd:
            self._gpu_idx: Optional[str] = "0"
            self._gpu_uuid: Optional[str] = None
        elif local_rank is not None:
            self._gpu_idx = str(local_rank % 8)
            self._gpu_uuid = None
        else:
            self._gpu_idx = None
            self._gpu_uuid = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            with open(self.counters_csv, "w") as f:
                f.write("wall_ns,gpu," + ",".join(("link",) + _NVLINK_FIELDS) + "\n")
        except OSError:
            return
        self._t_start_ns = time.time_ns()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"nvlink-cap-r{self.rank}"
        )
        self._thread.start()
        print(
            f"[rank {self.rank}] ARTIFACT CAPTURE started dir={self.dir} gpu={self._gpu_idx}",
            flush=True,
        )

    def stop(self, extra_settle_sec: float = 5.0) -> None:
        """Sample `extra_settle_sec` more, then stop, snapshot diagnostics, write summary."""
        if extra_settle_sec > 0:
            time.sleep(extra_settle_sec)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._t_stop_ns = time.time_ns()
        self._snapshot_dmesg()
        self._snapshot_imex_log()
        self._write_summary()
        print(
            f"[rank {self.rank}] ARTIFACT CAPTURE done xid_count={self.summary.xid_count} "
            f"imex_err={self.summary.imex_error_count} "
            f"recovered={self.summary.recovered} dir={self.dir}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # injector-side markers
    # ------------------------------------------------------------------ #

    def mark_inject(self, mode: str, is_victim: bool) -> None:
        self.summary.inject_mode = mode
        self.summary.role = "victim" if is_victim else "peer"
        self.summary.inject_event_ts = time.time_ns()
        try:
            with open(self.counters_csv, "a") as f:
                f.write(f"# INJECT ts={self.summary.inject_event_ts} mode={mode} victim={is_victim}\n")
        except OSError:
            pass

    def mark_first_error(self, error_type: str) -> None:
        if self.summary.observed_first_error_ts is None:
            self.summary.observed_first_error_ts = time.time_ns()
            self.summary.observed_error_type = error_type

    def mark_recovered(self, recovered: bool) -> None:
        self.summary.recovered = recovered

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        nvsmi = shutil.which("nvidia-smi")
        if not nvsmi:
            self._append_comment("nvidia-smi not found; sampler exiting")
            return
        cmd = [nvsmi, "nvlink", "--errorcounters"]
        if self._gpu_idx is not None:
            cmd += ["-i", self._gpu_idx]
        while not self._stop.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
                text = out.stdout or ""
            except Exception as e:  # noqa: BLE001
                self._append_comment(f"nvidia-smi failed: {e!r}")
                self._stop.wait(_SAMPLE_INTERVAL_SEC)
                continue
            rows = _parse_nvsmi_errorcounters(text, ts_ns=time.time_ns(), fallback_gpu=self._gpu_idx or "?")
            if rows:
                try:
                    with open(self.counters_csv, "a") as f:
                        for r in rows:
                            f.write(",".join(r) + "\n")
                except OSError:
                    pass
            self._stop.wait(_SAMPLE_INTERVAL_SEC)

    def _append_comment(self, msg: str) -> None:
        try:
            with open(self.counters_csv, "a") as f:
                f.write(f"# {msg}\n")
        except OSError:
            pass

    def _snapshot_dmesg(self) -> None:
        """Best-effort dmesg snapshot. On Lyris gb200-backfill this is expected
        to fail (dmesg_restrict=1, user not in adm/systemd-journal). We still
        write dmesg.log with the exact stderr text as evidence.
        """
        body = ""
        for cmd in (
            ["dmesg", "-T", f"--since={_DMESG_WINDOW}"],
            ["dmesg", "-T"],
            ["dmesg"],
        ):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
            except Exception as e:  # noqa: BLE001
                body = f"# dmesg({cmd!r}) failed: {e!r}\n"
                continue
            if out.returncode == 0 and (out.stdout or ""):
                body = out.stdout
                self.summary.dmesg_readable = True
                break
            body = out.stderr or body
        if self.summary.dmesg_readable is None:
            self.summary.dmesg_readable = False
        xid_count = sum(1 for line in body.splitlines() if "NVRM: Xid" in line)
        self.summary.xid_seen = xid_count > 0
        self.summary.xid_count = xid_count
        try:
            with open(self.dmesg_log, "w") as f:
                f.write(body)
        except OSError:
            pass

    def _snapshot_imex_log(self) -> None:
        """Copy the IMEX verbose log fragment bounded by [start_ns, stop_ns].

        IMEX log format:
            [Jul 01 2026 03:33:05] [ERROR] [tid 26306] Node disconnect event ...

        We do a simple time-window filter (parse the leading [MMM DD YYYY
        HH:MM:SS] timestamp; fall back to including the line if we can't
        parse). Counts [ERROR] and [WARNING] occurrences for summary.json.
        """
        chosen: Optional[str] = None
        for p in _IMEX_LOG_PATHS:
            if os.path.isfile(p):
                chosen = p
                break
        if chosen is None:
            self._imex_note(f"# imex log not found in {_IMEX_LOG_PATHS}\n")
            return
        try:
            with open(chosen, "r", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            self._imex_note(f"# imex log ({chosen}) open failed: {e!r}\n")
            return

        start_ns = self._t_start_ns or 0
        stop_ns = self._t_stop_ns or time.time_ns()
        # Include a small padding on either side to catch fabric events that
        # happen just before/after the actual window.
        pad_ns = int(2 * 1e9)
        lo = start_ns - pad_ns
        hi = stop_ns + pad_ns

        kept: List[str] = []
        err_count = 0
        warn_count = 0
        for line in lines:
            ts_ns = _parse_imex_ts_ns(line)
            if ts_ns is not None and not (lo <= ts_ns <= hi):
                continue
            kept.append(line)
            if "[ERROR]" in line:
                err_count += 1
                if self.summary.observed_first_error_ts is None:
                    self.summary.observed_first_error_ts = ts_ns or time.time_ns()
                    self.summary.observed_error_type = "imex_error"
            elif "[WARNING]" in line:
                warn_count += 1

        self.summary.imex_log_captured = True
        self.summary.imex_error_count = err_count
        self.summary.imex_warning_count = warn_count
        try:
            with open(self.imex_log, "w") as f:
                f.write(f"# IMEX log source: {chosen}\n")
                f.write(f"# window [{lo}..{hi}] ns (start={start_ns} stop={stop_ns})\n")
                f.write(f"# errors={err_count} warnings={warn_count} lines_kept={len(kept)}\n")
                f.writelines(kept)
        except OSError:
            pass

    def _imex_note(self, msg: str) -> None:
        try:
            with open(self.imex_log, "w") as f:
                f.write(msg)
        except OSError:
            pass

    def _write_summary(self) -> None:
        try:
            with open(self.summary_path, "w") as f:
                json.dump(asdict(self.summary), f, indent=2, sort_keys=True)
                f.write("\n")
        except OSError:
            pass


# ---------------------------------------------------------------------- #
# parsers
# ---------------------------------------------------------------------- #

# IMEX log timestamp: "[Jul 01 2026 03:33:05]"
_IMEX_TS_RE = re.compile(r"^\[(\w{3})\s+(\d{1,2})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\]")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _parse_imex_ts_ns(line: str) -> Optional[int]:
    m = _IMEX_TS_RE.match(line)
    if not m:
        return None
    mon = _MONTHS.get(m.group(1))
    if mon is None:
        return None
    try:
        # Assume local time; struct_time expects tm_isdst=-1 for auto DST.
        st = time.struct_time((
            int(m.group(3)),  # year
            mon,
            int(m.group(2)),
            int(m.group(4)), int(m.group(5)), int(m.group(6)),
            0, 0, -1,
        ))
        return int(time.mktime(st) * 1_000_000_000)
    except (ValueError, OverflowError):
        return None


def _parse_nvsmi_errorcounters(text: str, ts_ns: int, fallback_gpu: str) -> List[List[str]]:
    """Parse `nvidia-smi nvlink --errorcounters` (GB200 format).

    Example (per GPU header + per link block, tab-indented):
        GPU 00000008:06:00.0
             Link 0: Tx packets: 12345
             Link 0: Rx Errors: 0
             Link 0: Rx remote Errors: 0
             Link 0: Local link integrity Errors: 0
             Link 0: Total link recovery events: 0
             Link 0: Effective Errors: 0
             Link 0: Symbol Errors: 0
             ... other fields, including FEC Errors - N which we skip ...

    Emits one CSV row per (gpu, link) with the fields in `_NVLINK_FIELDS`,
    filling '' for any unparsed slot.
    """
    rows: List[List[str]] = []
    cur_gpu = fallback_gpu
    per_link: Dict[str, Dict[str, str]] = {}

    def _flush() -> None:
        for link, fields in sorted(
            per_link.items(),
            key=lambda kv: int(kv[0]) if kv[0].isdigit() else 9999,
        ):
            rows.append(
                [str(ts_ns), cur_gpu, link]
                + [fields.get(f, "") for f in _NVLINK_FIELDS]
            )

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        gpu_id = _parse_gpu_header(line)
        if gpu_id is not None:
            _flush()
            per_link = {}
            cur_gpu = gpu_id
            continue
        for fname, patt in _NVLINK_FIELD_PATTERNS:
            fm = patt.match(line)
            if fm:
                slot = per_link.setdefault(fm.group(1), {})
                slot[fname] = fm.group(2)
                break
    _flush()
    return rows
