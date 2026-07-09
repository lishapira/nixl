#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MNNVL fabric clique probe.

Reads `nvidia-smi -q -d FABRIC` and prints one JSON object per GPU with
{gpu, uuid, cluster_uuid, clique_id, state}. Two nodes are on the same
MNNVL clique iff their GPUs report the same (cluster_uuid, clique_id).

Used by run_phase5_2node.sh both pre- and post-injection to verify:
  pre : both nodes' GPUs share ClusterUUID + CliqueId (are actually
        MNNVL-coupled, not IB fallback);
  post: same (fabric was not corrupted by the unmap injection).

Runs `nvidia-smi` as a subprocess and returns 0 on success even if
FABRIC info is missing on a given GPU (records "unknown" for the
missing fields so the caller can decide what to do).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List


def _run_nvsmi() -> str:
    # On GB200 with driver 580.173, `nvidia-smi -q -d FABRIC` is NOT
    # recognized ("Failed to parse --display/-d flags"). The Fabric
    # block is present in the plain `nvidia-smi -q` dump though, and
    # we filter to the fields we need on the parse side. Plain -q is
    # ~300 KB per node so we accept the extra bytes for portability
    # across driver versions.
    try:
        return subprocess.check_output(
            ["nvidia-smi", "-q"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as ex:
        out = getattr(ex, "output", "") or ""
        print(f"# nvsmi failed: {ex!r}\n{out}", file=sys.stderr)
        return ""


# `nvidia-smi -q` GPU header comes in two forms across driver versions:
#   "GPU 00000008:06:00.0"   (BDF form, GB200)
#   "GPU 0"                  (index form, some older drivers)
# We just capture the trailing token after "GPU " and hand it back.
_GPU_HDR = re.compile(r"^GPU\s+(\S+)")
_UUID = re.compile(r"^\s*GPU UUID\s*:\s*(\S+)")
_CLUSTER_UUID = re.compile(r"^\s*ClusterUUID\s*:\s*(\S+)")
_CLIQUE_ID = re.compile(r"^\s*CliqueId\s*:\s*(\S+)")
# The Fabric block on 580.x drivers uses "State" AND "Status"; both are
# useful for diagnosing "GPU present but fabric detached". We capture the
# FIRST such match under each GPU which is the fabric state.
_STATE = re.compile(r"^\s*(?:State|Status)\s*:\s*(\S+)")


def _parse(nvsmi_output: str, hostname: str) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for line in nvsmi_output.splitlines():
        m = _GPU_HDR.match(line)
        if m:
            if current is not None:
                gpus.append(current)
            current = {
                "host": hostname,
                "gpu": m.group(1),
                "uuid": "unknown",
                "cluster_uuid": "unknown",
                "clique_id": "unknown",
                "fabric_state": "unknown",
            }
            continue
        if current is None:
            continue
        for key, rgx in (
            ("uuid", _UUID),
            ("cluster_uuid", _CLUSTER_UUID),
            ("clique_id", _CLIQUE_ID),
            ("fabric_state", _STATE),
        ):
            m = rgx.match(line)
            if m:
                current[key] = m.group(1)
                break
    if current is not None:
        gpus.append(current)
    return gpus


def main() -> int:
    hostname = os.uname().nodename.split(".")[0]
    output = _run_nvsmi()
    gpus = _parse(output, hostname) if output else []
    print(json.dumps({"host": hostname, "gpus": gpus}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
