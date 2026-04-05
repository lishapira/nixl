# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import signal
import sys
import time
from datetime import timedelta
from functools import partial

import numpy as np
import torch

# Add tests directory to path to import shared utils package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nixl_ep  # noqa: E402
from utils import rank_server, store_group  # noqa: E402

TCP_STORE_PORT = 9999
RANK_SERVER_PORT = 10000


class CudaTimer:
    """CUDA event timer. Same timing methodology as bench() in tests/utils/helpers.py."""

    def __enter__(self):
        torch.cuda.synchronize()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, *args):
        self._end.record()
        torch.cuda.synchronize()
        self.elapsed_s = self._start.elapsed_time(self._end) / 1e3


def handle_sigterm(signum, frame, rank_client):
    print(
        f"SIGTERM ({signum}) received for process {os.getpid()}! "
        f"releasing rank and exiting...",
        flush=True,
    )
    rank_client.release_rank()
    sys.exit(1)


_barrier_counter = 0


def tcp_store_barrier(tcp_store, rank, num_ranks, timeout=60):
    """Synchronize all ranks using TCPStore set/wait."""
    global _barrier_counter
    name = f"ctrl_{_barrier_counter}"
    _barrier_counter += 1
    key = f"{name}/{rank}"
    tcp_store.set(key, "1")
    keys = [f"{name}/{r}" for r in range(num_ranks)]
    tcp_store.wait(keys, timedelta(seconds=timeout))


def _stats(times):
    """Compute (avg, min, max) from a list of seconds."""
    if not times:
        return 0.0, 0.0, 0.0
    a = np.array(times)
    return float(np.average(a)), float(np.min(a)), float(np.max(a))


def create_buffer(rank, disable_ll_nvlink, tcp_store, num_ranks,
                  num_experts_per_rank, num_rdma_bytes):
    """Create and initialize a nixl_ep.Buffer with memory buffers allocated."""
    buf = nixl_ep.Buffer(
        rank=rank,
        disable_ll_nvlink=disable_ll_nvlink,
        explicitly_destroy=True,
        tcp_store_group=tcp_store,
    )
    buf.update_memory_buffers(
        num_ranks=num_ranks,
        num_experts_per_rank=num_experts_per_rank,
        num_rdma_bytes=num_rdma_bytes,
    )
    return buf


def bench_init(rank, disable_ll_nvlink, tcp_store, num_ranks,
               num_experts_per_rank, num_rdma_bytes):
    """Time buffer creation. Returns (buffer, elapsed_s)."""
    with CudaTimer() as t:
        buffer = create_buffer(rank, disable_ll_nvlink, tcp_store, num_ranks,
                               num_experts_per_rank, num_rdma_bytes)
    return buffer, t.elapsed_s


def bench_connect(buffer, other_ranks):
    """Time connect_ranks. Returns elapsed_s (0.0 if no other_ranks)."""
    if not other_ranks:
        return 0.0
    with CudaTimer() as t:
        buffer.connect_ranks(other_ranks)
    return t.elapsed_s


def bench_disconnect(buffer, other_ranks):
    """Time disconnect_ranks. Returns elapsed_s (0.0 if no other_ranks)."""
    if not other_ranks:
        return 0.0
    with CudaTimer() as t:
        buffer.disconnect_ranks(other_ranks)
    return t.elapsed_s


def bench_destroy(buffer):
    """Time buffer.destroy(). Returns elapsed_s."""
    with CudaTimer() as t:
        buffer.destroy()
    return t.elapsed_s


def run_cycle(rank, num_ranks, other_ranks, tcp_store, disable_ll_nvlink,
              num_experts_per_rank, num_rdma_bytes, warmup, rounds):
    """Run full cycle: init -> connect -> disconnect -> reconnect -> destroy."""
    init_times, connect_times, disconnect_times = [], [], []
    reconnect_times, destroy_times = [], []

    for i in range(warmup + rounds):
        is_measure = i >= warmup

        tcp_store_barrier(tcp_store, rank, num_ranks)
        buffer, elapsed = bench_init(rank, disable_ll_nvlink, tcp_store,
                                     num_ranks, num_experts_per_rank,
                                     num_rdma_bytes)
        if is_measure:
            init_times.append(elapsed)

        tcp_store_barrier(tcp_store, rank, num_ranks)
        elapsed = bench_connect(buffer, other_ranks)
        if is_measure:
            connect_times.append(elapsed)

        tcp_store_barrier(tcp_store, rank, num_ranks)
        elapsed = bench_disconnect(buffer, other_ranks)
        if is_measure:
            disconnect_times.append(elapsed)
        time.sleep(0.5)

        tcp_store_barrier(tcp_store, rank, num_ranks)
        elapsed = bench_connect(buffer, other_ranks)
        if is_measure:
            reconnect_times.append(elapsed)

        tcp_store_barrier(tcp_store, rank, num_ranks)
        elapsed = bench_destroy(buffer)
        if is_measure:
            destroy_times.append(elapsed)

    return {
        "init": _stats(init_times),
        "connect": _stats(connect_times),
        "disconnect": _stats(disconnect_times),
        "reconnect": _stats(reconnect_times),
        "destroy": _stats(destroy_times),
    }


def run_single_op(mode, rank, num_ranks, other_ranks, tcp_store,
                  disable_ll_nvlink, num_experts_per_rank, num_rdma_bytes,
                  warmup, rounds):
    """Run a single control plane operation benchmark."""
    latencies = []

    if mode == "init":
        for i in range(warmup + rounds):
            tcp_store_barrier(tcp_store, rank, num_ranks)
            buffer, elapsed = bench_init(rank, disable_ll_nvlink, tcp_store,
                                         num_ranks, num_experts_per_rank,
                                         num_rdma_bytes)
            if i >= warmup:
                latencies.append(elapsed)
            buffer.destroy()

    elif mode == "connect":
        buffer = create_buffer(
            rank, disable_ll_nvlink, tcp_store, num_ranks,
            num_experts_per_rank, num_rdma_bytes,
        )
        for i in range(warmup + rounds):
            tcp_store_barrier(tcp_store, rank, num_ranks)
            elapsed = bench_connect(buffer, other_ranks)
            if i >= warmup:
                latencies.append(elapsed)
            if other_ranks:
                buffer.disconnect_ranks(other_ranks)
            time.sleep(0.5)
        buffer.destroy()

    elif mode == "disconnect":
        buffer = create_buffer(
            rank, disable_ll_nvlink, tcp_store, num_ranks,
            num_experts_per_rank, num_rdma_bytes,
        )
        for i in range(warmup + rounds):
            if other_ranks:
                buffer.connect_ranks(other_ranks)
            tcp_store_barrier(tcp_store, rank, num_ranks)
            elapsed = bench_disconnect(buffer, other_ranks)
            if i >= warmup:
                latencies.append(elapsed)
            time.sleep(0.5)
        buffer.destroy()

    elif mode == "reconnect":
        buffer = create_buffer(
            rank, disable_ll_nvlink, tcp_store, num_ranks,
            num_experts_per_rank, num_rdma_bytes,
        )
        if other_ranks:
            buffer.connect_ranks(other_ranks)
        for i in range(warmup + rounds):
            if other_ranks:
                buffer.disconnect_ranks(other_ranks)
            time.sleep(0.5)
            tcp_store_barrier(tcp_store, rank, num_ranks)
            elapsed = bench_connect(buffer, other_ranks)
            if i >= warmup:
                latencies.append(elapsed)
        buffer.destroy()

    elif mode == "destroy":
        for i in range(warmup + rounds):
            buffer = create_buffer(
                rank, disable_ll_nvlink, tcp_store, num_ranks,
                num_experts_per_rank, num_rdma_bytes,
            )
            if other_ranks:
                buffer.connect_ranks(other_ranks)
            tcp_store_barrier(tcp_store, rank, num_ranks)
            elapsed = bench_destroy(buffer)
            if i >= warmup:
                latencies.append(elapsed)

    return {mode: _stats(latencies)}


def worker(torch_rank: int, args: argparse.Namespace):
    server_addr = args.tcp_server if args.tcp_server else "127.0.0.1"
    rank_client = rank_server.RankClient(server_addr, RANK_SERVER_PORT)
    local_rank, global_rank, _ = rank_client.get_rank()
    num_ranks = args.num_ranks

    print(
        f"Process {torch_rank} -> global_rank={global_rank}, local_rank={local_rank}",
        flush=True,
    )

    signal.signal(
        signal.SIGTERM,
        partial(handle_sigterm, rank_client=rank_client),
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    tcp_store = store_group.create_client_store(
        master_addr=server_addr,
        port=TCP_STORE_PORT,
    )

    num_experts = args.num_experts_per_rank * num_ranks
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        args.num_tokens, args.hidden_dim, num_ranks, num_experts,
    )
    if local_rank == 0:
        print(f"Buffer RDMA size: {num_rdma_bytes / 1e6:.1f} MB", flush=True)

    other_ranks = [r for r in range(num_ranks) if r != global_rank]

    common_kwargs = dict(
        rank=global_rank,
        num_ranks=num_ranks,
        other_ranks=other_ranks,
        tcp_store=tcp_store,
        disable_ll_nvlink=args.disable_ll_nvlink,
        num_experts_per_rank=args.num_experts_per_rank,
        num_rdma_bytes=num_rdma_bytes,
        warmup=args.warmup,
        rounds=args.rounds,
    )

    if args.mode == "cycle":
        results = run_cycle(**common_kwargs)
        total_avg = sum(v[0] for v in results.values())
        print(f"[rank {global_rank}] Control plane cycle:", flush=True)
        for op in ("init", "connect", "disconnect", "reconnect", "destroy"):
            avg_t, min_t, max_t = results[op]
            print(
                f"[rank {global_rank}]   {op:12s}: "
                f"avg_t={avg_t * 1e3:.2f} ms, "
                f"min_t={min_t * 1e3:.2f} ms, "
                f"max_t={max_t * 1e3:.2f} ms",
                flush=True,
            )
        print(
            f"[rank {global_rank}]   {'total':12s}: "
            f"avg_t={total_avg * 1e3:.2f} ms",
            flush=True,
        )
    else:
        results = run_single_op(mode=args.mode, **common_kwargs)
        avg_t, min_t, max_t = results[args.mode]
        print(
            f"[rank {global_rank}] {args.mode}: "
            f"avg_t={avg_t * 1e3:.2f} ms, "
            f"min_t={min_t * 1e3:.2f} ms, "
            f"max_t={max_t * 1e3:.2f} ms",
            flush=True,
        )

    print(f"[rank {global_rank}] done", flush=True)


def run_server():
    _store = store_group.create_master_store(port=TCP_STORE_PORT)  # noqa: F841
    rank_server.start_server(port=RANK_SERVER_PORT)


def main():
    parser = argparse.ArgumentParser(description="Control Plane Latency Test")
    parser.add_argument(
        "--mode",
        type=str,
        default="cycle",
        choices=["cycle", "init", "connect", "disconnect", "reconnect", "destroy"],
        help="Operation to benchmark (default: cycle)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of worker processes to launch",
    )
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=None,
        help="Total number of ranks across all nodes "
             "(default: same as --num-processes)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=128,
        help="Number of tokens (for buffer sizing)",
    )
    parser.add_argument(
        "--num-experts-per-rank",
        type=int,
        default=2,
        help="Number of experts per rank (for buffer sizing)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=7168,
        help="Hidden dimension (for buffer sizing)",
    )
    parser.add_argument(
        "--tcp-server",
        type=str,
        help="TCP server address (for both TCPStore and rank server). "
             "If not set, both will be started locally.",
    )
    parser.add_argument(
        "--disable-ll-nvlink",
        action="store_true",
        help="Disable NVLink communication for low-latency kernels",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup rounds before measurement (default: 0)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Measurement rounds (default: 1)",
    )

    args = parser.parse_args()

    if args.num_ranks is None:
        args.num_ranks = args.num_processes

    if not args.tcp_server:
        print("Starting TCPStore and rank server locally", flush=True)
        server_process = torch.multiprocessing.Process(target=run_server, daemon=True)
        server_process.start()
        time.sleep(0.5)

    if args.num_processes == 1:
        worker(0, args)
        return

    ctx = torch.multiprocessing.spawn(
        worker,
        args=(args,),
        nprocs=args.num_processes,
        join=False,
        daemon=False,
        start_method="spawn",
    )

    failed = []
    for i, p in enumerate(ctx.processes):
        p.join()
        if p.exitcode != 0:
            failed.append((i, p.exitcode))
    if failed:
        raise RuntimeError(
            f"Worker processes failed: "
            f"{', '.join(f'worker {i} (exit code {code})' for i, code in failed)}"
        )


if __name__ == "__main__":
    main()
