# NIXL EP Performance Tests

Performance tests for NIXL EP Buffer:
- **Data Plane** (`test_data_plane.py`): dispatch/combine throughput and latency
- **Control Plane** (`test_control_plane.py`): init/connect/disconnect/reconnect/destroy latency

## Prerequisites

- CUDA device with RDMA support
- etcd running locally only if using `--use-etcd` (default is TCPStore, no etcd needed)
- Scripts use bare imports and must be run from within this directory (`cd test/python/nixl_ep_perf`)

## Environment Setup

```bash
# Let UCX auto-select devices (container images may override UCX_NET_DEVICES)
unset UCX_NET_DEVICES
```

## Multi-Node

Both tests support multi-node runs via environment variables or CLI flags:

- `WORLD_SIZE` / `--world-size` = number of nodes (not total ranks)
- `RANK` / `--rank` = node rank (0 = master, runs TCPStore/rank server)
- `MASTER_ADDR` / `--master-addr` = master node hostname/IP
- `--num-processes` = GPUs per node; total ranks = WORLD_SIZE x num-processes
- TCPStore is used by default (no etcd dependency); use `--use-etcd` to switch to etcd

---

## Data Plane Tests

Measures throughput and latency for dispatch, combine, and end-to-end (dispatch + combine) operations.

### Usage

```bash
cd test/python/nixl_ep_perf

# IPC/NVLink backend (default)
python3 test_data_plane.py --num-processes=8 --mode=e2e

# RDMA only (force RDMA transport instead of NVLink/IPC)
# Run `unset UCX_NET_DEVICES` first if UCX_NET_DEVICES is set in your environment
python3 test_data_plane.py --num-processes=8 --mode=e2e --disable-ll-nvlink

# Dispatch only (measures dispatch throughput)
python3 test_data_plane.py --num-processes=8 --mode=dispatch

# Combine only (one dispatch, many combines)
python3 test_data_plane.py --num-processes=8 --mode=combine
```

### Multi-Node Setup

```bash
# Master node (RANK=0)
WORLD_SIZE=2 RANK=0 python3 test_data_plane.py --num-processes=8 --mode=e2e

# Worker node (RANK=1)
WORLD_SIZE=2 RANK=1 MASTER_ADDR=node0.example.com \
  python3 test_data_plane.py --num-processes=8 --mode=e2e
```

Or use CLI flags:
```bash
# Master
python3 test_data_plane.py --num-processes=8 --mode=e2e --world-size=2 --rank=0

# Worker
python3 test_data_plane.py --num-processes=8 --mode=e2e --world-size=2 --rank=1 --master-addr=node0
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-processes` | 8 | Number of ranks/GPUs per node |
| `--mode` | e2e | Test mode: dispatch, combine, e2e |
| `--tokens` | 512 | Number of tokens |
| `--hidden` | 4096 | Hidden dimension |
| `--experts-per-rank` | 8 | Experts per rank |
| `--topk` | 2 | TopK value |
| `--disable-ll-nvlink` | false | Disable NVLink communication for low-latency kernels (RDMA only) |
| `--warmup` | 10 | Warmup iterations |
| `--iters` | 100 | Measurement iterations |
| `--timeout` | 300 | Timeout in seconds |
| `--use-etcd` | false | Use etcd for metadata exchange (default: TCPStore) |
| `--world-size` | 1 | Number of nodes (overrides `WORLD_SIZE` env var) |
| `--rank` | 0 | Node rank, 0=master (overrides `RANK` env var) |
| `--master-addr` | — | Master node address (overrides `MASTER_ADDR` env var) |

### Example Output

Generated with:
```bash
python3 test_data_plane.py --num-processes=8 --mode=e2e \
  --tokens=128 --hidden=7168 --experts-per-rank=36 --topk=8 \
  --disable-ll-nvlink --warmup=10 --iters=100
```

```
======================================================================
NIXL EP Data Plane Performance Test
======================================================================
Single-node setup: 8 processes
Mode: e2e
Tokens: 128, Hidden: 7168, TopK: 8
Experts: 36/rank (288 total)
Disable LL NVLink: True
Metadata exchange: TCPStore
Warmup: 10, Measure: 100 iterations
======================================================================

======================================================================
Data Plane (e2e): 8/8 ranks passed
======================================================================
Bandwidth (GB/s): avg=40.04, min=40.03, max=40.04
Latency (μs):     avg=556.1, min=556.1, max=556.3
```

### Expected Performance (RDMA, `--disable-ll-nvlink`)

| Mode | Bandwidth | Latency |
|------|-----------|---------|
| E2E | ~40.0 GB/s | ~556 μs |
| Dispatch | ~35.0 GB/s | ~217 μs |
| Combine | ~42.3 GB/s | ~347 μs |

*Config: 128 tokens, 7168 hidden, topk=8, 288 experts (36/rank), 8 GPUs, `--disable-ll-nvlink`*

---

## Control Plane Tests

Measures latency of control plane operations (init, connect, disconnect, reconnect, destroy).

### Usage

```bash
cd test/python/nixl_ep_perf

# Full cycle (init → connect → disconnect → reconnect → destroy)
python3 test_control_plane.py --num-processes=8

# Specific expert counts
python3 test_control_plane.py --num-processes=8 --experts-per-rank=8,32

# Single operation
python3 test_control_plane.py --num-processes=8 --test=connect

# RDMA only (force RDMA transport instead of NVLink/IPC)
python3 test_control_plane.py --num-processes=8 --disable-ll-nvlink

# Use etcd instead of TCPStore (if needed)
python3 test_control_plane.py --num-processes=8 --use-etcd
```

### Multi-Node Setup

```bash
# Master node (RANK=0)
WORLD_SIZE=2 RANK=0 python3 test_control_plane.py --num-processes=8

# Worker node (RANK=1)
WORLD_SIZE=2 RANK=1 MASTER_ADDR=node0.example.com \
  python3 test_control_plane.py --num-processes=8
```

Or use CLI flags:
```bash
# Master
python3 test_control_plane.py --num-processes=8 --world-size=2 --rank=0

# Worker
python3 test_control_plane.py --num-processes=8 --world-size=2 --rank=1 --master-addr=node0
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-processes` | 8 | Number of ranks/GPUs per node |
| `--test` | cycle | Test to run: init, connect, disconnect, destroy, cycle |
| `--experts-per-rank` | 8,32 | Experts per rank, comma-separated for multiple runs |
| `--num-tokens` | 512 | Number of tokens per rank |
| `--hidden` | 4096 | Hidden dimension |
| `--disable-ll-nvlink` | false | Disable NVLink communication for low-latency kernels (RDMA only) |
| `--warmup` | 0 | Warmup rounds |
| `--rounds` | 1 | Measurement rounds |
| `--timeout` | 300 | Timeout in seconds |
| `--use-etcd` | false | Use etcd for metadata exchange (default: TCPStore) |
| `--world-size` | 1 | Number of nodes (overrides `WORLD_SIZE` env var) |
| `--rank` | 0 | Node rank, 0=master (overrides `RANK` env var) |
| `--master-addr` | — | Master node address (overrides `MASTER_ADDR` env var) |

### Example Output

Generated with:
```bash
python3 test_control_plane.py --num-processes=8 --experts-per-rank=8 --disable-ll-nvlink
```

```
======================================================================
Control Plane: 8 experts/rank x 8 ranks = 64 total
Node results: ranks 0-7 (8 processes)
======================================================================
Operation       Avg (ms)     Min (ms)     Max (ms)
----------------------------------------------------------------------
init            2498.68      2498.68      2498.68
connect         168.09       168.09       168.09
disconnect      4.97         4.97         4.97
reconnect       123.45       123.45       123.45
destroy         708.15       708.15       708.15
----------------------------------------------------------------------
TOTAL           3503.35
======================================================================
```

---

## Files

| File | Description |
|------|-------------|
| `test_data_plane.py` | Data plane test (dispatch/combine/e2e) |
| `test_control_plane.py` | Control plane test (init/connect/disconnect/reconnect/destroy) |
| `mp_runner.py` | Multi-process test runner |
| `rank_server.py` | Coordination server for distributed tests |
| `store_group.py` | PyTorch TCPStore master/client helpers |
