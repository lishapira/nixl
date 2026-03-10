# NIXL EP: Concepts and APIs

## Dispatch and Combine

- **Dispatch** — Scatter step for MoE: each rank sends token activations to the ranks that own the selected experts. Tokens are routed by `topk_idx`; data is moved via NIXL (RDMA or NVLink). Input: `x` (tokens), `topk_idx` (which experts); output: received tokens per local expert and a handle for combine.

- **Combine** — Gather step for MoE: each rank sends its expert outputs back; the originating rank gathers and does a **weighted reduction** using `topk_weights`. Input: local expert outputs, `topk_idx`, `topk_weights`, and the handle from dispatch; output: combined tensor `[num_tokens, hidden]`.

## Two Official NIXL Interfaces

- **CPU (host) API** (`nixl.h`) — Used for setup and metadata: create **nixlAgent**, get UCX plugin params and **createBackend**, **registerMem** (RDMA/sync/barrier buffers), **getLocalMD** / **loadRemoteMD** or **fetchRemoteMD**, **invalidateRemoteMD**, **prepMemView** / **releaseMemView** (build GPU memory views), and **genNotif** / **getNotifs** (exchange peer info for GPU addressing). No data transfer on host; it only prepares for device transfers.

- **GPU (device) API** (`nixl_device.cuh`) — Used inside CUDA kernels for actual transfers: **nixlGetPtr**(remote_mvh, rank), **nixlMemViewElem**, **nixlPut** (GPU-initiated send), **nixlAtomicAdd** (signaling). Dispatch and combine run on device using these primitives.

## The `nixl/examples/device/ep` Folder

| Path | Purpose |
|------|--------|
| `README.md` | Build, run, and API overview |
| `nixl_ep/` | Python package: `Buffer`, utils |
| `nixl_ep/buffer.py` | Python API: init, update_memory_buffers, connect/disconnect_ranks, dispatch, combine |
| `csrc/` | C++/CUDA: NIXL agent, buffers, dispatch/combine kernels |
| `csrc/kernels/nixl_ep.cu` | CUDA kernels using NIXL device API |
| `tests/elastic/` | Elastic scaling tests and plan-driven phases |

## Buffer (short)

The **buffer** is **GPU (device) memory**: RDMA buffer, sync buffers, and workspace are allocated with `cudaMalloc`. The `Buffer` object in Python/C++ lives on the host and holds rank, NIXL agent/backend state, and **pointers** to that device memory. “RDMA buffer” here means GPU memory used for NIXL transfers (RDMA or NVLink).

## Main APIs

| API | Purpose |
|-----|--------|
| `Buffer(rank_id, ...)` | Create the NIXL-backed buffer object for this rank (host only; no GPU alloc yet). |
| `update_memory_buffers(num_ranks, num_experts_per_rank, num_rdma_bytes)` | Allocate and lay out GPU communication buffers; register with NIXL. Call once. |
| `connect_ranks(remote_ranks)` | Establish NIXL connections to remote ranks (metadata + transport). Can be called multiple times to add ranks. |
| `disconnect_ranks(remote_ranks)` | Tear down connections and release state for those ranks. |
| `dispatch(x, topk_idx, ...)` | Scatter tokens to expert-owning ranks; returns received data and a handle. |
| `combine(x, topk_idx, topk_weights, handle, ...)` | Gather and weighted-reduce expert outputs; returns combined tensor. |

Use `Buffer.get_rdma_size_hint(...)` to get a safe `num_rdma_bytes` for `update_memory_buffers`.
