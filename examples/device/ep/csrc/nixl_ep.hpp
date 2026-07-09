/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Forcibly disable NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include <optional>
#include <tuple>
#include <vector>
#include <string>

#include <memory>
#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"
#include "vmm.hpp"

#include "nixl.h"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME nixl_ep_cpp
#endif

namespace nixl_ep {

enum InKernelFaultMarkerIndex : int {
    IN_KERNEL_FAULT_TARGET = 0,
    IN_KERNEL_FAULT_SEQUENCE = 1,
    IN_KERNEL_FAULT_SPIN_CYCLES = 2,
    IN_KERNEL_FAULT_RESERVED = 3,
    IN_KERNEL_FAULT_DISPATCH_SEND_ENTERED = 4,
    IN_KERNEL_FAULT_DISPATCH_SEND_EXITED = 5,
    IN_KERNEL_FAULT_DISPATCH_RECV_ENTERED = 6,
    IN_KERNEL_FAULT_DISPATCH_RECV_EXITED = 7,
    IN_KERNEL_FAULT_COMBINE_SEND_ENTERED = 8,
    IN_KERNEL_FAULT_COMBINE_SEND_EXITED = 9,
    IN_KERNEL_FAULT_COMBINE_RECV_ENTERED = 10,
    IN_KERNEL_FAULT_COMBINE_RECV_EXITED = 11,
    // P2P probe (post-Approach-A diagnostic): count how many times the
    // dispatch/combine send-warp's p2p_ptr_get(dst_rank=probe_target) call
    // returned null vs. non-null. Disambiguates:
    //   * mechanism (1) NIXL send-path graceful degradation:
    //     probe_target set to victim rank; NULL count > 0 on peers'
    //     post-fault iterations => nixlGetPtr detected the stale fabric
    //     mapping and returned null => warp took the UCX-managed
    //     nixlPut fallback path (no direct st.na.global into freed range).
    //   * mechanism (2) importer-side mapping persistence:
    //     NONNULL count > 0 (and no peer CUDA fault) on peers'
    //     post-fault iterations => nixlGetPtr returned a still-valid
    //     pointer => warp did the direct st.na.global => write landed
    //     on the still-live physical fabric page (peer's imported
    //     mapping survives exporter's cuMemUnmap).
    // Slot layout is monotonic across iterations; enable_in_kernel_fault_marker
    // deliberately does NOT reset slots >= IN_KERNEL_P2P_PROBE_TARGET so
    // pre-fault iteration counts survive to be compared against post-fault.
    IN_KERNEL_P2P_PROBE_TARGET = 12,
    IN_KERNEL_P2P_NULL_COUNT = 13,
    IN_KERNEL_P2P_NONNULL_COUNT = 14,
    IN_KERNEL_FAULT_MARKER_SIZE = 15,
    IN_KERNEL_P2P_PROBE_DISABLED = -1,
};

enum InKernelFaultTarget : int {
    IN_KERNEL_FAULT_DISABLED = 0,
    IN_KERNEL_FAULT_DISPATCH_SEND = 1,
    IN_KERNEL_FAULT_DISPATCH_RECV = 2,
    IN_KERNEL_FAULT_COMBINE_SEND = 3,
    IN_KERNEL_FAULT_COMBINE_RECV = 4,
};

struct NixlPeerInfo {
    void* rdma_buffer_ptr;
    int* sync_buffer_ptr;
    uint64_t* ht_barrier_ptr;
    int device_id;
    int rank;
};

struct NixlAgentInfo
{
    NixlAgentInfo(std::shared_ptr<nixlAgent> agent, nixlBackendH* backend, int max_num_ranks): agent(agent), backend(backend) {
        wire_up_done.resize(max_num_ranks, false);
        remote_agent_names.resize(max_num_ranks);
    }

    std::shared_ptr<nixlAgent> agent;
    std::string agent_name;
    std::vector<std::string> remote_agent_names;
    nixl_opt_args_t extra_params;
    nixlBackendH* backend;
    nixl_reg_dlist_t rdma_reg_descs{VRAM_SEG};
    nixl_reg_dlist_t sync_reg_descs{VRAM_SEG};
    nixl_reg_dlist_t sync_count_reg_descs{VRAM_SEG};
    nixl_reg_dlist_t ht_barrier_reg_descs{VRAM_SEG};
    std::vector<bool> wire_up_done; // [num_peers]
};

struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
    int buffer_idx = 0; // Double buffering index
    bool low_latency_mode = false;
    uint64_t timeout_ms = 30000;

    // NVLink Buffer
    int64_t num_nvl_bytes;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // RDMA Buffer
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    int *mask_buffer_ptr = nullptr;
    int *sync_buffer_ptr = nullptr;
    int *sync_count_ptr = nullptr;

    /* Owning VMM allocations (keep raw ptrs above as aliases) */
    std::unique_ptr<vmm_region> m_rdma_alloc;
    std::unique_ptr<vmm_region> m_mask_alloc;
    std::unique_ptr<vmm_region> m_sync_alloc;
    std::unique_ptr<vmm_region> m_sync_count_alloc;
    std::unique_ptr<vmm_region> m_workspace_alloc;

    // Device info and communication
    int device_id;
    int num_device_sms;
    uint64_t timeout_cycles = 0;
    int rank, rdma_rank, nvl_rank;
    int num_ranks, num_rdma_ranks, num_nvl_ranks;
    std::vector<int> remote_ranks; /* global ranks */
    cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

    // Stream for communication
    at::cuda::CUDAStream comm_stream;

    // After synchronization, this flag will be true
    bool available = false;

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy;
    // After `destroy()` be called, this flag will be true
    bool destroyed = false;

    // Barrier signals
    int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Host-side MoE info
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;

    // Host-side expert-level MoE info
    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;

    // Host-side RDMA-level MoE info
    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;

    // Test-only in-kernel fault markers. GPU writes entered/exited slots, while
    // elastic.py polls the mapped host view before sending SIGKILL.
    volatile int* in_kernel_fault_marker = nullptr;
    int* in_kernel_fault_marker_mapped = nullptr;

    std::unique_ptr<NixlAgentInfo> nixl_agent_info;
    std::vector<NixlPeerInfo> nixl_peer_info;
    NixlPeerInfo my_peer_info;
    int max_num_ranks;
    int max_experts_per_rank;
    nixl_ep::gpu_nixl_ctx gpu_ctx;
    nixl_ep::gpu_nixl_ctx* gpu_ctx_ptr = nullptr;
    uint64_t* last_ht_barrier_counter = nullptr;
    uint64_t* local_ht_barrier_counter = nullptr;

    /* Common private funcs */
    void _nixl_agent_init();
    void _nixl_agents_connect(const std::vector<int>& ranks, const std::vector<nixl_blob_t>& remote_mds = {});
    void _nixl_agents_disconnect(const std::vector<int>& ranks);
    void _nixl_agents_peer_info_gather(std::vector<int>& ranks);
    void _nixl_agents_peer_info_cleanup(const std::vector<int>& ranks);

    void _nixl_ep_init(void);
    void _nixl_ep_memory_views_create(void);
    void _nixl_ep_memory_views_destroy(void);
    void _nixl_ep_destroy(void);

    /* high-throughput mode private funcs */
    void _ipc_handles_sync(const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles);

public:
    Buffer(int rank, bool explicitly_destroy, bool low_latency_mode, int timeout_ms);

    void update_memory_buffers(int num_ranks, int max_experts_per_rank, int64_t num_rdma_bytes, int64_t num_nvl_bytes = 0);

    void connect_ranks(const std::vector<int>& remote_ranks_list, const std::optional<std::vector<nixl_blob_t>>& remote_mds = std::nullopt, const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles = {}, bool activate = true);

    void disconnect_ranks(const std::vector<int>& remote_ranks_list);

    void init(int num_ranks, int max_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes);

    ~Buffer() noexcept;

    bool is_available() const;

    bool is_ht_available() const;

    int get_num_rdma_ranks() const;

    int get_rdma_rank() const;

    int get_root_rdma_rank(bool global) const;

    int get_local_device_id() const;

    pybind11::bytearray get_local_ipc_handle() const;

    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer = false) const;

    torch::Stream get_comm_stream() const;


    void destroy();

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
    get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event,
                        bool async, bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<EventHandle>>
    ht_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                        const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                        const std::optional<torch::Tensor>& num_tokens_per_rank, const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                        const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                        int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                        const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                        const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                        int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    ht_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                        const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
                        const torch::Tensor& src_meta, const torch::Tensor& is_combined_token_in_rank,
                        const torch::Tensor& rdma_channel_prefix_matrix, const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
                        const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_nvl_head,
                        const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    void clean_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
    dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                         const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                         const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                         int num_max_dispatch_tokens_per_rank, int num_experts,
                         bool use_fp8, bool round_scale, bool use_ue8m0,
                         bool async, bool return_recv_hook);

    std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
    combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                        const torch::Tensor& src_info, const torch::Tensor& layout_range,
                        const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
                        int num_max_dispatch_tokens_per_rank, int num_experts,
                        bool use_logfmt, bool zero_copy, bool async, bool return_recv_hook,
                        const std::optional<torch::Tensor>& out = std::nullopt);

    void barrier();

    torch::Tensor
    get_next_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const;

    void update_mask_buffer(int rank_to_mask, bool mask);

    void query_mask_buffer(const torch::Tensor& mask_status);

    void clean_mask_buffer();

    void enable_in_kernel_fault_marker(int target, int sequence, int spin_cycles);

    void disable_in_kernel_fault_marker();

    std::vector<int> get_in_kernel_fault_marker_snapshot() const;

    // Post-Approach-A diagnostic (see IN_KERNEL_P2P_PROBE_TARGET slot
    // comment above): set the target dst_rank whose p2p_ptr_get() null /
    // non-null count the send-warp should track. Pass
    // IN_KERNEL_P2P_PROBE_DISABLED (-1) to disable. Counts are exposed
    // via get_in_kernel_fault_marker_snapshot()[IN_KERNEL_P2P_NULL_COUNT]
    // and [IN_KERNEL_P2P_NONNULL_COUNT].
    void set_p2p_probe_target(int target);

    // Reset the P2P null/nonnull counters to 0 (leaves probe target
    // unchanged). Call between iterations to isolate per-iteration deltas.
    void reset_p2p_probe_counts();

    // Test-only fault injection: tear down THIS rank's peer-exposed RDMA
    // buffer while the process stays alive. Peers actively reading over
    // NVLink attempt a P2P load against the invalidated mapping and take
    // a local MMU fault (XID 31 + cudaErrorIllegalAddress); this rank's
    // next CUDA op on the freed range returns cudaErrorIllegalAddress as
    // well. Safe to call multiple times (subsequent calls are no-ops).
    // See EXPERIMENT_UNMAP_FAULT.md.
    void inject_unmap_fault();

    std::string get_local_metadata() const;
};

} // namespace nixl_ep
