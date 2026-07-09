# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0
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
import random
import signal
import sys
import threading
import time
from functools import partial
from typing import List, Optional, Set, cast

import nixl_ep
import rank_server
import store_group
import torch
from fault_artifacts import ArtifactCapture, enabled as _artifacts_enabled
from nixl_ep.buffer import DEFAULT_TIMEOUT_MS
from plan import Plan

# Add tests directory to path to import test utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (  # noqa: E402
    bench,
    bench_kineto,
    calc_diff,
    hash_tensor,
    per_token_cast_back,
)

TCP_STORE_PORT = 9999
RANK_SERVER_PORT = 10000

# In-kernel fault timings: 4 phases x 2 hook modes = 8 cells. The four phases
# are dispatch-send (target=1), dispatch-receive (2), combine-send (3),
# combine-receive (4). The two hook modes differ only in how dispatch/combine
# is launched, NOT in what the GPU kernel marker does once armed:
#
#   * `*-during-kernel-no-hook`          Marker is armed in the
#                                        return_recv_hook=False branch before
#                                        the single FUSED send+receive kernel
#                                        launch. The kernel internally fires
#                                        the marker the moment execution
#                                        reaches the marked phase, regardless
#                                        of which phase (1..4) it is.
#   * `*-during-kernel-hook-separated`   Marker is armed in the
#                                        return_recv_hook=True branch, which
#                                        SPLITS dispatch/combine into two
#                                        kernels (send, then receive) with a
#                                        host-side hook() between them. The
#                                        send-phase marker is armed before
#                                        the dispatch/combine call; the
#                                        receive-phase marker is armed in
#                                        the host gap, between the call and
#                                        hook().
#
# Both flavours of the same (op, phase) cell target the SAME marker target id
# and SAME (entered, exited) snapshot slot pair (the dicts below have
# matching values per pair). The only thing that differs is WHERE we arm.
IN_KERNEL_FAULT_TARGETS = {
    "dispatch-send-during-kernel-hook-separated": 1,
    "dispatch-send-during-kernel-no-hook": 1,
    "dispatch-receive-during-kernel-hook-separated": 2,
    "dispatch-receive-during-kernel-no-hook": 2,
    "combine-send-during-kernel-hook-separated": 3,
    "combine-send-during-kernel-no-hook": 3,
    "combine-receive-during-kernel-hook-separated": 4,
    "combine-receive-during-kernel-no-hook": 4,
}

IN_KERNEL_FAULT_MARKER_SLOTS = {
    "dispatch-send-during-kernel-hook-separated": (4, 5),
    "dispatch-send-during-kernel-no-hook": (4, 5),
    "dispatch-receive-during-kernel-hook-separated": (6, 7),
    "dispatch-receive-during-kernel-no-hook": (6, 7),
    "combine-send-during-kernel-hook-separated": (8, 9),
    "combine-send-during-kernel-no-hook": (8, 9),
    "combine-receive-during-kernel-hook-separated": (10, 11),
    "combine-receive-during-kernel-no-hook": (10, 11),
}

# P2P probe - must stay in sync with InKernelFaultMarkerIndex in
# csrc/nixl_ep.hpp.
IN_KERNEL_P2P_PROBE_TARGET_SLOT = 12
IN_KERNEL_P2P_NULL_COUNT_SLOT = 13
IN_KERNEL_P2P_NONNULL_COUNT_SLOT = 14
IN_KERNEL_P2P_PROBE_DISABLED = -1


def non_negative_int(value: str) -> int:
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if int_value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return int_value


def handle_sigterm(
    signum,
    frame,
    buffer: nixl_ep.Buffer,
    plan: Plan,
    rank_client: rank_server.RankClient,
):
    print(
        f"SIGTERM ({signum}) received for process {os.getpid()}! releasing rank and exiting...",
        flush=True,
    )
    if plan is not None:
        rank_client.release_rank(user_context=plan.get_phase())
    else:
        rank_client.release_rank()
    if buffer is not None and buffer.runtime is not None:
        buffer.destroy()  # to invalidate local MD
        del buffer

    # Continue with default signal handler
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)


def self_kill(signum: int = signal.SIGTERM):
    os.kill(os.getpid(), signum)


def _build_expected_mask(
    ground_truth_dead_ranks: Set[int],
    max_num_ranks: int,
) -> torch.Tensor:
    """Construct the ground-truth mask tensor from the test plan.

    `mask_status` semantics: 1 means "rank is masked / failed", 0 means alive.
    The ground truth is derived from the orchestrator plan (which ranks the
    plan says were killed up to and including the current phase), NOT from the
    runtime mask buffer itself, so we can validate the runtime mask.
    """
    expected = torch.zeros((max_num_ranks,), dtype=torch.int32, device="cuda")
    for r in ground_truth_dead_ranks:
        expected[r] = 1
    return expected


def _check_mask_no_false_positives(
    mask_status: torch.Tensor,
    expected_mask: torch.Tensor,
    rank: int,
    where: str,
    counters: Optional[List[int]] = None,
) -> None:
    """Assert the runtime mask never marks an alive rank as dead.

    False negatives (a killed rank not yet detected) are tolerated mid-test
    because failure detection is asynchronous, but a false positive would mean
    the runtime is excluding tokens from a rank that is actually alive, which
    would silently corrupt the dispatch/combine correctness checks below.

    `counters`, if provided, is a [calls, passes] mutable pair; the test
    harness greps the per-phase summary that test_main prints from it.
    """
    if counters is not None:
        counters[0] += 1
    actual_dead = mask_status != 0
    expected_dead = expected_mask != 0
    false_positives = actual_dead & ~expected_dead
    if false_positives.any().item():
        raise AssertionError(
            f"[rank {rank}] runtime mask has false positives at {where}: "
            f"actual={mask_status.cpu().tolist()}, "
            f"expected={expected_mask.cpu().tolist()}"
        )
    if counters is not None:
        counters[1] += 1


def _maybe_log_first_mask_detection(
    mask_status: torch.Tensor,
    rank: int,
    where: str,
    observed_dead_ranks: Set[int],
) -> None:
    """Emit a structured first-detection event for each peer the runtime mask
    has just started reporting as dead.

    Pairs with the victim-side `Killing rank at ... timestamp_ns=` /
    `HIT_IN_KERNEL_WINDOW ... timestamp_ns=` lines so the sweep harness can
    compute true mask-propagation latency = (this rank's first observation
    timestamp) - (victim's kill timestamp). The previous "detection latency"
    metric measured time-to-end-of-phase, not time-to-mask-update, and was
    dominated by the remaining test_main iteration budget on the survivor.
    """
    num = int(mask_status.numel())
    for j in range(num):
        if mask_status[j].item() != 0 and j not in observed_dead_ranks:
            observed_dead_ranks.add(j)
            print(
                f"[rank {rank}] MASK DETECTED dead_rank={j} where={where} "
                f"timestamp_ns={time.time_ns()}",
                flush=True,
            )


def test_main(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    max_num_ranks: int,
    buffer: nixl_ep.Buffer,
    use_logfmt: bool = False,
    seed: int = 0,
    kineto: bool = False,
    fault_tolerance_test: bool = False,
    fault_kill_timing: str = "before-dispatch",
    fault_kill_signal: str = "sigterm",
    fault_inject_mode: str = "sigkill",
    in_kernel_fault_spin_cycles: int = 0,
    p2p_probe_target: int = IN_KERNEL_P2P_PROBE_DISABLED,
    fault_evidence_dir: str = "",
    ground_truth_dead_ranks: Optional[Set[int]] = None,
    artifacts: "Optional[ArtifactCapture]" = None,
):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    # Track masked ranks (like shrink_test in elastic.py)
    mask_status = torch.zeros((max_num_ranks,), dtype=torch.int32, device="cuda")

    # Build the ground-truth mask once. It's used to (a) verify that the
    # runtime mask never has false positives and (b) to drive the per-rank
    # cross-validation of received tokens against an independently computed
    # set of surviving ranks (instead of trusting the runtime mask). It is
    # derived purely from the orchestrator plan, so it's independent of the
    # fault-kill timing (CPU-level or in-kernel).
    expected_mask = (
        _build_expected_mask(ground_truth_dead_ranks, max_num_ranks)
        if ground_truth_dead_ranks is not None
        else None
    )
    # [calls, passes] counters for the no-false-positive mask check; the
    # sweep harness greps the per-phase summary printed at the end of
    # test_main to confirm the safety property held on every iteration.
    mask_check_counters: List[int] = [0, 0]
    # Tracks every peer that has been seen as dead in the runtime mask in
    # this test_main call. We emit one MASK DETECTED line per peer the first
    # time it transitions from alive to dead in mask_status, so the sweep
    # parser can pair the victim's kill timestamp with the survivor's first
    # observation of that kill in the mask. Reset per test_main invocation
    # (i.e. per phase) so latency is measured against THIS phase's kill.
    observed_dead_ranks: Set[int] = set()

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (
        rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for i in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(
            torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
            * 0.5
            * random.random()
        )
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(
        torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.1
    )

    torch.manual_seed(seed + rank + 1000)
    torch.cuda.manual_seed(seed + rank + 1000)
    random.seed(seed + rank)

    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_idx = topk_idx.to(nixl_ep.topk_idx_t)
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    ).abs()

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = (
            -1
        )

    all_topk_idx = torch.full(
        (max_num_ranks, num_tokens, num_topk), -1, dtype=topk_idx.dtype, device="cuda"
    )
    for r in range(num_ranks):
        # Use same deterministic reset as above (seed + r + 1000)
        torch.manual_seed(seed + r + 1000)
        torch.cuda.manual_seed(seed + r + 1000)
        r_random = random.Random(seed + r)
        r_scores = (
            torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device="cuda"
            ).abs()
            + 1
        )
        r_topk_idx = torch.topk(r_scores, num_topk, dim=-1, largest=True, sorted=True)[
            1
        ]
        r_topk_idx = r_topk_idx.to(nixl_ep.topk_idx_t)
        # Apply same random masking
        for i in range(10):
            r_topk_idx[
                r_random.randint(0, num_tokens - 1), r_random.randint(0, num_topk - 1)
            ] = -1
        all_topk_idx[r] = r_topk_idx

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    kill_scheduled = False
    kill_signal = signal.SIGKILL if fault_kill_signal == "sigkill" else signal.SIGTERM
    in_kernel_sequence = os.getpid() % 1_000_000 + 1
    # Unmap-fault mode selection. In unmap-mid-flight the process SURVIVES;
    # peers observe a stale mapping. See EXPERIMENT_UNMAP_FAULT.md.
    _use_unmap_inject = fault_inject_mode == "unmap-mid-flight"

    def _do_inject_unmap():
        """Fire the unmap injection on THIS rank. Called from a Timer to keep
        the main thread in-flight so peers are actively touching the buffer."""
        try:
            print(
                f"[rank {rank}] INJECT unmap "
                f"timing={fault_kill_timing} timestamp_ns={time.time_ns()}",
                flush=True,
            )
            if artifacts is not None:
                artifacts.mark_inject(mode=fault_inject_mode, is_victim=True)
            buffer.inject_unmap_fault()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[rank {rank}] INJECT unmap FAILED: {exc!r} "
                f"timestamp_ns={time.time_ns()}",
                flush=True,
            )

    def maybe_schedule_self_kill(timing: str):
        nonlocal kill_scheduled
        if (
            fault_tolerance_test
            and not kill_scheduled
            and fault_kill_timing == timing
        ):
            kill_scheduled = True
            if _use_unmap_inject:
                print(
                    f"[rank {rank}] Scheduling unmap inject at {timing} "
                    f"mode={fault_inject_mode} timestamp_ns={time.time_ns()}",
                    flush=True,
                )
                timer = threading.Timer(0.0001, _do_inject_unmap)
            else:
                print(
                    f"[rank {rank}] Killing rank at {timing} "
                    f"timestamp_ns={time.time_ns()}",
                    flush=True,
                )
                timer = threading.Timer(0.0001, self_kill, args=(kill_signal,))
            timer.start()

    def maybe_schedule_in_kernel_self_kill(timing: str):
        nonlocal kill_scheduled
        if (
            not fault_tolerance_test
            or kill_scheduled
            or fault_kill_timing != timing
        ):
            return
        target = IN_KERNEL_FAULT_TARGETS[timing]
        entered_idx, exited_idx = IN_KERNEL_FAULT_MARKER_SLOTS[timing]
        kill_scheduled = True
        buffer.enable_in_kernel_fault_marker(
            target, in_kernel_sequence, in_kernel_fault_spin_cycles
        )
        evidence_dir = fault_evidence_dir or os.getcwd()
        os.makedirs(evidence_dir, exist_ok=True)
        evidence_path = os.path.join(
            evidence_dir,
            f"in_kernel_fault_rank{rank}_{timing}_pid{os.getpid()}.log",
        )

        def helper():
            deadline = time.monotonic() + (DEFAULT_TIMEOUT_MS / 1000)
            while time.monotonic() < deadline:
                snapshot = buffer.get_in_kernel_fault_marker_snapshot()
                entered = snapshot[entered_idx]
                exited = snapshot[exited_idx]
                if entered >= in_kernel_sequence:
                    verdict = (
                        "MISSED_IN_KERNEL_TIMING"
                        if exited >= in_kernel_sequence
                        else "HIT_IN_KERNEL_WINDOW"
                    )
                    evidence = (
                        f"verdict={verdict}\n"
                        f"rank={rank}\n"
                        f"pid={os.getpid()}\n"
                        f"timing={timing}\n"
                        f"target={target}\n"
                        f"sequence={in_kernel_sequence}\n"
                        f"entered={entered}\n"
                        f"exited_before_sigkill={exited}\n"
                        f"signal={kill_signal}\n"
                        f"timestamp_ns={time.time_ns()}\n"
                    )
                    with open(evidence_path, "w", encoding="utf-8") as f:
                        f.write(evidence)
                        f.flush()
                        os.fsync(f.fileno())
                    print(f"[rank {rank}] {verdict} {evidence.strip()}", flush=True)
                    if verdict == "HIT_IN_KERNEL_WINDOW":
                        if _use_unmap_inject:
                            # In unmap modes we deliberately do NOT kill; the
                            # unmap call is what perturbs the wire.
                            #
                            # After Buffer::inject_unmap_fault() had its
                            # internal cudaDeviceSynchronize() removed, the
                            # verdict above only proves DETECTION landed
                            # inside the marker window. To know whether the
                            # actual cuMemUnmap ran while the kernel was
                            # still spinning (TRUE_IN_KERNEL_UNMAP) or after
                            # the kernel exited (LATE_UNMAP), we re-read the
                            # exited slot immediately after the injector
                            # returns.
                            _do_inject_unmap()
                            try:
                                post_snap = buffer.get_in_kernel_fault_marker_snapshot()
                                post_exited = post_snap[exited_idx]
                            except Exception as _post_exc:  # noqa: BLE001
                                post_exited = -1
                                print(
                                    f"[rank {rank}] post-unmap snapshot failed "
                                    f"(context likely poisoned): {_post_exc!r}",
                                    flush=True,
                                )
                            unmap_verdict = (
                                "TRUE_IN_KERNEL_UNMAP"
                                if 0 <= post_exited < in_kernel_sequence
                                else "LATE_UNMAP"
                            )
                            print(
                                f"[rank {rank}] {unmap_verdict} "
                                f"exited_after_unmap={post_exited} "
                                f"sequence={in_kernel_sequence} "
                                f"timestamp_ns={time.time_ns()}",
                                flush=True,
                            )
                            try:
                                with open(evidence_path, "a", encoding="utf-8") as f:
                                    f.write(
                                        f"unmap_verdict={unmap_verdict}\n"
                                        f"exited_after_unmap={post_exited}\n"
                                        f"post_unmap_timestamp_ns={time.time_ns()}\n"
                                    )
                                    f.flush()
                                    os.fsync(f.fileno())
                            except OSError:
                                pass
                        else:
                            os.kill(os.getpid(), kill_signal)
                    return
                time.sleep(0.00001)
            with open(evidence_path, "w", encoding="utf-8") as f:
                f.write(
                    f"verdict=IN_KERNEL_MARKER_TIMEOUT\n"
                    f"rank={rank}\n"
                    f"pid={os.getpid()}\n"
                    f"timing={timing}\n"
                    f"target={target}\n"
                    f"sequence={in_kernel_sequence}\n"
                    f"timestamp_ns={time.time_ns()}\n"
                )
                f.flush()
                os.fsync(f.fileno())
            print(
                f"[rank {rank}] IN_KERNEL_MARKER_TIMEOUT timing={timing}",
                flush=True,
            )

        print(
            f"[rank {rank}] Armed in-kernel SIGKILL at {timing} "
            f"sequence={in_kernel_sequence} spin_cycles={in_kernel_fault_spin_cycles}",
            flush=True,
        )
        threading.Thread(target=helper, daemon=True).start()

    # P2P probe: from each non-victim rank's send warp, count how many
    # p2p_ptr_get(dst_rank=victim) calls returned null vs. non-null.
    # non-null == the peer's own imported P2P mapping into the victim's
    # fabric memory is still valid from the peer's GPU perspective (rank
    # <victim> is still a "valid NVLink destination" for this rank). Works
    # for both --fault-inject-mode sigkill and unmap-mid-flight. See
    # WHY_NO_NVLINK_TRANSPORT_FAULT.md for the reasoning.
    _probe_p2p = (
        not fault_tolerance_test
        and p2p_probe_target >= 0
        and p2p_probe_target != rank
    )
    if _probe_p2p:
        buffer.set_p2p_probe_target(p2p_probe_target)
        buffer.reset_p2p_probe_counts()
        print(
            f"[rank {rank}] P2P_PROBE armed target={p2p_probe_target} "
            f"timestamp_ns={time.time_ns()}",
            flush=True,
        )

    def _log_p2p_probe_snapshot(tag: str) -> None:
        if not _probe_p2p:
            return
        snap = buffer.get_in_kernel_fault_marker_snapshot()
        null_ct = snap[IN_KERNEL_P2P_NULL_COUNT_SLOT]
        nonnull_ct = snap[IN_KERNEL_P2P_NONNULL_COUNT_SLOT]
        print(
            f"[rank {rank}] P2P_PROBE_COUNTS tag={tag} "
            f"target={p2p_probe_target} null={null_ct} nonnull={nonnull_ct} "
            f"timestamp_ns={time.time_ns()}",
            flush=True,
        )

    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False,):
                    for use_ue8m0 in (False, True) if round_scale else (False,):
                        num_times += 1
                        for i in range((num_times % 2) + 1):
                            # Kill this rank at the selected CPU-level timing if marked to be killed.
                            maybe_schedule_self_kill("before-dispatch")

                            cumulative_local_expert_recv_stats = torch.zeros(
                                (num_local_experts,), dtype=torch.int, device="cuda"
                            )
                            if return_recv_hook:
                                # Hook-separated dispatch: send and receive
                                # run as two separate kernels with a host
                                # hook() between them. Send-phase marker arms
                                # here; receive-phase marker arms below in
                                # the host gap between dispatch() and hook().
                                maybe_schedule_in_kernel_self_kill(
                                    "dispatch-send-during-kernel-hook-separated"
                                )
                            else:
                                # No-hook fused dispatch: single send+receive
                                # kernel. Only place to arm an in-kernel
                                # marker for this call; the kernel fires it
                                # internally when execution reaches the marked
                                # phase (target=1 -> send region, target=2 ->
                                # recv region). Both arm calls are listed but
                                # only the one matching fault_kill_timing
                                # actually arms (the other is a no-op via the
                                # guard in maybe_schedule_in_kernel_self_kill).
                                maybe_schedule_in_kernel_self_kill(
                                    "dispatch-send-during-kernel-no-hook"
                                )
                                maybe_schedule_in_kernel_self_kill(
                                    "dispatch-receive-during-kernel-no-hook"
                                )
                            packed_recv_x, packed_recv_count, handle, event, hook = (
                                buffer.dispatch(
                                    current_x,
                                    topk_idx,
                                    num_tokens,
                                    num_experts,
                                    use_fp8=dispatch_use_fp8,
                                    round_scale=round_scale,
                                    use_ue8m0=use_ue8m0,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    async_finish=not return_recv_hook,
                                    return_recv_hook=return_recv_hook,
                                )
                            )
                            if return_recv_hook:
                                maybe_schedule_self_kill("dispatch-between-send-receive")
                                maybe_schedule_in_kernel_self_kill(
                                    "dispatch-receive-during-kernel-hook-separated"
                                )
                            hook() if return_recv_hook else event.current_stream_wait()
                            maybe_schedule_self_kill("after-dispatch")
                            _log_p2p_probe_snapshot("post-dispatch")
                        # Query mask buffer to get current failure status
                        buffer.query_mask_buffer(mask_status)
                        # Record first-observation timestamps so the sweep
                        # can compute true mask-propagation latency from the
                        # victim's kill timestamp to this rank's first sight
                        # of the kill in the mask.
                        _maybe_log_first_mask_detection(
                            mask_status,
                            rank,
                            where="post-dispatch",
                            observed_dead_ranks=observed_dead_ranks,
                        )
                        # The runtime mask must never claim an alive rank is
                        # dead - that would corrupt the dispatch checks below.
                        # We do not require an exact match here because the
                        # kill victim may not yet have been detected on the
                        # very first iteration.
                        if expected_mask is not None and not fault_tolerance_test:
                            _check_mask_no_false_positives(
                                mask_status,
                                expected_mask,
                                rank,
                                where="post-dispatch",
                                counters=mask_check_counters,
                            )
                        maybe_schedule_self_kill("between-dispatch-combine")
                        packed_recv_x = (
                            (packed_recv_x[0], packed_recv_x[1].contiguous())
                            if dispatch_use_fp8
                            else packed_recv_x
                        )
                        simulated_gemm_x = (
                            per_token_cast_back(
                                packed_recv_x[0].view(-1, hidden),
                                packed_recv_x[1].view(-1, hidden // 128),
                            ).view(packed_recv_x[0].shape)
                            if dispatch_use_fp8
                            else cast(torch.Tensor, packed_recv_x).clone()
                        )

                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = (
                                per_token_cast_back(
                                    packed_recv_x[0][i], packed_recv_x[1][i]
                                )
                                if dispatch_use_fp8
                                else packed_recv_x[i]
                            )
                            recv_count, recv_src_info, recv_layout_range = (
                                packed_recv_count[i],
                                handle[0][i],
                                handle[1][i],
                            )

                            # Check expert indices
                            int_mask = (2**32) - 1
                            num_valid_tokens = recv_count.item()
                            assert (
                                cumulative_local_expert_recv_stats[i].item()
                                == num_valid_tokens
                            ), f"{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}"
                            assert (
                                num_valid_tokens
                                == (recv_layout_range & int_mask).sum().item()
                            ), f"{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()"
                            # Per-peer mask / transfer consistency.
                            #
                            # A late in-kernel kill gives the runtime mask
                            # legitimate freedom for a killed peer:
                            #   - if the peer died AFTER fully transferring its
                            #     tokens to this rank, it may still be treated
                            #     as alive and its (complete) data is valid;
                            #   - if it died MID-transfer, it must be treated as
                            #     dead and its partial data must be discarded.
                            # So we validate each peer j against the RUNTIME
                            # mask rather than requiring it to equal ground
                            # truth:
                            #   runtime-alive peer -> ALL of j's tokens received
                            #   runtime-dead  peer -> NONE of j's tokens kept
                            # plus the ground-truth safety check that a peer the
                            # plan never killed is never masked (no false
                            # positive).
                            for j in range(num_ranks):
                                full_j = (all_topk_idx[j] == expert_id).sum().item()
                                recv_j = (recv_layout_range[j] & int_mask).item()
                                runtime_alive_j = mask_status[j].item() == 0
                                if (
                                    expected_mask is not None
                                    and expected_mask[j].item() == 0
                                ):
                                    assert runtime_alive_j, (
                                        f"[rank {rank}] expert {expert_id}: peer {j} "
                                        f"is alive in the plan but the runtime masked "
                                        f"it (false positive)"
                                    )
                                if runtime_alive_j:
                                    assert recv_j == full_j, (
                                        f"[rank {rank}] expert {expert_id}: peer {j} "
                                        f"treated as alive but only {recv_j}/{full_j} "
                                        f"tokens were received (an incomplete transfer "
                                        f"must not be accepted as alive)"
                                    )
                                else:
                                    assert recv_j == 0, (
                                        f"[rank {rank}] expert {expert_id}: peer {j} "
                                        f"masked as dead but {recv_j} tokens were kept "
                                        f"(a partial transfer must be discarded)"
                                    )

                            # Aggregate: the total received equals the sum of
                            # every RUNTIME-alive peer's tokens for this expert.
                            # Safe to trust the runtime mask here because the
                            # per-peer loop above already proved it is a
                            # consistent interpretation of the transfer.
                            expected_num_tokens = (
                                (all_topk_idx == expert_id)
                                .sum(dim=[1, 2])[mask_status == 0]
                                .sum()
                                .item()
                            )
                            assert num_valid_tokens == expected_num_tokens, (
                                f"[rank {rank}] expert {expert_id}: "
                                f"got {num_valid_tokens} tokens, "
                                f"expected {expected_num_tokens} from runtime-alive ranks"
                            )

                            if num_valid_tokens == 0:
                                continue
                            # Check received data
                            if current_x is x:
                                recv_x = recv_x[:num_valid_tokens]
                                recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                recv_src_info = recv_src_info[:num_valid_tokens]
                                assert torch.equal(
                                    recv_x_amin, recv_x[:, :-128].amax(dim=-1)
                                )
                                if round_scale:
                                    assert (
                                        calc_diff(recv_x[:, -1], recv_src_info.view(-1))
                                        < 0.007
                                    )
                                else:
                                    assert (
                                        recv_x[:, -128:]
                                        - recv_src_info.view(-1, 1) % num_tokens
                                    ).sum().item() == 0
                                # Verify the payload from every RUNTIME-alive
                                # peer (these are the peers whose data we kept).
                                for j in range(num_ranks):
                                    if mask_status[j].item() != 0:
                                        continue
                                    begin_idx, count = (
                                        recv_layout_range[j] >> 32
                                    ).item(), (recv_layout_range[j] & int_mask).item()
                                    if not round_scale:
                                        assert (
                                            recv_x_amin == j - rank_offset
                                        ).sum().item() == (
                                            all_topk_idx[j] == expert_id
                                        ).sum().item()
                                        assert (
                                            recv_x[begin_idx : begin_idx + count, :-128]
                                            - j
                                            + rank_offset
                                        ).sum().item() == 0
                            if dispatch_use_fp8:
                                hash_value ^= hash_tensor(
                                    packed_recv_x[0][i, :num_valid_tokens]
                                )
                                hash_value ^= hash_tensor(
                                    packed_recv_x[1][i, :num_valid_tokens]
                                )
                            else:
                                hash_value ^= hash_tensor(
                                    cast(torch.Tensor, packed_recv_x)[
                                        i, :num_valid_tokens
                                    ]
                                )
                        if return_recv_hook and do_check:
                            print(
                                f"[rank {rank}] hook dispatch assertion passed "
                                f"return_recv_hook=True timing={fault_kill_timing}",
                                flush=True,
                            )

                        # Check combine correctness
                        for zero_copy in (False,) if use_logfmt else (False, True):
                            if zero_copy:
                                buffer.get_next_combine_buffer(handle)[
                                    :, :, :
                                ] = simulated_gemm_x
                            out = torch.empty(
                                (num_tokens, hidden),
                                dtype=torch.bfloat16,
                                device="cuda",
                            )
                            maybe_schedule_self_kill("before-combine")
                            if return_recv_hook:
                                # Hook-separated combine: send and receive run
                                # as two separate kernels with a host hook()
                                # between them. Send-phase marker arms here;
                                # receive-phase marker arms below in the host
                                # gap between combine() and hook().
                                maybe_schedule_in_kernel_self_kill(
                                    "combine-send-during-kernel-hook-separated"
                                )
                            else:
                                # No-hook fused combine: single send+receive
                                # kernel. Only place to arm an in-kernel
                                # marker for this call; the kernel internally
                                # fires it for whichever target (3=send,
                                # 4=recv) is armed. Both arm calls are listed
                                # but only the one matching fault_kill_timing
                                # actually arms (the other is a no-op via the
                                # guard in maybe_schedule_in_kernel_self_kill).
                                maybe_schedule_in_kernel_self_kill(
                                    "combine-send-during-kernel-no-hook"
                                )
                                maybe_schedule_in_kernel_self_kill(
                                    "combine-receive-during-kernel-no-hook"
                                )
                            combined_x, event, hook = buffer.combine(
                                simulated_gemm_x,
                                topk_idx,
                                topk_weights,
                                handle,
                                use_logfmt=use_logfmt,
                                async_finish=not return_recv_hook,
                                zero_copy=zero_copy,
                                return_recv_hook=return_recv_hook,
                                out=out,
                            )
                            if return_recv_hook:
                                maybe_schedule_self_kill("combine-between-send-receive")
                                maybe_schedule_in_kernel_self_kill(
                                    "combine-receive-during-kernel-hook-separated"
                                )
                            hook() if return_recv_hook else event.current_stream_wait()
                            maybe_schedule_self_kill("after-combine")
                            _log_p2p_probe_snapshot("post-combine")
                            # Query mask buffer again after combine
                            buffer.query_mask_buffer(mask_status)
                            _maybe_log_first_mask_detection(
                                mask_status,
                                rank,
                                where="post-combine",
                                observed_dead_ranks=observed_dead_ranks,
                            )
                            if expected_mask is not None and not fault_tolerance_test:
                                _check_mask_no_false_positives(
                                    mask_status,
                                    expected_mask,
                                    rank,
                                    where="post-combine",
                                    counters=mask_check_counters,
                                )
                            if do_check:
                                # Adjust topk_idx for validation: any topk
                                # selection that targets an expert owned by a
                                # RUNTIME-masked rank is treated as a
                                # non-selection. We use the runtime mask (not
                                # ground truth) because a peer that died after a
                                # complete transfer is legitimately still
                                # included, and one that died mid-transfer is
                                # legitimately excluded - the per-peer dispatch
                                # checks above already proved the runtime mask
                                # is a consistent interpretation, so the combine
                                # reference must follow the same mask the
                                # runtime actually used.
                                owner_by_expert = (
                                    torch.arange(num_experts, device="cuda")
                                    // num_local_experts
                                )
                                fail_owner_mask = (mask_status != 0).index_select(
                                    0, owner_by_expert
                                )
                                valid_topk_idx = topk_idx >= 0
                                failed_topk_idx = torch.zeros_like(
                                    topk_idx, device="cuda", dtype=torch.bool
                                )
                                failed_topk_idx[valid_topk_idx] = (
                                    fail_owner_mask.index_select(
                                        0, topk_idx[valid_topk_idx].to(torch.int64)
                                    )
                                )
                                topk_idx[failed_topk_idx] = -1
                                diff = calc_diff(
                                    current_x
                                    * topk_weights.masked_fill(topk_idx == -1, 0)
                                    .sum(dim=1)
                                    .view(-1, 1),
                                    combined_x,
                                )
                                assert torch.isnan(combined_x).sum().item() == 0
                                assert diff < (
                                    9e-4 if dispatch_use_fp8 else 1e-5
                                ), f"Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}"
                                hash_value ^= hash_tensor(combined_x)
                                if return_recv_hook:
                                    print(
                                        f"[rank {rank}] hook combine assertion passed "
                                        f"return_recv_hook=True timing={fault_kill_timing}, "
                                        f"diff={diff}",
                                        flush=True,
                                    )

    # NOTE: we intentionally do NOT assert the runtime mask exactly equals the
    # ground-truth mask here. For a late in-kernel kill a peer that died after
    # fully delivering its tokens may legitimately remain unmasked, so an exact
    # match would be too strict. Correctness is instead enforced per-iteration,
    # per-peer in the dispatch loop above: a runtime-alive peer must have
    # delivered all its tokens, and a runtime-dead peer must have had its
    # partial transfer discarded.

    if expected_mask is not None and mask_check_counters[0] > 0:
        # All checks pass silently in _check_mask_no_false_positives; if any
        # had failed we'd already have raised AssertionError. Emit a single
        # greppable summary line so the sweep harness can verify coverage:
        #   passes == calls means every dispatch/combine across the sweep
        #   confirmed no alive peer was masked.
        print(
            f"[rank {rank}] MASK CHECK SUMMARY "
            f"passes={mask_check_counters[1]} calls={mask_check_counters[0]} "
            f"ground_truth_dead={sorted(ground_truth_dead_ranks or [])}",
            flush=True,
        )

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = buffer.dispatch(
            current_x,
            topk_idx,
            num_tokens,
            num_experts,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            use_fp8=True,
            async_finish=False,
            return_recv_hook=return_recv_hook,
        )
        return_recv_hook and large_gemm_with_hook(hook)
        combined_x, event, hook = buffer.combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            use_logfmt=use_logfmt,
            return_recv_hook=return_recv_hook,
        )
        return_recv_hook and large_gemm_with_hook(hook)

    def test_barrier():
        buffer.barrier()

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += (
            num_logfmt10_bytes if use_logfmt else num_bf16_bytes
        ) * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    print(
        f"[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, "
        f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us",
        flush=True,
    )

    # Separate profiling
    if not kineto:
        return

    for return_recv_hook in (False, True):
        buffer.barrier()
        dispatch_t, combine_t = bench_kineto(
            partial(test_func, return_recv_hook=return_recv_hook),
            kernel_names=("dispatch", "combine"),
            barrier_comm_profiling=True,
            suppress_kineto_output=False,
            num_kernels_per_period=2 if return_recv_hook else 1,
            barrier_fn=test_barrier,
        )
        if not return_recv_hook:
            print(
                f"[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | "
                f"Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us",
                flush=True,
            )
        else:
            print(
                f"[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | "
                f"Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us",
                flush=True,
            )


def worker(torch_rank: int, args: argparse.Namespace):
    server_addr = args.tcp_server if args.tcp_server else "127.0.0.1"
    rank_client = rank_server.RankClient(server_addr, RANK_SERVER_PORT)
    local_rank, global_rank, last_active_phase = rank_client.get_rank()
    plan = Plan(
        args.plan,
        global_rank,
        start_phase=last_active_phase if last_active_phase is not None else 0,
    )
    if plan.current_phase == -1:
        print(
            f"Process {torch_rank} -> no plan phases were found for rank {global_rank} after phase {last_active_phase}, exiting",
            flush=True,
        )
        return

    max_num_ranks = plan.get_max_rank() + 1
    print(
        f"Process {torch_rank} -> global_rank={global_rank}, local_rank={local_rank}",
        flush=True,
    )

    # Initialize torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    # Artifact capture (nvlink counters CSV + dmesg + summary.json). Gated by
    # NIXL_FAULT_CAPTURE=1 so the existing SIGKILL sweep is unchanged when the
    # env is unset. Auto-on for non-sigkill fault-inject modes (Phase 3+).
    _capture_wanted = _artifacts_enabled() or getattr(args, "fault_inject_mode", "sigkill") != "sigkill"
    artifacts: Optional[ArtifactCapture] = None
    if _capture_wanted:
        artifacts = ArtifactCapture(
            rank=global_rank,
            evidence_dir=args.fault_evidence_dir or os.getcwd(),
            local_rank=local_rank,
        )
        artifacts.start()

    tcp_store = store_group.create_client_store(
        master_addr=server_addr,
        port=TCP_STORE_PORT,
    )

    # Initialize nixl_ep buffer
    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        args.num_tokens,
        args.hidden_dim,
        max_num_ranks,
        args.num_experts_per_rank * max_num_ranks,
    )
    if local_rank == 0:
        print(f"Allocating buffer size: {num_rdma_bytes / 1e6} MB ...", flush=True)

    buffer = nixl_ep.Buffer(
        rank=global_rank,
        disable_ll_nvlink=args.disable_ll_nvlink,
        explicitly_destroy=True,
        tcp_store_group=tcp_store,
        timeout_ms=args.timeout_ms,
    )
    buffer.update_memory_buffers(
        num_ranks=max_num_ranks,
        num_experts_per_rank=args.num_experts_per_rank,
        num_rdma_bytes=num_rdma_bytes,
    )
    signal.signal(
        signal.SIGTERM,
        partial(handle_sigterm, buffer=buffer, plan=plan, rank_client=rank_client),
    )
    remote_ranks = set()
    mask_status = torch.zeros((max_num_ranks,), dtype=torch.int32, device="cuda")
    # Accumulated set of ranks the orchestrator plan has killed up to and
    # including the current phase. This is the ground truth we validate the
    # runtime mask against in test_main.
    ground_truth_dead_ranks: Set[int] = set()

    # In unmap-mid-flight mode a peer that hits a real P2P fault will raise
    # from within test_main (AssertionError from _check_mask_no_false_
    # positives, or torch.AcceleratorError from a stale CUDA context). Without
    # the try/finally below, artifacts.stop() never runs and summary.json is
    # missing, which makes the Phase 5 pass gate falsely report "victim never
    # fired". We flush artifacts (partial or complete) so the run is
    # diagnosable even on failure.
    _worker_ok = False
    _worker_exc: Optional[BaseException] = None
    try:
        while True:
            print(
                f"global_rank={global_rank}, local_rank={local_rank} -> start phase {plan.get_phase()}",
                flush=True,
            )

            added_ranks = plan.get_new_ranks()
            cleanly_removed = plan.get_removed_ranks()
            ranks_to_kill = plan.get_killed_ranks()

            # If this rank is being removed in this phase, exit gracefully
            if global_rank in cleanly_removed:
                print(
                    f"global_rank={global_rank}, local_rank={local_rank} -> this rank is being removed in this phase, exiting",
                    flush=True,
                )
                rank_client.release_rank(user_context=plan.get_phase())
                break

            if len(added_ranks) > 0:
                print(
                    f"global_rank={global_rank}, local_rank={local_rank} -> adding connections to {added_ranks}",
                    flush=True,
                )
                buffer.connect_ranks(added_ranks)
                remote_ranks.update(added_ranks)

            kill_rank = global_rank in ranks_to_kill

            if len(cleanly_removed) > 0:
                print(
                    f"global_rank={global_rank}, local_rank={local_rank} -> removing connections to {cleanly_removed}",
                    flush=True,
                )
                buffer.disconnect_ranks(cleanly_removed)
                remote_ranks.difference_update(cleanly_removed)
                time.sleep(
                    5
                )  # required to avoid race between MD invalidation and readdition of same ranks, if this is part of the test

            active_ranks_list = plan.get_active_ranks()
            current_num_ranks = max(active_ranks_list) + 1
            current_num_experts = args.num_experts_per_rank * current_num_ranks

            # Ground truth for this phase: every rank the plan has killed so far
            # (previous phases) plus every rank the plan kills in THIS phase.
            # For both sigkill and unmap-mid-flight, peers correctly flag the
            # victim as dead via the mask (SIGKILL: process gone; unmap: peers
            # observe silent victim via NIXL-EP timeout), so ground truth must
            # include the victim in both modes.
            ground_truth_dead_for_phase = ground_truth_dead_ranks | set(ranks_to_kill)
            _ground_truth_for_test = ground_truth_dead_for_phase

            test_main(
                args.num_tokens,
                args.hidden_dim,
                current_num_experts,
                args.num_topk,
                global_rank,
                current_num_ranks,
                max_num_ranks,
                buffer,
                kineto=args.kineto,
                fault_tolerance_test=kill_rank,
                fault_kill_timing=args.fault_kill_timing,
                fault_kill_signal=args.fault_kill_signal,
                fault_inject_mode=args.fault_inject_mode,
                in_kernel_fault_spin_cycles=args.in_kernel_fault_spin_cycles,
                p2p_probe_target=args.p2p_probe_target,
                fault_evidence_dir=args.fault_evidence_dir,
                ground_truth_dead_ranks=_ground_truth_for_test,
                artifacts=artifacts,
            )
            # Persist the ground truth across phases so subsequent phases
            # validate against ALL ranks killed up to that point.
            ground_truth_dead_ranks = ground_truth_dead_for_phase
            buffer.query_mask_buffer(mask_status)
            newly_failed_ranks = set()
            for r in range(current_num_ranks):
                if mask_status[r].item() != 0 and r in remote_ranks:
                    newly_failed_ranks.add(r)

            if len(newly_failed_ranks) > 0:
                print(
                    f"global_rank={global_rank}, local_rank={local_rank} -> "
                    f"detected unexpected rank failures: {newly_failed_ranks}, "
                    f"cleaning up... timestamp_ns={time.time_ns()}",
                    flush=True,
                )
                remote_ranks.difference_update(newly_failed_ranks)
                buffer.disconnect_ranks(list(newly_failed_ranks))
                time.sleep(5)

            print(
                f"global_rank={global_rank}, local_rank={local_rank} -> end phase {plan.get_phase()}",
                flush=True,
            )

            if not plan.next():
                break

        _worker_ok = True
    except BaseException as _ex:
        _worker_exc = _ex
        print(
            f"[rank {global_rank}] WORKER EXCEPTION type={type(_ex).__name__} msg={_ex}",
            flush=True,
        )
    finally:
        # Best-effort buffer teardown. If the exception was a CUDA context
        # corruption, destroy() may itself raise; swallow so the outer
        # summary flush still happens.
        try:
            buffer.destroy()
        except Exception as _dex:
            print(
                f"[rank {global_rank}] buffer.destroy() failed post-exception: {_dex!r}",
                flush=True,
            )
        # Flush artifacts unconditionally so summary.json is present even on
        # failure. mark_recovered reflects whether the worker reached the end
        # of its plan cleanly (True) or exited via exception (False).
        if artifacts is not None:
            try:
                artifacts.mark_recovered(_worker_ok)
                if not _worker_ok and _worker_exc is not None:
                    artifacts.summary.extra["exception"] = f"{type(_worker_exc).__name__}: {_worker_exc}"
                artifacts.stop(extra_settle_sec=0.0)
            except Exception as _aex:
                print(
                    f"[rank {global_rank}] artifacts.stop() failed: {_aex!r}",
                    flush=True,
                )

    print(f"global_rank={global_rank}, local_rank={local_rank} -> done", flush=True)
    # Structured greppable line for the sweep harness. Rank that SIGKILLed
    # itself never reaches here, so the harness can count these lines to
    # verify every expected survivor reached clean exit.
    print(
        f"[rank {global_rank}] WORKER DONE survived={str(_worker_ok).lower()} "
        f"phases_completed={plan.get_phase() + 1}",
        flush=True,
    )
    # If the worker died with an exception, re-raise now that artifacts are
    # flushed. Torch multiprocessing will capture the traceback via its
    # error_files and elastic.main() will surface it.
    if not _worker_ok and _worker_exc is not None:
        raise _worker_exc


def run_server():
    _store = store_group.create_master_store(port=TCP_STORE_PORT)  # noqa: F841
    rank_server.start_server(port=RANK_SERVER_PORT)


def main():
    parser = argparse.ArgumentParser(description="Elastic EP Test")
    parser.add_argument(
        "--plan", type=str, default="plan.json", help="Path to plan file"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of worker processes to launch",
    )
    parser.add_argument("--num-tokens", type=int, default=128, help="Number of tokens")
    parser.add_argument(
        "--num-experts-per-rank", type=int, default=2, help="Number of experts per rank"
    )
    parser.add_argument("--hidden-dim", type=int, default=7168, help="Hidden dimension")
    parser.add_argument("--num-topk", type=int, default=8, help="Number of topk")
    parser.add_argument(
        "--tcp-server",
        type=str,
        help="TCP server address (for both TCPStore and rank server). If not set, both will be started locally.",
    )
    parser.add_argument("--kineto", action="store_true", help="Enable kineto profiling")
    parser.add_argument(
        "--disable-ll-nvlink",
        action="store_true",
        help="Disable NVLink communication for low-latency kernels",
    )
    parser.add_argument(
        "--timeout-ms",
        type=non_negative_int,
        default=DEFAULT_TIMEOUT_MS,
        help="GPU timeout in milliseconds (non-negative integer)",
    )
    parser.add_argument(
        "--fault-kill-timing",
        choices=(
            "before-dispatch",
            "after-dispatch",
            "between-dispatch-combine",
            "dispatch-between-send-receive",
            "before-combine",
            "combine-between-send-receive",
            "after-combine",
            # In-kernel timings: 4 phases x 2 hook modes = 8 cells. The
            # `-no-hook` suffix arms in the return_recv_hook=False pass
            # (single fused send+receive kernel). The `-hook-separated`
            # suffix arms in the return_recv_hook=True pass (send and
            # receive split into two kernels with a host hook between).
            "dispatch-send-during-kernel-hook-separated",
            "dispatch-send-during-kernel-no-hook",
            "dispatch-receive-during-kernel-hook-separated",
            "dispatch-receive-during-kernel-no-hook",
            "combine-send-during-kernel-hook-separated",
            "combine-send-during-kernel-no-hook",
            "combine-receive-during-kernel-hook-separated",
            "combine-receive-during-kernel-no-hook",
        ),
        default="before-dispatch",
        help="CPU-level or in-kernel timing for the fault-tolerance self kill.",
    )
    parser.add_argument(
        "--fault-kill-signal",
        choices=("sigterm", "sigkill"),
        default="sigterm",
        help="Signal to use for the fault-tolerance self kill.",
    )
    parser.add_argument(
        "--fault-inject-mode",
        choices=("sigkill", "unmap-mid-flight"),
        default="sigkill",
        help=(
            "Fault-injection mechanism. 'sigkill' (default): the killed rank "
            "receives --fault-kill-signal. 'unmap-mid-flight': tears down the "
            "real peer-exposed RDMA buffer while peers are actively reading; "
            "process SURVIVES; peers observe a local MMU fault (XID 31 + "
            "cudaErrorIllegalAddress). See EXPERIMENT_UNMAP_FAULT.md."
        ),
    )
    parser.add_argument(
        "--in-kernel-fault-spin-cycles",
        type=non_negative_int,
        default=0,
        help=(
            "GPU spin cycles inside the in-kernel fault marker (only "
            "applies to '*-during-kernel-*' timings). Used to keep the "
            "target send/recv kernel spinning long enough that the "
            "host-thread's fault injection (cuMemUnmap or SIGKILL) "
            "returns while the kernel is still in the marked window "
            "(verdict TRUE_IN_KERNEL_UNMAP / HIT_IN_KERNEL_WINDOW). "
            "GB200 warm-path: ~200_000_000 (~100 ms) reliably lands "
            "in-window; cold-first cuMem* teardown can be ~26 ms so "
            "smaller values may collapse to LATE_UNMAP."
        ),
    )
    parser.add_argument(
        "--p2p-probe-target",
        type=int,
        default=IN_KERNEL_P2P_PROBE_DISABLED,
        help=(
            "dst_rank whose p2p_ptr_get() null / non-null count the "
            "dispatch/combine send-warp on non-victim ranks tracks. "
            "Typically set to the victim rank. Non-null == the peer's "
            "own imported P2P mapping to the victim's fabric memory is "
            "still valid from the peer's GPU perspective "
            "(rank <victim> stays a 'valid NVLink destination' even "
            "after cuMemUnmap/SIGKILL on the victim). Pass -1 (default) "
            "to disable. Emits 'P2P_PROBE_COUNTS tag=...' log lines. "
            "See WHY_NO_NVLINK_TRANSPORT_FAULT.md."
        ),
    )
    parser.add_argument(
        "--fault-evidence-dir",
        type=str,
        default="",
        help="Directory for durable in-kernel SIGKILL evidence files.",
    )

    args = parser.parse_args()

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
    expected_fault_signal = (
        signal.SIGKILL if args.fault_kill_signal == "sigkill" else signal.SIGTERM
    )
    # Per-mode exit-code tolerance:
    #   sigkill        - victim exits via SIGKILL (negative signal); peers 0.
    #   unmap-mid-flight - victim exits 0 (survived at Python level). Peers
    #     MAY exit non-zero: reading a freed VMM region produces
    #     cudaErrorIllegalAddress, which permanently poisons that peer's
    #     CUDA context; any subsequent CUDA op on that context raises.
    #     Our try/finally around the worker loop still flushes summary.json
    #     so the peer's evidence is captured before the process dies. That
    #     non-zero exit is the EXPECTED HARDWARE OUTCOME of a mid-flight
    #     unmap and must not fail the whole run. exit=-11 (SIGSEGV) or
    #     exit=-6 (SIGABRT) also happen when torch's Python-level teardown
    #     races with the corrupted context during interpreter shutdown.
    _fim_main = getattr(args, "fault_inject_mode", "sigkill")
    for i, p in enumerate(ctx.processes):
        p.join()
        if _fim_main == "sigkill":
            allowed = (0, -expected_fault_signal)
        elif _fim_main == "unmap-mid-flight":
            # Peers are allowed to die from illegal-address propagation.
            # We deliberately allow ANY exit code here because the exact
            # signal depends on which torch teardown ordering hits first.
            # phase5_pass_gate.py checks summary.json + observability
            # signals for the real verdict.
            allowed = None
        else:
            allowed = (0,)
        if allowed is not None and p.exitcode not in allowed:
            failed.append((i, p.exitcode))
    if failed:
        # torch mp writes each child's traceback to ctx.error_files[i]
        # (a pickle of the exception). Extract and print so the reason
        # for the failure is visible instead of a bare exit-code list.
        for i, code in failed:
            err_path = None
            try:
                err_path = ctx.error_files[i]
            except (AttributeError, IndexError, KeyError):
                pass
            if err_path and os.path.exists(err_path):
                try:
                    with open(err_path, "rb") as fh:
                        payload = fh.read()
                    print(
                        f"--- worker {i} (rc={code}) traceback ({err_path}) ---",
                        flush=True,
                    )
                    try:
                        import pickle
                        err_obj = pickle.loads(payload)
                        print(repr(err_obj), flush=True)
                    except Exception:
                        try:
                            print(payload.decode("utf-8", errors="replace"), flush=True)
                        except Exception:
                            print(payload, flush=True)
                    print(f"--- end worker {i} traceback ---", flush=True)
                except Exception as ex:
                    print(
                        f"[worker {i}] error-file {err_path} unreadable: {ex}",
                        flush=True,
                    )
        raise RuntimeError(
            f"Worker processes failed: {', '.join(f'worker {i} (exit code {code})' for i, code in failed)}"
        )


if __name__ == "__main__":
    main()
