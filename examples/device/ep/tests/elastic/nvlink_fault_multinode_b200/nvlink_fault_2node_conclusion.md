# NVLink fault during two-node NIXL EP on B200 - conclusion

## TL;DR

- These conclusions come from a two-node B200 setup, where each NVLink domain is one node.
- **Previous finding that still holds: the fault stays in the faulted NVLink domain.** Ranks in the other
  NVLink domain reach the victim only over RDMA, and they always detect the failure and
  finish.
- **Rule (NVLink and driver):** a rank dies if it has an NVLink write to the victim in
  flight at the drop, or issues one after it.
- **What this means for nixl_ep:**
  - If the drop comes before or during a rank's send part, the rank dies (the send part =
    the rank sends its tokens, and at the end a flag, to the victim).
  - Survival case: if the rank has already sent to the victim, but the victim's data never
    reached it, it survives. Its receive times out, the FT mechanism masks the victim, and
    the rank goes on without it.
  - Why: the send part (dispatch and combine) always ends by writing a flag to every
    unmasked rank, even with 0 tokens, so every send writes to the victim. The receive
    part only reads local memory.
- **So in the victim's NVLink domain, a rank survives only if the drop comes after its send part
  finished and before the victim's data reached it. This case is narrow.**

Setup: 201 = ranks 0-3, 200 = ranks 4-7, 4 GPUs per node. In phase 1, link 0 of 200 GPU 2
(rank 4, the victim) is forced down; the error is fatal, so the driver takes all 18 links of
that GPU down. Ranks 5-7 are the victim's NVLink neighbours.
All runs and numbers: `nvlink_fault_2node_results.md`.

"Send" and "receive" here are the nixl_ep kernels' send and receive parts (dispatch and
combine), not NVLink terms.

## Main results

Same method in all three runs below (send and receive parts launched separately); the only
difference is when ranks 5-7 run their send part relative to the link drop:

| Run | Ranks 5-7 send to rank 4 | Ranks 5-7 | Rank 4 | Ranks 0-3 on 201 |
|---|---|---|---|---|
| `inject-after_send-1` | Finished before the drop (measured on the GPU) | Survived | Died | Finished |
| `inject-during_send-2` | Drop ~8 ms into a ~15 ms send | Died | Died | Finished |
| `inject-before_send-1` | Started after the drop | Died | Died | Finished |

Two earlier runs with a different method (fault at the first full dispatch of phase 1) agree:
`inject-tnvlink-1` (5-7 start after the drop: died) and `inject-trdma-1` (5-7 start ~30 ms
before the drop: survived). The same three split runs on one node (200, 4 ranks, victim
rank 0, `/swgwork/lishapira/nixl_ep_b200_1node/`) gave the same result.

Three more runs, each answering one question:

| Run | What is held | Ranks 5-7 | Rank 4 | Ranks 0-3 on 201 |
|---|---|---|---|---|
| `inject-after_recv-2` | 4-7 receive the victim's dispatch 5.0 data, then wait; the drop comes; they go on to dispatch 5.1 | Died (at dispatch 5.1) | Died | Finished |
| `inject-after_victim_send-1` | Only the victim: it injects after its own send part finishes, and holds its next steps | Survived | Died | Finished |
| `inject-before_all_send-1` | All 8 ranks launch their send part after the drop | Died | Died | Finished (sent to rank 4 over RDMA after the drop) |

## Explanation

**nixl_ep code: only writes cross NVLink.** Every NVLink access to a peer is a store into the
peer's memory (token copies, count flags). Every receive polls local memory.

**How a dispatch sends** (`csrc/kernels/nixl_ep_ll.cu`). The send part writes each token to
its destination: an NVLink copy to ranks on the same node, an RDMA put to ranks on the other
node. It ends by writing a "count" flag to every destination that isn't masked (lines
257-273), even when it sent that destination 0 tokens. So every send part writes to the
victim at least once, at the very end. The receive part only reads local memory, waiting for
those flags. If a flag does not arrive within 30 s, the receiver logs `timeout for dispatch
receive ... src_rank N` and masks rank N; later sends skip it. Combine works the same way.

**What kills a process** (NVLink and driver). The writing GPU tracks each NVLink write until the
destination answers. A downed GPU never answers, so about 250 ms later the writer logs
Xid 145 `RLW_SRC_TRACK`, and the driver tears down its process's GPU channels (Xid 45). The
process dies.
- NVIDIA's Xid catalog lists exactly this effect for our lines, which are "Nonfatal" with
  status 0x8: the application crashes, but the GPU stays usable. This matches the
  neighbours' GPUs keeping 18 links with no reset needed.
- The name `RLW_SRC_TRACK` and the catalog support source-side tracking; NVIDIA does not
  document the internals. The ~250 ms is our measurement.
- Reading local memory, and RDMA, create no NVLink request to the downed GPU, so they
  cannot cause this.

**Why each case ends as it does:**
- **Send finished before the drop:** ranks 5-7 have no NVLink write open to GPU 2. Rank 4
  sends only after the drop, so its data never arrives. Ranks 5-7 wait 30 s, mask rank 4,
  never write to it again, and finish.
- **Drop during or before the send:** the count flag (at least) is written to GPU 2 after
  the drop, so ranks 5-7 get Xid 145 and die. This is why `during_send` ends like
  `before_send`.

## Conclusions and evidence

1. **The fault stays on the faulted node.** In every real-fault run, 201 had no Xid, kept 18
   links on every GPU, and ranks 0-3 detected the lost ranks and finished all phases. Their
   writes to rank 4 go over RDMA only (no NVLink between the nodes), so they have no NVLink
   write that could get stuck.
   - `before_all_send`: ranks 0-3 ran their whole send part, including the writes to rank 4,
     from +18.5 to +36.6 ms, after the drop at +16.1 ms. They survived; ranks 5-7, which
     started at the same moment over NVLink, died (Xid 145 on GPUs 3, 5, 6 at +268 ms).

2. **A rank dies if it has an NVLink write to the victim open at the drop or issues one
   after it. It survives only if its last write to the victim completed before the drop and
   it masks the victim before its next write.**
   - `after_send`: the GPU finished the send part about 4 ms before the injector call
     started, 20 ms before the drop. Xid 145 only on GPU 2; ranks 5-7 survived.
   - `during_send`: send launched at +8.2 ms and takes 15.1-15.8 ms on an idle GPU; the
     drop came at +16.2 ms. The "send part finished" line never printed. Xid 145 on GPUs
     3, 5, 6 at +266 ms puts their first stuck write at about +16 ms, at the drop rather
     than at the launch. Ranks 5-7 died.
   - `before_send`: send launched at +18.5 ms, after the drop at +16.2 ms. Xid 145 on GPUs
     3, 5, 6 at +269 ms; ranks 5-7 died.
   - Writes to the other ranks on 200 do not kill: once the survivors masked rank 4, they
     wrote to each other over NVLink for the rest of the run and finished.

3. **Second evidence: a mixed result without any hold (one node).** Sep 16, 201, 8 ranks,
   victim rank 2, the injector fired at a random point in phase 1
   (`/swgwork/lishapira/nixl_ep_nvlink_fault_final_8rank/nvlink_8rank_3phase_inject_20260916_151855`).
   - Ranks 0, 6, 7 survived: they logged `timeout for combine receive` for src 1-5, so they
     had finished their combine send and were in the receive part. Their GPUs logged no
     Xid 145, so none of their writes to GPU 2 was open at the drop. They probably finished
     sending just before the drop.
   - Ranks 1, 3, 4, 5 died: their GPUs logged Xid 145 (an NVLink write to GPU 2 that never
     completed), and their combine data never reached the survivors. So they were most
     likely in the middle of their combine send at the drop.

4. **The survival case is narrow.** The neighbours survived only when the victim's own data
   never arrived, so they timed out and masked it. If the victim's data already arrived,
   nothing masks it and the neighbour's next write to it kills the neighbour. This is why some
   or all neighbours died in 4 of the 5 runs without a hold: in the same op when the drop came
   before or during their send, and in the next op when it came after the victim's own send.
   - Sep 16 repeat (one node): only GPU 4 (rank 4) got Xid 145 at the drop. Every other rank
     received the victim's data, timed out only on rank 4, and 30 s later wrote to the victim
     in the next op: Xid 145 on all GPUs, all died.
   - Nov 2 on 201 and on 200 (one node, 8 ranks): all 8 died within ~1 s, with no timeouts.
   - Sep 16 first: 4 of 7 neighbours died (above).
   - Exception: two-node `inject-nvlink-1`, where ranks 5-7 timed out on rank 4 and survived.
   - `after_recv` (two nodes): ranks 4-7 finished receiving dispatch 5.0 at -6.4 ms, before
     the drop at +16.2 ms; released at +18.3 ms, they launched dispatch 5.1. Xid 145 on
     GPUs 3, 5, 6 at +268-269 ms puts the first stuck write at about +18 ms, the start of
     dispatch 5.1. All died at the next op.
   - `after_victim_send` (two nodes) is consistent with the same rule one op later, but its
     timing is inferred, not measured. Ranks 5-7 received the victim's dispatch 5.0 data at
     -30.3 ms, without a timeout, and went straight on to dispatch 5.1. Its ~15 ms send runs
     on the GPU after that receive, so it most likely ended at about -15 ms, before the drop
     at +16.2 ms. The victim, held
     until after the drop, never sent dispatch 5.1. Ranks 5-7 timed out on `src_rank 4` in
     the dispatch 5.1 receive, masked it, and finished; no Xid 145 on their GPUs.

5. **The victim's writes to a rank do not harm that rank.** In `after_send`, rank 4 sent to
   ranks 5-7 after the drop, and only GPU 2 logged Xid 145. Ranks 5-7 simply never got rank
   4's data, timed out on `src_rank 4`, and finished.

6. **Only NVLink writes to the victim kill; RDMA writes don't.** In `before_all_send` all 8
   ranks wrote to rank 4 after the drop: 5-7 over NVLink died, 0-3 over RDMA survived. The
   death rule (an open NVLink write kills the writer's process) is NVLink and driver
   behaviour. When a rank writes to the victim, and whether it stops in time, is nixl_ep
   behaviour (count flags, 30 s timeout, mask).

**Your hypothesis:**
- Drop before a rank sends to the victim -> the rank dies: **confirmed** (`before_send`,
  `inject-tnvlink-1`). Also true for a drop in the middle of the send (`during_send`).
- Drop after a rank sends to the victim, before it receives from it -> the rank detects the
  failure and finishes: **confirmed only when the victim's data has not arrived**
  (`after_send` measured, `inject-trdma-1` inferred). If it has arrived, the rank dies at
  the next op, unless that op's send also finished before the drop and the victim's data
  for it never arrives (`after_victim_send`).
- Drop after a rank receives from the victim -> the next communication is affected:
  **confirmed** on two nodes (`after_recv`: all neighbours died at the next op) and on one
  node (Sep 16 repeat).

## Limits

- Each two-node timing ran once, and every NIXL EP fault (two-node and one-node) was on
  GPU 2.
- Writes from ranks 5-7 to each other are shown safe only from ~30 s after the drop (once
  rank 4 is masked); in the first ~250 ms they always also wrote to rank 4.
- In `elastic.py` the next op waits on the same GPU stream behind the receive that times
  out, so the victim is always masked before the next write. Overlapping work (a second
  stream, or the next send issued before the receive ends) could write to the victim before
  the mask is set; not tested.
- We see when the send part starts and ends, not which write to rank 4 got stuck. In
  `during_send` the overlap comes from launch time + GPU time on an idle GPU, the missing
  "send part finished" line, and the Xid 145 time.
- The one-node mixed result (conclusion 3) is from a run with no timestamps; "finished just
  before" and "in the middle of" are inferred from the logs and Xids, not measured.
- In `inject-nvlink-1` (random timing) the injector call took 266 ms instead of ~16.5 ms, so
  the drop can't be placed in that run.
- Survivors stall at least 30 s (the EP receive timeout) before they drop the failed ranks.
