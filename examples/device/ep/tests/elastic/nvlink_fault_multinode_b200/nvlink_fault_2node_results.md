# NVLink fault during two-node NIXL EP on B200 - runs and results

The most relevant results, the explanation and the conclusions are in
`nvlink_fault_2node_conclusion.md`. This file lists every real-fault run.

The timing-mode names `nvlink` and `rdma` are misleading: they don't choose a transport. Both
modes use NVLink inside a node and RDMA between nodes; they differ only in when ranks 5-7
send relative to the drop (see Per-run details).

Setup (all runs): adv-dev-dgxb-201 = master, ranks 0-3; adv-dev-dgxb-200 = peer, ranks 4-7.
GPUs 2,3,5,6 on each node, RDMA rails 3,4,6,7, 8192 tokens, EP receive timeout 30 s, 3 phases.
Fault runs: `nvlink_hwinject` forces link 0 of 200 GPU 2 = rank 4 (the victim) down in
phase 1; the error is fatal, so the driver takes all 18 links of that GPU down. "Neighbours" =
ranks 5-7, the other ranks on 200.

Controls, all passed: no fault (`baseline-4rank-6`: all 8 ranks finish); ranks 4-7 killed
with SIGTERM (`control-sigterm-1`: 0-3 detect it and finish); each timing mode below as a dry run
with the injector's harmless probe (`dry-*`: all 8 ranks finish, no Xid).

Each run's details are in `/swgwork/lishapira/nixl_ep_b200_2node/<run>/summary.log` and,
for timed runs, `timeline.log`. Times below are ms from the start of the injector call.

## Runs with a real NVLink fault

| Run | Timing | When ranks 5-7 sent to rank 4 | Ranks that died | Ranks that finished |
|---|---|---|---|---|
| `inject-nvlink-1` | Random (injector thread runs during the step) | Unknown | 4 | 0-3, 5-7 |
| `inject-tnvlink-1` | `nvlink`: 5-7 launch their first phase-1 dispatch after the injector returns | After the drop | 4-7 | 0-3 |
| `inject-trdma-1` | `rdma`: 5-7 launch first; rank 4 starts the injector ~10 ms later; the drop comes ~30 ms after their launch | Before the drop (inferred) | 4 | 0-3, 5-7 |
| `inject-after_send-1` | `after_send`: send part alone, finished on the GPU, then inject | Before the drop (measured) | 4 | 0-3, 5-7 |
| `inject-before_send-1` | `before_send`: inject, then send part | After the drop | 4-7 | 0-3 |
| `inject-during_send-1` | `during_send` | - | Not run: 200 waited at the sudo prompt; nothing was injected | - |
| `inject-during_send-2` | `during_send`: 5-7 launch their send part 12 ms after the victim starts the injector | During: drop 8 ms into a ~15 ms send | 4-7 | 0-3 |
| `inject-after_victim_send-1` | `after_victim_send`: nobody holds the send part; the victim injects after its own send finishes | Dispatch 5.0 and (inferred) 5.1 before the drop | 4 | 0-3, 5-7 |
| `inject-after_recv-1` | `after_recv` | - | Not run: 200 stopped in preflight (GPU 2 not yet reset); nothing was injected | - |
| `inject-after_recv-2` | `after_recv`: 4-7 receive dispatch 5.0, wait for the injector, then go on | Dispatch 5.1 right after the drop | 4-7 | 0-3 |
| `inject-before_all_send-1` | `before_all_send`: like `before_send`, and 0-3 also launch their send part after the injector returns | After the drop (0-3 too, over RDMA) | 4-7 | 0-3 |

In all real-fault runs: the injector returned NV_OK; 200 GPU 2 ended with 0 of 18 links
and "Reset required", so 200 needs a power cycle after each run; 201 had no Xid, kept 18 links
on every GPU, and its ranks 0-3 detected the lost ranks and finished all 3 phases.

## Per-run details

**About the names `nvlink` / `rdma` (`inject-tnvlink-1`, `inject-trdma-1`).** They suggest the
fault hits NVLink traffic in one and RDMA traffic in the other. It doesn't: in both, ranks
5-7 send to rank 4 over NVLink and to ranks 0-3 over RDMA, exactly as in every other run.
The only difference is the order: in `nvlink` ranks 5-7 send after the drop (and die), in
`rdma` they send before it (and survive). They are really "send after the drop" and "send
before the drop", like `before_send` and `after_send`.

**`inject-nvlink-1`** - injector call 266 ms (16.5 ms in all later runs). No timestamps
in this run, so the drop can't be placed in the step. Ranks 5-7 logged `timeout for
dispatch receive ... src_rank 4`, dropped rank 4 and finished. Xid 145 only on GPU 2.

**`inject-tnvlink-1`** - injector call 16.6 ms. Ranks 5-7 held their first phase-1
dispatch and launched it 1.9 ms after the injector returned, i.e. after the drop. All four
GPUs on 200 logged Xid 145 (then Xid 45) and ranks 4-7 died. (Kernel-log marks in this run
were ~115 ms off, fixed afterwards; the order "drop before launch" holds regardless,
because the drop happens inside the injector call.)

**`inject-trdma-1`** - ranks 5-7 launched their first phase-1 dispatch at -13.9 to -13.3 ms;
link drop (Xid 149) at +16.2 ms, about 30 ms later; rank 4 launched its dispatch at +18.2 ms.
A dispatch send part takes ~15 ms on the GPU (measured in the split runs), so 5-7 had most
likely finished sending before the drop. Xid 145 only on GPU 2. Ranks 5-7 timed out waiting
for rank 4, dropped it and finished.

**`inject-after_send-1`** - fault at the first split dispatch of phase 1 (op 5). Ranks 5-7
ran only their send part and waited for the GPU to finish it: -19.8 to -4.1 ms (GPU time
15.2-15.4 ms). Rank 4 then injected; drop at +16.2 ms. Ranks 5-7 launched their receive
part at +18.2 ms; rank 4 launched its own send at +18.2 ms, after the drop. Xid 145 only on
GPU 2 (+268 ms). Ranks 5-7 timed out waiting for rank 4, dropped it and finished.

**`inject-before_send-1`** - same method; ranks 5-7 held their send part until the
injector returned. Drop at +16.2 ms; ranks 5-7 launched their send part at +18.5 ms. Xid 145
on GPUs 2, 3, 5, 6 at +269 ms, then Xid 45; ranks 4-7 died.

**`inject-during_send-2`** - same method; ranks 5-7 launched their send part at +8.2 ms.
The send part takes 15.1-15.8 ms on the GPU (dry run), so it would end at +23.3 to +24.0 ms.
The drop at +16.2 ms landed about 8 ms into it. Xid 145 on GPUs 3, 5, 6 at +266 ms and GPU 2
at +269 ms, then Xid 45; ranks 4-7 died.

**`inject-after_victim_send-1`** - nobody held its dispatch 5.0 send part. Ranks 5-7 sent
from -59 to -43 ms and finished receiving (with the victim's data, no timeout) at -30.3 ms;
rank 4's send ended at -31.6 ms, then it started the injector; drop at +16.2 ms. Ranks 5-7
went straight on to dispatch 5.1; its ~15 ms send runs on the GPU after the receive that
ended at -30.3 ms, so it most likely ended at about -15 ms, before the drop. Rank 4
launched dispatch 5.1 only at +20.4 ms and died. Xid 145 only on GPU 2. Ranks 5-7 timed out
on `src_rank 4` in the dispatch receive, dropped it and finished.

**`inject-after_recv-2`** - ranks 4-7 finished the dispatch 5.0 receive at -6.4 to -5.6 ms
and waited; drop at +16.2 ms; released at +18.3 ms into dispatch 5.1. Xid 145 on GPUs 2, 3,
5, 6 at +268-269 ms, i.e. stuck writes from about +18 ms; ranks 4-7 died.

**`inject-before_all_send-1`** - all 8 ranks held their dispatch 5.0 send part until the
injector returned (201 learns it from the TCP store). Drop at +16.1 ms; all launched at
+18.4-18.5 ms. Ranks 0-3 finished their send part (RDMA to 4-7, NVLink among themselves) at
+35.3 to +36.6 ms and survived. Ranks 5-7 never reported their send part finished; Xid 145
on GPUs 2, 3, 5, 6 at +268-269 ms; ranks 4-7 died.

## 201 bandwidth after recovery (phase 1, dispatch + combine, per master rank)

| Runs | Ranks left | Bandwidth |
|---|---|---|
| `inject-nvlink-1`, `inject-trdma-1`, `inject-after_send-1`, `inject-after_victim_send-1` | 0-3 and 5-7 (still two nodes) | ~52 GB/s |
| `inject-tnvlink-1`, `inject-before_send-1`, `inject-during_send-2`, `inject-after_recv-2`, `inject-before_all_send-1`, `control-sigterm-1` (no fault; ranks 4-7 killed with SIGTERM) | 0-3 only (one node) | ~169 GB/s |
