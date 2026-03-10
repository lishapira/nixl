# Elastic Test: High-Level Flow and Important Parts

## High-Level Flow

```
main()
  ├─ Parse args (--plan, --num-processes, --tcp-server, etc.)
  ├─ If no --tcp-server: start TCPStore master + rank server (daemon)
  ├─ If num_processes == 1: run worker(0, args) in-process
  └─ Else: torch.multiprocessing.spawn(worker, nprocs=num_processes)

worker(torch_rank, args)
  ├─ Get (local_rank, global_rank, last_active_phase) from rank server
  ├─ Load Plan from JSON (phases = list of rank sets; negative = kill that rank)
  ├─ Exit if this rank never appears in any phase (current_phase == -1)
  ├─ Create TCPStore client, NIXL EP Buffer, update_memory_buffers(max_num_ranks, ...)
  ├─ Install SIGTERM handler (release rank, destroy buffer, exit)
  └─ Phase loop:
        ├─ If this rank is in cleanly_removed → release rank, exit loop
        ├─ connect_ranks(added_ranks)
        ├─ disconnect_ranks(cleanly_removed); sleep 5
        ├─ test_main(...)  # correctness + bandwidth for current phase
        ├─ query_mask_buffer → disconnect_ranks(newly_failed_ranks) if any
        ├─ plan.next()
        └─ If no next phase → break
  └─ buffer.destroy()
```

So: one process per worker; each gets a global rank from the rank server, follows a multi-phase plan, and in each phase connects/disconnects ranks and runs the same EP test (`test_main`).

## Big Picture: What This Test Does

- **Plan-driven phases** — A JSON plan defines phases (e.g. `[[0,1,2,3], [0,1,2,3,4,5,6,7]]`). Each phase says which ranks are active; the test adds/removes connections and optionally kills a rank (negative id) for fault tolerance.
- **Coordination** — **Rank server**: assigns (local_rank, global_rank) per process. **TCPStore**: used by NIXL EP for metadata exchange in `connect_ranks`. **Plan**: who joins, who leaves, who is killed each phase.
- **Per-phase work** — For the current set of active ranks, `test_main` runs many dispatch/combine variants (FP8 on/off, zero_copy, hooks, etc.), checks correctness (received data and combined result), then measures dispatch+combine bandwidth (and optionally Kineto).

## Important Parts Inside `elastic.py`

| Part | Role |
|------|------|
| **main()** | CLI, optional TCPStore + rank server, spawn workers. |
| **worker()** | Identity from rank server; load plan; create buffer + store; phase loop (connect/disconnect, test_main, handle failures, advance plan). |
| **Phase loop** | `get_new_ranks()` / `get_removed_ranks()` / `get_killed_ranks()`; exit if this rank removed; connect new ranks; disconnect removed; run test_main; detect newly failed ranks via `query_mask_buffer` and disconnect them; `plan.next()`. |
| **test_main()** | Build test data (x, topk_idx, topk_weights, all_topk_idx); correctness loop over many options (dispatch → optional recv hook → combine → validate received data and combined_x); bandwidth bench (dispatch+combine); optional Kineto (separate dispatch/combine timings). |
| **Plan** (`plan.py`) | JSON phases; `get_new_ranks`, `get_removed_ranks`, `get_killed_ranks`, `get_active_ranks`, `next()`. |
| **Rank server** (`rank_server.py`) | Assigns global_rank and local_rank; handles release and re-use of ranks (e.g. for multi-node join). |
| **TCPStore** (`store_group.py`) | Master and client creation; used by NIXL EP for metadata in `connect_ranks`. |

Single-node: one process starts TCPStore and rank server; all workers use it. Multi-node: first node starts server; joiners pass `--tcp-server $MASTER_IP`. Fault test: a phase with a negative rank id (e.g. `-3`) causes that rank’s process to self-kill (SIGTERM) during the first dispatch; others detect it via `query_mask_buffer` and call `disconnect_ranks`.
