# NIXL EP In-Kernel SIGKILL Coverage Plan

Goal: prove rank 2 is killed while the GPU is inside the selected low-latency NIXL EP send/receive region, not only at a Python boundary.

Target cases:

- dispatch send
- dispatch receive
- combine send
- combine receive

Approach:

1. Add a test-only mapped host/device marker buffer in `nixl_ep::Buffer`.
2. Pass the device marker pointer into `nixl_ep_ll.cu` dispatch/combine kernels.
3. At each target region, GPU writes:
   - `entered[target] = sequence` at region entry
   - `exited[target] = sequence` at region exit
4. A CPU helper in the selected rank polls the marker.
5. When the helper sees `entered == sequence`, it immediately snapshots `exited`.
6. If `exited >= sequence`, mark `MISSED_TIMING` and do not count the run.
7. If `exited < sequence`, write durable evidence, then send `SIGKILL` to the same process.
8. Add optional test-only spin cycles after `entered` to widen the in-kernel window.

Required evidence for a counted pass:

```text
entered == sequence
exited < sequence before SIGKILL
SIGKILL sent
healthy ranks detect {2}
healthy ranks continue dispatch/combine
all non-killed ranks log done
cleanup shows no leftover GPU compute processes
post-fault baseline passes
```

Files to modify:

- `/lustre/fsw/network_research_advdev/lishapira/nixl/examples/device/ep/csrc/nixl_ep.hpp`
- `/lustre/fsw/network_research_advdev/lishapira/nixl/examples/device/ep/csrc/nixl_ep.cpp`
- `/lustre/fsw/network_research_advdev/lishapira/nixl/examples/device/ep/csrc/kernels/api.cuh`
- `/lustre/fsw/network_research_advdev/lishapira/nixl/examples/device/ep/csrc/kernels/nixl_ep_ll.cu`
- `/lustre/fsw/network_research_advdev/lishapira/nixl/examples/device/ep/nixl_ep/buffer.py`
- `/lustre/fsw/network_research_advdev/lishapira/nixl/examples/device/ep/tests/elastic/elastic.py`

Build/test note:

- Code can be implemented without an active allocation.
- Build and real validation need a fresh GPU node allocation in the container, or a batch script that runs the same container build command.
