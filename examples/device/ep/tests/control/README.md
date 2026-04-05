### Control Plane Latency Test

Measures latency of NIXL EP Buffer control plane operations:
**init**, **connect**, **disconnect**, **reconnect**, **destroy**, and a **full cycle**.

#### Single Node (8 GPUs):
```bash
python3 tests/control/control.py \
    --num-processes 8
python3 tests/control/control.py \
    --num-processes 8 \
    --mode connect
python3 tests/control/control.py \
    --num-processes 8 \
    --mode connect \
    --warmup 1 --rounds 5
```

#### Multi-Node Setup:

**Node 1** (launches 4 local workers, 8 total ranks):
```bash
python3 tests/control/control.py \
    --num-processes 4 \
    --num-ranks 8
```

**Node 2** (launches 4 additional workers):
```bash
python3 tests/control/control.py \
    --num-processes 4 \
    --num-ranks 8 \
    --tcp-server $MASTER_IP
```

### Available Modes

| Mode | What is measured |
|------|-----------------|
| `cycle` | Full cycle: init → connect → disconnect → reconnect → destroy |
| `init` | `Buffer()` + `update_memory_buffers()` |
| `connect` | `connect_ranks()` |
| `disconnect` | `disconnect_ranks()` |
| `reconnect` | `connect_ranks()` after a prior disconnect |
| `destroy` | `buffer.destroy()` |
