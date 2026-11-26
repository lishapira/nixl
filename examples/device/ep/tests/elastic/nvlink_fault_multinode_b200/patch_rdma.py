"""Pin each local NIXL EP rank to its rail-adjacent GPUDirect RDMA device.

This patches the elastic.py shipped in the container so it stays compatible
with the container's compiled nixl_ep module. It does not add fault injection.
"""

import sys


path = "examples/device/ep/tests/elastic/elastic.py"
source = open(path).read()
marker = "# B200_TWO_NODE_RDMA_PIN"

if marker in source:
    print("patch_rdma: elastic.py already patched", flush=True)
    sys.exit(0)

gpu_anchor = '    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)\n'
device_anchor = "    torch.cuda.set_device(0)\n"
if gpu_anchor not in source or device_anchor not in source:
    print("patch_rdma: ERROR GPU/device anchor not found", flush=True)
    sys.exit(2)

gpu_selection = (
    '    _gpus = os.environ["NIXL_EP_GPU_MAP"].split(",")\n'
    '    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[local_rank % len(_gpus)]\n'
)
injection = (
    f"    {marker}\n"
    '    _nics = os.environ["NIXL_EP_NIC_MAP"].split(",")\n'
    "    _nic = _nics[local_rank % len(_nics)]\n"
    '    os.environ["UCX_NET_DEVICES"] = f"{_nic}:1,cuda0-{_nic}:1"\n'
)

source = source.replace(gpu_anchor, gpu_selection, 1)
source = source.replace(device_anchor, device_anchor + injection, 1)
open(path, "w").write(source)
print("patch_rdma: pinned each local rank to its selected GPU and rc_gda rail", flush=True)
