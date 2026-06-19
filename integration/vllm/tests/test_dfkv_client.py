"""DfkvDeviceClient round-trip test against a live dfkv_server.

Requires GPU + a running dfkv_server. Env:
    DFKV_LIB, DFKV_MEMBERS (=c1=<ip>:<rdma-port>), DFKV_RDMA=1, DFKV_RDMA_DEV.
Skips if torch/CUDA or the env are unavailable.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dfkv_vllm.dfkv_client import DfkvDeviceClient  # noqa: E402

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not os.environ.get("DFKV_MEMBERS") or not torch.cuda.is_available(),
    reason="needs DFKV_MEMBERS + CUDA + a running dfkv_server",
)


def test_roundtrip_gpu_pointers():
    c = DfkvDeviceClient(
        members=os.environ["DFKV_MEMBERS"], model_hash=0x1234,
        lib_path=os.environ.get("DFKV_LIB"),
    )
    n = 1 << 20
    a = torch.arange(n, dtype=torch.uint8, device="cuda")
    b = torch.zeros(n, dtype=torch.uint8, device="cuda")
    c.register_memory(a.data_ptr(), n)
    c.register_memory(b.data_ptr(), n)

    assert c.batch_put(["k0"], [a.data_ptr()], [n]) == [0]
    hits, lens = c.batch_get(["k0"], [b.data_ptr()], [n])
    torch.cuda.synchronize()
    assert hits == [1] and lens == [n]
    assert torch.equal(a, b)
    assert c.batch_exist(["k0", "missing-key"]) == [1, 0]
    c.close()
