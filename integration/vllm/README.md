# dfkv connector for vLLM

A direct vLLM `KVConnectorBase_V1` connector (`DfkvStoreConnector`) that stores
and loads KV cache to/from a **dfkv** cluster over GPUDirect RDMA, bypassing
LMCache. It is the dfkv analogue of vLLM's `MooncakeStoreConnector`: both
producer and consumer read/write KV to a shared pool, enabling prefix-cache
reuse across requests and instances.

The connector is pure Python (ctypes over `libdfkv.so`); there is no native
build. It talks to dfkv on **raw GPU device pointers** — the paged KV cache is
registered once via `dfkv_register_memory` (an `ibv_reg_mr` that, under
nvidia-peermem, yields a GPUDirect MR), and transfers RDMA directly to/from GPU
memory with no host bounce.

## Enable

```
--kv-transfer-config '{
  "kv_connector": "DfkvStoreConnector",
  "kv_connector_module_path": "dfkv_vllm.connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "members": "c1=<server-ip>:<rdma-port>",
    "model_hash": "<uint64>",
    "lib": "/path/to/libdfkv.so"
  }
}'
```

Also set `DFKV_RDMA=1` and `DFKV_RDMA_DEV=<rail, e.g. ib7s400p0>` in the engine
environment to select the RDMA transport and rail.

## `kv_connector_extra_config` keys

| key | meaning |
|---|---|
| `members` | dfkv member string. **The port MUST be the server's `--rdma-port`** (the RDMA bootstrap listener), not the main `--port`, when RDMA is enabled. |
| `model_hash` | uint64 namespace for keys; isolates this model's KV from others sharing the pool. A shared `model_hash` requires identical kv-cache-dtype / page-size / layout. |
| `lib` | path to `libdfkv.so` (else env `DFKV_LIB` / `$DFKV_BUILD/libdfkv.so`). |
| `batch_concurrency` | client fan-out for batch ops (default 8). |

## Gotchas (validated on hd04 H100 + IB)

- **member port = rdma-port.** Pointing at the main `--port` makes every RDMA
  `put` fail (`rc=-1`); RDMA QP bootstrap listens on `--rdma-port`.
- **GPU buffers use the batch path only.** The single `dfkv_get_auto` computes a
  CRC over the destination on the CPU and segfaults on device memory; the batch
  `dfkv_batch_get_auto` is zero-copy and GPU-safe.
- **GPUDirect needs nvidia-peermem loaded** on the GPU node (`lsmod | grep
  nvidia_peermem`).
