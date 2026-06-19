# DfkvStoreConnector — direct vLLM KV connector for dfkv

**Date:** 2026-06-18
**Status:** Design approved, pending spec review
**Branch:** `feat/vllm_store_connector`

## 1. Problem & goal

DeepSeek-V4-Pro inference on vLLM 0.23.0 cannot use the LMCache path: the
in-process `LMCacheConnectorV1` does not work for this model, and even the
multi-process `LMCacheMPConnector` workaround is not a viable production answer
for V4-Pro. The existing dfkv↔vLLM integration (`integration/lmcache/`,
`DfkvConnectorAdapter`) sits *underneath* LMCache, so when LMCache is unusable,
dfkv has no bridge to vLLM.

**Goal:** give dfkv a *direct* vLLM KV connector — implementing vLLM's native
`KVConnectorBase_V1` interface — that bypasses LMCache entirely, exactly as
`MooncakeStoreConnector` does. dfkv becomes a drop-in alternative in the same
`--kv-transfer-config` slot Mooncake occupies in production today.

**Deliverable:** a production-grade connector, developed end-to-end on an
isolated GPU node (hd04-gpu1-0065) until it runs on real hardware, validated on
DeepSeek-V4-Flash.

### Non-goals / out of scope

- Forking vLLM. The connector is an out-of-tree plugin.
- V4-Pro multi-node validation. V4-Pro needs 3 nodes (h100x3) and cannot run on
  a single node. The connector is MLA-correct and validated on V4-Flash (same
  MLA architecture); final V4-Pro sign-off is a follow-on, coordinated with the
  team on a multi-node environment.
- Changing the existing LMCache path (`integration/lmcache/`). It stays as-is;
  this work is orthogonal.

## 2. Background: how vLLM KV connectors work (v0.23.0)

vLLM has a native KV-transfer framework. LMCache, Mooncake, and NIXL are all
*implementations* of one interface, `KVConnectorBase_V1`
(`vllm/distributed/kv_transfer/kv_connector/v1/base.py`). A connector is
selected via `--kv-transfer-config`, and out-of-tree connectors are loaded via
`kv_connector_module_path` (proven: production `LMCacheMPConnector` uses it).

The reference for this design is the production-deployed `MooncakeStoreConnector`
(`.../kv_connector/v1/mooncake/store/`), a store-backed prefix cache. A full
copy of its source and a seam-by-seam mapping is archived at
`01-工作/20260618-210410-dfkv对接vllm-MooncakeStore模板/` in the ai_david
knowledge base (outside this repo).

### Abstract methods the connector must implement

- **Scheduler role:** `get_num_new_matched_tokens`, `update_state_after_alloc`,
  `build_connector_meta`, `request_finished`.
- **Worker role:** `register_kv_caches`, `start_load_kv`, `wait_for_layer_load`,
  `save_kv_layer`, `wait_for_save`, `get_finished`.

**Key design borrowed from Mooncake:** `start_load_kv` / `save_kv_layer` /
`wait_for_save` / `wait_for_layer_load` are **no-ops**. All real I/O is issued
in `get_finished()` per whole request, for compute/I-O overlap. No layerwise
transfer.

## 3. Architecture & packaging

- **Form:** out-of-tree vLLM KV connector plugin, code lives in the dfkv repo
  under a new `integration/vllm/`, symmetric to `integration/lmcache/`.
- **Enable:**
  ```
  --kv-transfer-config '{"kv_connector":"DfkvStoreConnector",
    "kv_connector_module_path":"dfkv_vllm.connector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{ ...dfkv MDS url / group / lib path... }}'
  ```
- **Branch:** `feat/vllm_store_connector` off dfkv `main` (NOT the colleague's
  `feat/lmcache_connector`, which is the LMCache adapter path).
- **Reuse:** the dfkv core (native verbs transport + C-ABI + `RegisterUser`) and
  the ctypes pieces from `integration/lmcache/` (`native_client`, `key_mapper`,
  `exists_cache`). One dfkv core + two thin adapters (lmcache / vllm).
- **Dev boundary vs colleague:** developed independently on 0065, reusing
  environment artifacts (same vLLM 0.23.0 image, `/userdata/dsv4/` weights,
  dfkv_server deployment know-how) but separate code/branch/node — no
  interference with the LMCache experiment running on 0052.

## 4. Components (unit boundaries)

Mirrors Mooncake's `store/` three-part structure. Each unit has one purpose, a
defined interface, and is independently testable.

| Unit | Purpose | Interface (who calls) | Depends on |
|---|---|---|---|
| `connector.py` — `DfkvStoreConnector(KVConnectorBase_V1)` | Thin shell; dispatch by role; implement all base.py abstract methods | vLLM instantiates it | scheduler + worker |
| `scheduler.py` — `DfkvStoreScheduler` | Hit decision: `get_num_new_matched_tokens` / `update_state_after_alloc` / `build_connector_meta` / `request_finished` | connector | `LookupKeyClient`, coordinator |
| `worker.py` — `DfkvStoreWorker` | MR registration + real put/get + async send/recv threads + rank-0 `LookupKeyServer` | connector | `dfkv_client`, coordinator, token_db |
| `dfkv_client.py` **(new core adapter)** | Translate Mooncake store semantics → dfkv: `register_buffer→RegisterUser`, `batch_put_from_multi_buffers→dfkv_batch_put`, `batch_get_into_multi_buffers→dfkv_batch_get_auto`, `batch_is_exist→exists` | worker | `integration/lmcache` `native_client` (ctypes→libdfkv.so) |
| `coordinator.py` / `data.py` / `protocol.py` | key database (ChunkedTokenDatabase), metadata dataclasses, in-process RPC | scheduler/worker | `key_mapper` |
| `metrics.py` / `stats.py` | connector Prometheus metrics, KV-events aggregation | connector | dfkv `/metrics` (existing) |

**Isolation point = `dfkv_client.py`:** the *only* unit that touches dfkv. It
translates "Mooncake store semantics" into "dfkv semantics"; worker/scheduler
never touch libdfkv directly. Swapping the storage backend touches only this
file, and it can be unit-tested against a live dfkv_server without vLLM.

**Only genuinely new logic** = `dfkv_client.py` + the `Mooncake*→Dfkv*`
substitutions. `coordinator`/`data`/`protocol`/`scheduler` skeletons and the
async-thread model are copied near-verbatim (MLA single-segment and rank-less
keys already match dfkv conventions).

## 5. Data flow & timing

### A. One-time registration (`register_kv_caches`, GPUDirect)
```
kv_caches dict → per cache: cache.untyped_storage().data_ptr() (GPU addr) + nbytes
  → dfkv_client.register_buffer(gpu_ptr, len) = RcEndpoint::RegisterUser → ibv_reg_mr(GPU ptr)
    → peermem loaded → GPUDirect MR (whole KV region registered once; later refs by offset)
  → stride probe: MLA blocks-first → single segment (this path);
                  FlashAttn K/V-first → split segments
  → start async send/recv threads + rank-0 LookupKeyServer
```
dfkv's two-sided send/recv uses the user buffer as a local SGE, so
`IBV_ACCESS_LOCAL_WRITE` (dfkv's current flag) suffices — no access-flag change.

### B. Hit decision (scheduler, before prefill)
```
new request → scheduler.get_num_new_matched_tokens
  → compute dfkv keys from block hashes (coordinator/token_db)
  → LookupKeyClient → worker rank-0 LookupKeyServer → dfkv_client.exists(keys)
  → return "how many tokens reusable from remote" → vLLM skips that prefill span
```

### C. Actual transfer (`get_finished`, per-request async, overlaps compute)
```
start_load_kv / save_kv_layer / wait_for_save = all no-ops
get_finished():
  load (hit):   recv thread dfkv_batch_get_auto → RDMA scatter directly into
                pre-registered GPU blocks (zero-copy)
  save (after prefill): send thread waits CUDA event (KV computed) → dfkv_batch_put
                from GPU blocks directly
  reap completed req_ids from the previous round, report to vLLM
```

### D. Error handling (production invariant)
- Read miss / read failure → mark `block_ids_with_load_errors` → vLLM recomputes
  that span normally (not fatal; this is cache semantics).
- Write failure → count + drop that request's save (never blocks inference);
  dfkv's existing peer-health marks the bad peer and short-circuits.
- dfkv server node down → consistent-hash miss → recompute (dfkv has no replicas;
  the connector just treats it as a normal miss).
- **Invariant:** any dfkv-side failure only causes "fewer hits = more compute".
  It must NEVER 503 or hang inference (cf. the SGLang "never `wait_complete`"
  lesson).

### E. MLA specifics
`use_mla → num_kv_head=1 / put_step=tp_size / head_or_tp_rank=tp_rank//put_step`,
keys carry no per-head-rank suffix — matches dfkv's existing MLA convention; the
stride probe takes the single-segment branch.

## 6. Phased implementation & verification (on 0065)

Production-grade is the destination, reached incrementally with a real-hardware
gate at each phase.

**P0 — GPUDirect de-risk (first, ~half day).** A few-dozen-line standalone test:
`cudaMalloc` a buffer → dfkv `RegisterUser(gpu_ptr,len)` → one put+get against a
dfkv_server → verify bytes.
*Gate:* `ibv_reg_mr` on a GPU pointer succeeds on 0065 (peermem) and RDMA
read/write is byte-correct. If it fails, debug in place (access flags / peermem
state / dmabuf fallback) before going further.

**P1 — minimal correct connector.** `dfkv_client.py` three primitives +
`register_kv_caches` + scheduler hit + synchronous put/get (not yet async). Run
vLLM 0.23.0 + DeepSeek-V4-Flash + `DfkvStoreConnector`.
*Gate:* `bytes_match` byte-correct + cold→warm dfkv-L3 hit shows speedup + 0
failures.

**P2 — performance structure (async overlap).** Add send/recv async threads +
`get_finished` per-request async + rank-0 LookupKeyServer + GPUDirect zero-copy
path.
*Gate:* 50k-input warm throughput/TTFT in a reasonable band (order of the
colleague's LMCache path, ~90–120k tok/s).

**P3 — production hardening.** kv-events aggregation, full Prometheus metrics,
error retry/degradation, (if needed) disk-offload. Final validation on
DeepSeek-V4-Flash.
*Gate:* capability parity with `MooncakeStoreConnector` + V4-Flash real-hardware
0 failures + same-workload head-to-head bench (nocache / mooncake / dfkv).

**Cross-cutting constraints:** production is read-only throughout — never touch
lm-dsv4 / node 0052. On 0065, launch via on-node `ctr`/`nerdctl` (the node
carries a `nodepool=fault` taint, off normal scheduling). dfkv changes follow
upstream-repo discipline (no internal cluster names / IPs / product names in
code, comments, or docs).

## 7. Risks & open items

1. **P0 reg_mr on GPU pointer (biggest unknown).** Whether dfkv's plain
   `ibv_reg_mr` accepts a GPU device pointer under peermem on 0065. P0 exists
   solely to settle this before any connector code.
2. **0065 is single-rail IB.** Fine for functional bring-up, but final
   performance numbers must be re-measured on a multi-rail node to represent
   production.
3. **V4-Flash weights reachable on 0065.** `/userdata` is a hostPath; confirm
   `/userdata/dsv4/DeepSeek-V4-Flash` exists on 0065 or is served via shared
   storage.
4. **vLLM 0.23.0 is an openclaw fork.** Treat the in-image `base.py` as the
   source of truth for the `KVConnectorBase_V1` signature, not upstream docs.
5. **V4-Pro carry-over.** Same MLA architecture is expected to make the
   Flash-validated connector apply to V4-Pro, but this is an assumption until
   multi-node V4-Pro validation (follow-on, out of scope here).

## 8. References

- Mooncake template + seam-by-seam mapping (ai_david KB):
  `01-工作/20260618-210410-dfkv对接vllm-MooncakeStore模板/`
  (`DfkvStoreConnector-施工图.md` + full `mooncake/store/*` source).
- Production reference config (lm-dsv4): vLLM 0.23.0,
  `--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'`,
  `WITH_NVIDIA_PEERMEM=1`, mooncake rdma device_name ib7s400p0/p1.
- Existing dfkv LMCache adapter (reuse source): `integration/lmcache/`.
</content>
