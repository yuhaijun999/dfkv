# dfkv 独立上线 runbook（与现网 dingo-cache 混部）

> 在 GPU 节点上**独立**起一套 dfkv KV 缓存集群供 SGLang HiCache 用，与现网
> `dingo-cache` 混部互不干扰。**对现网 dingofs/sglang 命名空间、dingo-cache 进程、
> MDS、对象存储一律不动（只读边界）**；dfkv 用独立端口/目录/进程。
> 前提：GLM-5.1 = MLA（每页 KV ≈ 2.74 MiB 单对象、跨 TP 复制、仅 tp_rank0 写）。
> 已在 hd03 InfiniBand 400G 上端到端验证（两端零拷贝，单口 GET ~93% 线速）。

---

## 0. 网络模型（关键）

dfkv 把**控制面**与**数据面**解耦：

- **控制面 = TCP**：客户端用一份静态成员表 `name=ip:port` 连到各节点的 **bootstrap TCP 端口**，只交换 ~32B 的 QP 信息（LID/GID/QPN/PSN）。这条 IP 走任意两端可达的网（如 200G 存储网或管理网 bond0）。
- **数据面 = RDMA**：按**设备名**选 RDMA 口（`DFKV_RDMA_DEV=ib7s400p0`），数据走该 IB fabric，**无需该网有 IP**。因此 **400G 计算网（无 IP）与 200G 存储网（有 IP）即便是两张互不通的 IB 网也支持**：控制面走 200G、数据面走 400G。
- TCP 回退：未设 `DFKV_RDMA` 或无 RDMA 设备时自动用 TCP 传输（控制面那条连接直接当数据通道）。

发现：**静态成员表**，不连 MDS。无副本（一致性哈希单属主，节点挂 = 该分片 miss → 重算）。

---

## 1. 构建可移植产物（同架构构建机一次）

```bash
git clone git@github.com:ketor/dfkv.git && cd dfkv
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DDFKV_WITH_RDMA=ON -DDFKV_STATIC_LIBSTDCXX=ON      # RDMA + 跨发行版可移植
cmake --build build -j
# 产物：build/dfkv_server  build/libdfkv.so  build/dfkv_smoke  build/dfkvctl  build/dfkv_bench
ldd build/dfkv_server | grep -E 'ibverbs|rdmacm'          # 确认链接了 RDMA 库
```
依赖：`libibverbs-dev librdmacm-dev`（构建期）+ 运行节点装 `rdma-core`。无 RDMA 也可去掉 `-DDFKV_WITH_RDMA` 构建纯 TCP 版。

## 2. 每节点：分发 + 缓存目录

```bash
install -m755 build/dfkv_server /usr/local/bin/dfkv_server
install -m755 build/libdfkv.so  /usr/local/lib/ && ldconfig
mkdir -p /mnt/disk1/dfkv /mnt/disk2/dfkv /mnt/disk3/dfkv   # 与现网 dingo-cache 子目录错开
```
容量隔离：`--cap`（总量，按盘均分）自带 LRU 自限；设保守值，确认 `现网用量 + dfkv cap + 预留 < 物理总量`。

## 3. 每节点：systemd unit

`/etc/systemd/system/dfkv.service`：
```ini
[Unit]
Description=dfkv KV cache node (SGLang HiCache L3)
After=network-online.target
[Service]
Type=simple
# --port = TCP(bootstrap+TCP数据) 端口; --rdma-port = RDMA bootstrap 端口; --rdma-dev = 数据面 400G 口
ExecStart=/usr/local/bin/dfkv_server \
  --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv \
  --port 28000 --rdma-port 28001 --rdma-dev ib7s400p0 --cap 6597069766656
Restart=on-failure
RestartSec=2
CPUQuota=1600%
MemoryMax=64G
Nice=5
LimitNOFILE=1048576
LimitMEMLOCK=infinity        # RDMA 需要锁页内存
[Install]
WantedBy=multi-user.target
```
```bash
systemctl daemon-reload && systemctl enable --now dfkv
journalctl -u dfkv -n 10 --no-pager   # 应见 "PORT 28000" + "RDMA listening (TCP bootstrap) on port 28001, dev=ib7s400p0"
```
> server 的 bootstrap 监听 `0.0.0.0`，靠防火墙限制在内网。优雅关闭已修（`systemctl stop` 约 1s 退出）。
> **多轨**：让客户端在多张 400G 口间分散即可（见 §5 `DFKV_RDMA_DEV` 列表）；server 会按客户端请求的设备名在同轨开 QP，无需为多轨改 server 配置（保留 `--rdma-dev` 作默认）。
> ⚠️ 同机 8×400G 多轨受 NUMA 限制（NIC 跨双 socket，单内存域）；单口已近线速，多轨叠加需 NUMA 感知，暂不必配。

## 4. 集群成员表（静态）

成员 = 所有节点的 `name=<bootstrap-ip>:<rdma-port>`（RDMA）或 `:<port>`（TCP）。bootstrap-ip 用两端可达的网（200G/bond0）：
```
n57=192.168.1.57:28001,n58=192.168.1.58:28001,...
```
增减节点 = 改成员表并重载 SGLang 端（无 MDS 动态发现）。建议 N≥4 降低单点 miss 影响。

## 5. SGLang 侧接入（免 fork，dynamic 侧载）

把 `libdfkv.so` + `python/dfkv_hicache.py` 放到 pod 可访问路径，启动 `sglang serve` 前注入环境：
```bash
export PYTHONPATH=/userdata/dfkv:$PYTHONPATH
export DFKV_LIB=/userdata/dfkv/libdfkv.so
export DFKV_RDMA=1                       # 启用 RDMA 数据面（否则 TCP）
export DFKV_RDMA_DEV=ib7s400p0           # 数据面设备；多轨用逗号列表 ib7s400p0,ib7s400p1,...
# 可选: DFKV_RDMA_DEPTH=16 (单连接 pipeline, K 个请求在途; 利于 PUT; client+server 两端都要设)
sglang serve ... \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dfkv","module_path":"dfkv_hicache","class_name":"DfkvHiCache",
    "interface_v1":1,
    "members":"n57=192.168.1.57:28001,n58=192.168.1.58:28001",
    "model_hash": 81, "page_size":64, "dtype_tag":1178092852,
    "layer_num":78, "head_num":1, "head_dim":576 }'
```
> `interface_v1:1` 触发零拷贝 `batch_set_v1/get_v1` —— GET payload 经 RDMA 散射**直落 HiCache 宿主页**（client 端零拷贝），server 端直读入发送缓冲（server 端零拷贝），两端零拷贝。
> MLA 下插件自动单对象、无 rank 后缀、`backup_skip`（仅 tp_rank0 写）。decode 共享前缀配同 members。
> 多池模型（Mamba/SWA/DeepSeek-V4）用 v2 PoolTransfer 接口（插件已实现）。

## 6. 上线顺序 + 冒烟（无需 GPU）

1. 所有节点起 `dfkv_server`（§3），确认 PORT。
2. 冒烟（任一能访问内网的机器）：
   ```bash
   dfkv_smoke --members n57=192.168.1.57:28000 --size 2752512                          # TCP
   DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 dfkv_smoke --members n57=192.168.1.57:28001 --size 2752512  # RDMA 400G
   ```
3. 端到端零拷贝校验（插件 → libdfkv → RDMA → server，验证 payload 直落缓冲）：
   ```bash
   DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 DFKV_MEMBERS='n=192.168.1.57:28001' \
     python3 tests/python/rdma_e2e_validate.py    # 期望 RESULT: ZERO-COPY RDMA E2E OK
   ```
4. 压测（可选）：`DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 dfkv_bench --members ... --size 2752512 --count 8000 --threads 64`。
5. 在**一个受控 SGLang 副本**上切 `dynamic` 后端，发共享长前缀请求看命中上涨，确认后推广。

## 7. 回滚（秒级）
- SGLang：`--hicache-storage-backend` 改回原后端（mooncake 等）重启该副本，与 dfkv 解耦。
- dfkv：`systemctl stop dfkv`；缓存可丢（KV 可重算），彻底清理删 `/mnt/diskX/dfkv`。
- 全程不影响 dingo-cache / dingofs / MDS / 对象存储。

## 8. 监控 / 边界
- `dfkvctl stat <ip:port>` 拉 Prometheus 文本（put/hit/miss/bytes/accepts/objects/used/disks）；进程存活、端口可达、`/mnt/diskX/dfkv` 用量。
- SGLang `--enable-cache-report` 的 HiCache storage hit/miss、TTFT。
- 生产只读：不改现网组件；dfkv 端口仅内网开放、无鉴权勿暴露公网。
- 线协议带 1B 版本号，混版本部署会被 server 拒（不静默错读）；升级时整集群同版本。
