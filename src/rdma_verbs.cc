#include "rdma_verbs.h"

#include <cstring>

#include "net_util.h"  // PutU32/GetU32/PutU64 little-endian codec

namespace dfkv {
namespace rdma {

void SerializeQpInfo(const QpInfo& in, char out[kQpInfoBytes]) {
  std::memset(out, 0, kQpInfoBytes);
  net::PutU32(out + 0, in.qpn);
  net::PutU32(out + 4, in.psn);
  std::memcpy(out + 8, &in.lid, 2);   // host LE; both ends x86_64
  std::memcpy(out + 10, in.gid, 16);
}

QpInfo ParseQpInfo(const char in[kQpInfoBytes]) {
  QpInfo q;
  q.qpn = net::GetU32(in + 0);
  q.psn = net::GetU32(in + 4);
  std::memcpy(&q.lid, in + 8, 2);
  std::memcpy(q.gid, in + 10, 16);
  return q;
}

RcEndpoint::~RcEndpoint() { Close(); }

void RcEndpoint::Close() {
  if (qp_) { ibv_destroy_qp(qp_); qp_ = nullptr; }
  for (auto* m : smr_) if (m) ibv_dereg_mr(m);
  for (auto* m : rmr_) if (m) ibv_dereg_mr(m);
  for (auto& [addr, m] : user_mr_) if (m) ibv_dereg_mr(m);
  user_mr_.clear();
  smr_.clear(); rmr_.clear();
  for (auto* b : sbuf_) delete[] b;
  for (auto* b : rbuf_) delete[] b;
  sbuf_.clear(); rbuf_.clear();
  if (cq_) { ibv_destroy_cq(cq_); cq_ = nullptr; }
  if (chan_) { ibv_destroy_comp_channel(chan_); chan_ = nullptr; }
  if (pd_) { ibv_dealloc_pd(pd_); pd_ = nullptr; }
  if (ctx_) { ibv_close_device(ctx_); ctx_ = nullptr; }
}

bool RcEndpoint::Open(const char* dev_name, size_t cap, size_t depth, uint8_t ib_port) {
  cap_ = cap; depth_ = depth; ib_port_ = ib_port;

  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list || n == 0) { if (list) ibv_free_device_list(list); return false; }
  ibv_device* dev = nullptr;
  if (dev_name && *dev_name) {
    for (int i = 0; i < n; ++i)
      if (std::strcmp(ibv_get_device_name(list[i]), dev_name) == 0) { dev = list[i]; break; }
  } else {
    dev = list[0];
  }
  if (!dev) { ibv_free_device_list(list); return false; }
  ctx_ = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (!ctx_) return false;

  pd_ = ibv_alloc_pd(ctx_);
  if (!pd_) { Close(); return false; }
  chan_ = ibv_create_comp_channel(ctx_);
  if (!chan_) { Close(); return false; }
  int cqe = static_cast<int>(depth_ * 2 + 4);
  cq_ = ibv_create_cq(ctx_, cqe, nullptr, chan_, 0);
  if (!cq_) { Close(); return false; }
  if (ibv_req_notify_cq(cq_, 0) != 0) { Close(); return false; }

  ibv_qp_init_attr qa{};
  qa.send_cq = cq_; qa.recv_cq = cq_;
  qa.cap.max_send_wr = static_cast<uint32_t>(depth_ + 1);
  qa.cap.max_recv_wr = static_cast<uint32_t>(depth_ + 1);
  qa.cap.max_send_sge = 1; qa.cap.max_recv_sge = 2;  // recv may scatter [hdr | payload]
  qa.qp_type = IBV_QPT_RC;
  qa.sq_sig_all = 1;
  qp_ = ibv_create_qp(pd_, &qa);
  if (!qp_) { Close(); return false; }

  // QP -> INIT
  ibv_qp_attr at{};
  at.qp_state = IBV_QPS_INIT;
  at.pkey_index = 0;
  at.port_num = ib_port_;
  at.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
  if (ibv_modify_qp(qp_, &at,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    Close(); return false;
  }

  // buffers + MRs
  sbuf_.resize(depth_, nullptr); rbuf_.resize(depth_, nullptr);
  smr_.resize(depth_, nullptr); rmr_.resize(depth_, nullptr);
  for (size_t i = 0; i < depth_; ++i) {
    sbuf_[i] = new char[cap_]; rbuf_[i] = new char[cap_];
    smr_[i] = ibv_reg_mr(pd_, sbuf_[i], cap_, IBV_ACCESS_LOCAL_WRITE);
    rmr_[i] = ibv_reg_mr(pd_, rbuf_[i], cap_, IBV_ACCESS_LOCAL_WRITE);
    if (!smr_[i] || !rmr_[i]) { Close(); return false; }
  }

  // local addressing info
  ibv_port_attr pa{};
  if (ibv_query_port(ctx_, ib_port_, &pa) != 0) { Close(); return false; }
  mtu_ = pa.active_mtu;
  local_.lid = pa.lid;
  local_.qpn = qp_->qp_num;
  local_.psn = qp_->qp_num & 0xFFFFFF;  // deterministic, unique per QP
  union ibv_gid gid{};
  if (ibv_query_gid(ctx_, ib_port_, 0, &gid) == 0) std::memcpy(local_.gid, gid.raw, 16);
  return true;
}

bool RcEndpoint::Connect(const QpInfo& remote) {
  // INIT -> RTR
  ibv_qp_attr at{};
  at.qp_state = IBV_QPS_RTR;
  at.path_mtu = mtu_;
  at.dest_qp_num = remote.qpn;
  at.rq_psn = remote.psn;
  at.max_dest_rd_atomic = 1;
  at.min_rnr_timer = 12;
  at.ah_attr.port_num = ib_port_;
  at.ah_attr.sl = 0;
  if (remote.lid != 0) {  // IB: LID routing
    at.ah_attr.is_global = 0;
    at.ah_attr.dlid = remote.lid;
  } else {                // RoCE: GRH/GID routing
    at.ah_attr.is_global = 1;
    std::memcpy(at.ah_attr.grh.dgid.raw, remote.gid, 16);
    at.ah_attr.grh.sgid_index = 0;
    at.ah_attr.grh.hop_limit = 1;
  }
  if (ibv_modify_qp(qp_, &at,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
    return false;
  }

  // RTR -> RTS
  ibv_qp_attr ts{};
  ts.qp_state = IBV_QPS_RTS;
  ts.timeout = 14;
  ts.retry_cnt = 7;
  ts.rnr_retry = 7;
  ts.sq_psn = local_.psn;
  ts.max_rd_atomic = 1;
  if (ibv_modify_qp(qp_, &ts,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
        IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
    return false;
  }
  return true;
}

bool RcEndpoint::PostRecv(size_t slot) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(rbuf_[slot]);
  sge.length = static_cast<uint32_t>(cap_);
  sge.lkey = rmr_[slot]->lkey;
  ibv_recv_wr wr{}, *bad = nullptr;
  wr.wr_id = slot; wr.sg_list = &sge; wr.num_sge = 1;
  return ibv_post_recv(qp_, &wr, &bad) == 0;
}

bool RcEndpoint::PostSend(size_t slot, size_t len) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(sbuf_[slot]);
  sge.length = static_cast<uint32_t>(len);
  sge.lkey = smr_[slot]->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = slot; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND; wr.send_flags = IBV_SEND_SIGNALED;
  return ibv_post_send(qp_, &wr, &bad) == 0;
}

ibv_mr* RcEndpoint::RegisterUser(void* addr, size_t len) {
  auto key = reinterpret_cast<uintptr_t>(addr);
  auto it = user_mr_.find(key);
  if (it != user_mr_.end() && it->second->length >= len) return it->second;
  if (it != user_mr_.end()) { ibv_dereg_mr(it->second); user_mr_.erase(it); }  // grew
  ibv_mr* mr = ibv_reg_mr(pd_, addr, len, IBV_ACCESS_LOCAL_WRITE);
  if (mr) user_mr_[key] = mr;
  return mr;
}

bool RcEndpoint::PostRecvScatter(size_t slot, void* payload, size_t payload_len,
                                 ibv_mr* payload_mr, size_t hdr_bytes) {
  ibv_sge sge[2]{};
  sge[0].addr = reinterpret_cast<uintptr_t>(rbuf_[slot]);  // resp prefix + value header
  sge[0].length = static_cast<uint32_t>(hdr_bytes);
  sge[0].lkey = rmr_[slot]->lkey;
  sge[1].addr = reinterpret_cast<uintptr_t>(payload);      // payload -> caller buffer
  sge[1].length = static_cast<uint32_t>(payload_len);
  sge[1].lkey = payload_mr->lkey;
  ibv_recv_wr wr{}, *bad = nullptr;
  wr.wr_id = slot; wr.sg_list = sge; wr.num_sge = 2;
  return ibv_post_recv(qp_, &wr, &bad) == 0;
}

int RcEndpoint::WaitComp(ibv_wc* out, int max) {
  for (;;) {
    int got = ibv_poll_cq(cq_, max, out);
    if (got != 0) return got;  // >0 completions, or <0 error
    // No completion ready: arm notify, poll once more (close the race where a
    // completion arrived between the empty poll and the notify), then block.
    if (ibv_req_notify_cq(cq_, 0) != 0) return -1;
    got = ibv_poll_cq(cq_, max, out);
    if (got != 0) return got;
    ibv_cq* ev_cq = nullptr; void* ev_ctx = nullptr;
    if (ibv_get_cq_event(chan_, &ev_cq, &ev_ctx) != 0) return -1;
    ibv_ack_cq_events(ev_cq, 1);
  }
}

}  // namespace rdma
}  // namespace dfkv
