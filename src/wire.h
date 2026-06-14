/* dfkv wire protocol — the single place that defines the request/response frame.
 * Both prefixes start with a 1-byte protocol version so a mixed-version deploy
 * fails fast (the server rejects an unknown version) instead of mis-parsing.
 * Centralizing encode/decode here keeps the byte offsets in one spot rather than
 * scattered across the TCP and RDMA transports + the server. */
#ifndef DFKV_WIRE_H_
#define DFKV_WIRE_H_

#include <cstddef>
#include <cstdint>

#include "kv_store.h"   // Status
#include "kv_types.h"   // BlockKey
#include "net_util.h"   // net::PutU32/PutU64/GetU32/GetU64 (little-endian)

namespace dfkv {

// Wire op codes (shared by TcpTransport, RdmaTransport and the server).
// kMembers (legacy static-list query) is removed in M3 once the MDS replaces it.
// kRegister/kHeartbeat/kListMembers are the MDS membership ops (M0+): the op byte
// reuses the existing request framing; variable content rides the payload/data blob.
enum class WireOp : uint8_t {
  kCache = 1, kRange = 2, kExist = 3, kStats = 4, kMembers = 5,
  kRegister = 6, kHeartbeat = 7, kListMembers = 8
};

constexpr uint8_t kProtoVersion = 1;

// Request prefix: ver(1) op(1) id(8) index(4) size(4) offset(8) length(8) payload_len(8)
constexpr size_t kReqPrefix = 1 + 1 + 8 + 4 + 4 + 8 + 8 + 8;  // = 42
// Response prefix: ver(1) status(1) data_len(8)
constexpr size_t kRespPrefix = 1 + 1 + 8;  // = 10

inline void EncodeReq(char* p, WireOp op, const BlockKey& k, uint64_t offset,
                      uint64_t length, uint64_t payload_len) {
  p[0] = static_cast<char>(kProtoVersion);
  p[1] = static_cast<char>(op);
  net::PutU64(p + 2, k.id);
  net::PutU32(p + 10, k.index);
  net::PutU32(p + 14, k.size);
  net::PutU64(p + 18, offset);
  net::PutU64(p + 26, length);
  net::PutU64(p + 34, payload_len);
}

struct ReqFields {
  uint8_t op;
  uint64_t id;
  uint32_t index;
  uint32_t size;
  uint64_t offset;
  uint64_t length;
  uint64_t payload_len;
};

// Returns false on a protocol-version mismatch (caller drops the connection).
inline bool DecodeReq(const char* p, ReqFields* o) {
  if (static_cast<uint8_t>(p[0]) != kProtoVersion) return false;
  o->op = static_cast<uint8_t>(p[1]);
  o->id = net::GetU64(p + 2);
  o->index = net::GetU32(p + 10);
  o->size = net::GetU32(p + 14);
  o->offset = net::GetU64(p + 18);
  o->length = net::GetU64(p + 26);
  o->payload_len = net::GetU64(p + 34);
  return true;
}

inline void EncodeResp(char* p, Status st, uint64_t data_len) {
  p[0] = static_cast<char>(kProtoVersion);
  p[1] = static_cast<char>(st);
  net::PutU64(p + 2, data_len);
}

inline bool DecodeResp(const char* p, Status* st, uint64_t* data_len) {
  if (static_cast<uint8_t>(p[0]) != kProtoVersion) return false;
  *st = static_cast<Status>(static_cast<uint8_t>(p[1]));
  *data_len = net::GetU64(p + 2);
  return true;
}

}  // namespace dfkv

#endif  // DFKV_WIRE_H_
