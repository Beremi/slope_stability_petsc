/* Stable ID projections for field nodes and coupling quadrature points.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include "slope_internal/core/sha256.h"

#include <string.h>

static void ss_solver_store_u32_be(unsigned char output[4], uint32_t value) {
  output[0] = (unsigned char)(value >> 24U);
  output[1] = (unsigned char)(value >> 16U);
  output[2] = (unsigned char)(value >> 8U);
  output[3] = (unsigned char)value;
}

static void ss_solver_store_u64_be(unsigned char output[8], uint64_t value) {
  size_t byte;
  for (byte = 0U; byte < 8U; ++byte) {
    output[byte] = (unsigned char)(value >> ((7U - byte) * 8U));
  }
}

static int ss_solver_hex_nibble(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  return -1;
}

static SsStatus ss_solver_digest_prefix(const unsigned char *payload, size_t size,
                                        SsEntityId *prefix) {
  char digest[SS_SHA256_HEX_CAPACITY];
  uint64_t value = 0U;
  if (ss_sha256_bytes(payload, size, digest) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  for (size_t index = 0U; index < 16U; ++index) {
    const int nibble = ss_solver_hex_nibble(digest[index]);
    if (nibble < 0) {
      return SS_ERROR_INTERNAL;
    }
    value = (value << 4U) | (uint64_t)(uint32_t)nibble;
  }
  *prefix = value;
  return SS_OK;
}

SsStatus ss_solver_field_node_id(uint32_t dimension, uint32_t order,
                                 SsEntityId entity_id, uint32_t entity_slot,
                                 SsEntityId *stable_id) {
  static const unsigned char domain[] = "slope-field-node-v1";
  unsigned char projection[(sizeof(domain) - 1U) + 4U + 4U + 8U + 4U];
  SsEntityId value = 0U;
  size_t offset = 0U;
  if (stable_id == NULL || (dimension != 2U && dimension != 3U) ||
      (order != 1U && order != 2U && order != 4U && order != 8U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memcpy(projection + offset, domain, sizeof(domain) - 1U);
  offset += sizeof(domain) - 1U;
  ss_solver_store_u32_be(projection + offset, dimension);
  offset += 4U;
  ss_solver_store_u32_be(projection + offset, order);
  offset += 4U;
  ss_solver_store_u64_be(projection + offset, entity_id);
  offset += 8U;
  ss_solver_store_u32_be(projection + offset, entity_slot);
  offset += 4U;
  if (offset != sizeof(projection) ||
      ss_solver_digest_prefix(projection, sizeof(projection), &value) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  *stable_id = value;
  return SS_OK;
}

SsStatus ss_solver_coupling_point_id(uint32_t dimension, SsEntityId stable_cell_id,
                                     uint32_t quadrature_index,
                                     SsEntityId *stable_point_id) {
  static const unsigned char domain[] = "slope-coupling-point-v1";
  unsigned char payload[(sizeof(domain) - 1U) + 4U + 8U + 4U];
  SsEntityId value = 0U;
  size_t offset = 0U;
  if (stable_point_id == NULL || (dimension != 2U && dimension != 3U) ||
      stable_cell_id == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memcpy(payload + offset, domain, sizeof(domain) - 1U);
  offset += sizeof(domain) - 1U;
  ss_solver_store_u32_be(payload + offset, dimension);
  offset += 4U;
  ss_solver_store_u64_be(payload + offset, stable_cell_id);
  offset += 8U;
  ss_solver_store_u32_be(payload + offset, quadrature_index);
  offset += 4U;
  if (offset != sizeof(payload) ||
      ss_solver_digest_prefix(payload, sizeof(payload), &value) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  *stable_point_id = value | (UINT64_C(1) << 63U);
  return SS_OK;
}
