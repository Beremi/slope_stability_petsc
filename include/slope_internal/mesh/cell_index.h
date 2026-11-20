/** @file
 * @brief Stable-identity search in sorted native cell arrays.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_MESH_CELL_INDEX_H
#define SS_MESH_CELL_INDEX_H

#include "slope_internal/core/core.h"

#include <string.h>

/** Binary search for the first record with a given stable identity.
 *
 * The records are sorted by an ::SsEntityId stored at @p id_offset in each
 * record. Pure; safe inside OpenMP regions.
 *
 * @param[in] records @p count records of @p record_size bytes.
 * @param[in] count Number of records.
 * @param[in] record_size Size of one record in bytes.
 * @param[in] id_offset Byte offset of the identity in a record.
 * @param[in] target Identity to find.
 * @return Index of the first matching record, or @p count when it is absent. */
static inline size_t ss_mesh_cell_index_find(const void *records, size_t count,
                                             size_t record_size, size_t id_offset,
                                             SsEntityId target) {
  const unsigned char *bytes = (const unsigned char *)records;
  size_t left = 0U;
  size_t right = count;
  SsEntityId id;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    memcpy(&id, bytes + middle * record_size + id_offset, sizeof(id));
    if (id < target) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left == count) {
    return count;
  }
  memcpy(&id, bytes + left * record_size + id_offset, sizeof(id));
  return id == target ? left : count;
}

#endif
