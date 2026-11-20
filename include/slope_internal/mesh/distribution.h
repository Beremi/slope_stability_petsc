/** @file
 * @brief Batch routing and distributed stable-identity joins for mesh setup.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_MESH_DISTRIBUTION_H
#define SS_MESH_DISTRIBUTION_H

#include "slope_internal/core/core.h"

/** Destination rank of one routed record.
 * @param record Record to route.
 * @param communicator_size Number of ranks.
 * @param context Caller context passed to ss_mesh_route_records().
 * @return Destination rank in `[0, communicator_size)`. */
typedef int (*SsMeshRouteOwner)(const void *record, int communicator_size,
                                const void *context);

/** Mix a stable identity or tuple hash into a well-distributed shard key.
 * @param[in] key Identity or hash to mix.
 * @return Mixed 64-bit key (SplitMix64 finalizer). */
uint64_t ss_mesh_distribution_hash(uint64_t key);

/** Route fixed-size records to the rank chosen for each record.
 *
 * Collective on @p communicator. Empty inputs are allowed, and messages are
 * split so that record counts may exceed the range of an MPI `int`.
 *
 * @param[in] communicator Communicator of the exchange.
 * @param[in] records @p record_count local records.
 * @param[in] record_count Number of local records.
 * @param[in] record_size Size of one record in bytes.
 * @param[in] owner Destination of each record.
 * @param[in] context Context passed to @p owner.
 * @param[out] received Records routed to this rank; the caller releases them
 *   with free().
 * @param[out] received_count Number of received records.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, ::SS_ERROR_OVERFLOW or an
 *   allocation or MPI failure, agreed on all ranks. */
SsStatus ss_mesh_route_records(MPI_Comm communicator, const void *records,
                               size_t record_count, size_t record_size,
                               SsMeshRouteOwner owner, const void *context,
                               void **received, size_t *received_count);

/** Value attached to a stable identity. */
typedef struct {
  SsEntityId id;  /**< Stable identity. */
  uint32_t value; /**< Nonzero value assigned to @ref id. */
} SsMeshIdValue;

/** Join distributed source values to local queries by stable identity.
 *
 * Collective on @p communicator. Repeated identical assignments and repeated
 * queries are allowed; conflicting assignments and sources that no rank
 * queries fail.
 *
 * @param[in] communicator Communicator of the join.
 * @param[in] sources @p source_count local assignments with nonzero values.
 * @param[in] source_count Number of local assignments.
 * @param[in] queries @p query_count local identities to look up.
 * @param[in] query_count Number of local queries.
 * @param[out] values One value per query; zero for an unmatched query.
 * @return ::SS_OK; ::SS_ERROR_INTEGRITY for conflicting assignments,
 *   ::SS_ERROR_NOT_FOUND for an assignment that no rank queries, or another
 *   failure, agreed on all ranks. */
SsStatus ss_mesh_join_ids(MPI_Comm communicator, const SsMeshIdValue *sources,
                          size_t source_count, const SsEntityId *queries,
                          size_t query_count, uint32_t *values);

#endif
