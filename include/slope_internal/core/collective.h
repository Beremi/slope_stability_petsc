/** @file
 * @brief Collective agreement on native status codes.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_CORE_COLLECTIVE_H
#define SS_CORE_COLLECTIVE_H

#include "slope_internal/core/status.h"

#include <mpi.h>
#include <petscvec.h>

#include <stddef.h>

/** Agree on the maximum status code across a communicator.
 *
 * Collective on @p communicator. Call it with the local status before any
 * collective operation that a failing rank would skip.
 *
 * @param[in] communicator Communicator whose ranks agree.
 * @param[in] local_status Status of the calling rank.
 * @return The maximum status over all ranks, or ::SS_ERROR_INTERNAL when the
 *   reduction itself fails. */
SsStatus ss_status_allreduce_max(MPI_Comm communicator, SsStatus local_status);

/** Duplicate a vector into several destinations, agreeing on failure after
 * each duplication.
 *
 * Collective on the communicator of @p source. The caller destroys every
 * nonnull destination on every path, including failure.
 *
 * @param[in] source Vector whose layout is duplicated.
 * @param[out] destinations @p count pointers to the new vectors.
 * @param[in] count Number of duplicates.
 * @return ::SS_OK, or ::SS_ERROR_OUT_OF_MEMORY on every rank when one rank
 *   failed. */
SsStatus ss_vec_duplicate_collective(Vec source, Vec *const *destinations,
                                     size_t count);

#endif
