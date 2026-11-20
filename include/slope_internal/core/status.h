/** @file
 * @brief Status codes returned by every native function.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_CORE_STATUS_H
#define SS_CORE_STATUS_H

/**
 * Native status code.
 *
 * The numeric order is the priority of a collective reduction: agreeing on a
 * status takes the maximum over the ranks (::ss_status_allreduce_max), so an
 * internal error always wins.
 */
typedef enum {
  SS_OK = 0,                     /**< Success. */
  SS_ERROR_INVALID_ARGUMENT = 1, /**< Null or out-of-range argument. */
  SS_ERROR_OVERFLOW = 2,         /**< Integer overflow of a size or index. */
  SS_ERROR_OUT_OF_MEMORY = 3,    /**< Allocation failure. */
  SS_ERROR_NOT_FOUND = 4,        /**< Missing entity, key or file. */
  SS_ERROR_DUPLICATE = 5,        /**< Duplicate entity or key. */
  SS_ERROR_VERSION_MISMATCH = 6, /**< Unsupported protocol or file version. */
  SS_ERROR_INVALID_STATE = 7,    /**< Call in the wrong lifecycle state. */
  SS_ERROR_INTEGRITY = 8,        /**< Identity, digest or consistency failure. */
  SS_ERROR_SCHEMA = 9,           /**< Document violates its schema. */
  SS_ERROR_JSON = 10,            /**< Invalid or non-canonical JSON. */
  SS_ERROR_LIMIT = 11,           /**< A configured or built-in limit is exceeded. */
  SS_ERROR_INVALID_PATH = 12,    /**< Unsafe or malformed path. */
  SS_ERROR_HDF5 = 13,            /**< HDF5 call failed. */
  SS_ERROR_RESOURCE = 14,        /**< Memory admission or resource limit. */
  SS_ERROR_TYPE_MISMATCH = 15,   /**< Value of the wrong type or unit. */
  SS_ERROR_CANCELLED = 16,       /**< Cancelled at a safe point. */
  SS_ERROR_THREAD = 17,          /**< OpenMP or thread-level failure. */
  SS_ERROR_INJECTED = 18,        /**< Failure injected by a test hook. */
  SS_ERROR_IO = 19,              /**< File-system or stream I/O failure. */
  SS_ERROR_NUMERICAL = 20,       /**< Numerical failure (nonfinite, breakdown). */
  SS_ERROR_INTERNAL = 21         /**< Internal error, including PETSc and MPI. */
} SsStatus;

#endif
