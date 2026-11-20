/** @file
 * @brief Identities, error selection, memory diagnostics and stage digests.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_DIAGNOSTICS_DIAGNOSTICS_H
#define SS_DIAGNOSTICS_DIAGNOSTICS_H

#include "slope_internal/core/core.h"

#include <mpi.h>

#include <stddef.h>

#ifndef NDEBUG
#include <petscmat.h>
#endif

/** Capacity of a `sha256:<hex>` identity including the terminator. */
#define SS_ID_TEXT_CAPACITY 72U

/** Identities linked into the provenance graph of a run; all are required. */
typedef struct {
  const char *source_id;     /**< Source identity of the build. */
  const char *build_id;      /**< Build identity. */
  const char *executable_id; /**< Installed executable identity. */
  const char *problem_id;    /**< Problem package identity. */
  const char *run_id;        /**< Resolved-run identity. */
  const char *machine_id;    /**< Identity of the observed machine description. */
  const char *launcher_id;   /**< Identity of the observed launcher description. */
} SsIdentityInputs;

/** Identity of the provenance graph that links the run identities.
 * @param[in] inputs Identities to link; none may be NULL.
 * @param[out] output `sha256:` digest of the sorted `name=value` projection.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_LIMIT. */
SsStatus ss_identity_link(const SsIdentityInputs *inputs,
                          char output[SS_ID_TEXT_CAPACITY]);

/** Namespaced content identity of a text.
 * @param[in] namespace_name Namespace, for example `machine`.
 * @param[in] text Text to identify.
 * @param[out] output `sha256:` digest of `<namespace>:<text>`.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_LIMIT when the
 *   projection exceeds 1 KiB. */
SsStatus ss_identity_from_text(const char *namespace_name, const char *text,
                               char output[SS_ID_TEXT_CAPACITY]);

/** Agree on one error record across a communicator.
 *
 * Collective on @p communicator. Among the valid local records, the one with
 * the smallest ::SsErrorKey wins and is broadcast from its owner, so all
 * ranks report the same error.
 *
 * @param[in] communicator Communicator of the ranks.
 * @param[in] local Error record of the calling rank (valid or not).
 * @param[out] selected Selected record, zeroed when no rank has an error.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT on all ranks when any rank
 *   passed NULL, or ::SS_ERROR_INTERNAL. */
SsStatus ss_collective_error_select(MPI_Comm communicator, const SsErrorRecord *local,
                                    SsErrorRecord *selected);

/** Check that the caller is the master thread of any enclosing OpenMP region.
 * @return ::SS_OK, or ::SS_ERROR_THREAD on a worker thread. */
SsStatus ss_master_thread_guard(void);

/** Print a per-rank memory snapshot when `-native_memory_view` is set.
 *
 * Collective on @p communicator (PETSc synchronized output); a no-op without
 * the option. Diagnostic only.
 *
 * @param[in] communicator Communicator of the ranks.
 * @param[in] stage Name of the stage, printed in the snapshot.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_memory_view(MPI_Comm communicator, const char *stage);

/** Return free glibc heap pages to the system; a no-op on other allocators.
 * @return ::SS_OK. */
SsStatus ss_memory_trim(void);

#ifndef NDEBUG
/** Print the SHA-256 of the local rows of a matrix on standard error.
 *
 * Debug builds only; active when `SLOPE_DEBUG_STAGE_DIGEST=1`. Local and
 * non-collective.
 *
 * @param[in] stage Name of the stage.
 * @param[in] ordinal Ordinal of the matrix within the stage.
 * @param[in] matrix Matrix whose rows are hashed in PETSc row order. */
void ss_debug_stage_matrix(const char *stage, size_t ordinal, Mat matrix);
/** Print the SHA-256 of the local entries of a vector on standard error.
 *
 * Debug builds only; active when `SLOPE_DEBUG_STAGE_DIGEST=1`. Local and
 * non-collective.
 *
 * @param[in] stage Name of the stage.
 * @param[in] ordinal Ordinal of the vector within the stage.
 * @param[in] vector Vector whose local entries are hashed. */
void ss_debug_stage_vector(const char *stage, size_t ordinal, Vec vector);
#endif

#endif
