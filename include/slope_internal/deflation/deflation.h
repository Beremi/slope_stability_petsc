/** @file
 * @brief Deflation spaces recycled across continuation steps.
 *
 * A deflation store holds a left basis `W` and a right basis `Z`. With
 * `E = W^T A Z`, the coarse correction is `Q = Z E^{-1} W^T`, the right
 * projector `P_R = I - Q A` and the left projector `P_L = I - A Q`
 * (`P_L A = A P_R`). Candidates added during an analysis transaction are
 * provisional; ss_deflation_commit() makes them persistent at an accepted
 * state and ss_deflation_rollback() discards them with a rejected attempt.
 *
 * Unless stated otherwise, functions that take a store are collective on the
 * store's communicator and return a status agreed on all ranks. A store is
 * used by one thread at a time (the master thread).
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_DEFLATION_DEFLATION_H
#define SS_DEFLATION_DEFLATION_H

#include "slope_internal/core/sha256.h"
#include "slope_internal/core/status.h"

#include <petscmat.h>
#include <petscvec.h>

#include <stddef.h>
#include <stdint.h>

/** Capacity of a candidate provenance string. */
#define SS_DEFLATION_PROVENANCE_CAPACITY 96U

/** Why a candidate was not admitted. */
typedef enum {
  SS_DEFLATION_DROP_NONE = 0,            /**< Admitted. */
  SS_DEFLATION_DROP_NONFINITE = 1,       /**< Nonfinite entries. */
  SS_DEFLATION_DROP_NEGLIGIBLE = 2,      /**< Norm below the tolerance. */
  SS_DEFLATION_DROP_DEPENDENT = 3,       /**< Dependent on the current basis. */
  SS_DEFLATION_DROP_ILL_CONDITIONED = 4, /**< Coarse matrix too ill-conditioned. */
  SS_DEFLATION_DROP_CAPACITY = 5         /**< Basis is full. */
} SsDeflationDropReason;

/** Which vectors become deflation candidates. */
typedef enum {
  /** Directions between accepted continuation states. */
  SS_DEFLATION_CANDIDATE_ACCEPTED_DIRECTIONS = 0,
  /** Accepted state vectors themselves. */
  SS_DEFLATION_CANDIDATE_ACCEPTED_STATE_VECTORS = 1,
  /** Experimental upper-bound mode: accepted states, Newton solutions and
   * every numerically independent outer FGMRES correction direction. */
  SS_DEFLATION_CANDIDATE_ACCEPTED_STATES_AND_FULL_KRYLOV = 2,
  /** Accepted continuation states plus one final correction from every
   * completed linear system; compression selects the rolling system window. */
  SS_DEFLATION_CANDIDATE_ACCEPTED_STATES_AND_SYSTEM_SOLUTIONS = 3
} SsDeflationCandidatePolicy;

/** Projector formulation. */
typedef enum {
  /** General oblique left and right projectors with `E = W^T A Z`. */
  SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT = 0,
  /** Reference A-orthogonal projector with a coarse initial guess. */
  SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL = 1
} SsDeflationProjector;

/** Compression of the basis when it is committed. */
typedef enum {
  SS_DEFLATION_COMPRESSION_NONE = 0, /**< Keep every admitted column. */
  /** Best rank-k approximation of the normalized Krylov and linear-solution
   * snapshots in the Euclidean norm. */
  SS_DEFLATION_COMPRESSION_SNAPSHOT_POD = 1,
  /** Retain the vectors in the snapshot span with the smallest `||A z||_2`;
   * for the symmetric tangent these approximate its difficult near-null
   * directions without an inverse application. */
  SS_DEFLATION_COMPRESSION_OPERATOR_SVD = 2,
  /** POD of every source, including accepted continuation states; the
   * compression dimension is the total active cap. */
  SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL = 3,
  /** Preserve accepted continuation states and retain only the newest
   * `compression_dimension` completed-system corrections. */
  SS_DEFLATION_COMPRESSION_SYSTEM_SOLUTION_WINDOW = 4
} SsDeflationCompression;

/** Numerically relevant provenance retained across accepted-boundary
 * restarts. Compression distinguishes protected accepted states from
 * historical POD modes; all other sources follow the ordinary path. */
typedef enum {
  SS_DEFLATION_PERSISTENT_SOURCE_OTHER = 0,          /**< Any other source. */
  SS_DEFLATION_PERSISTENT_SOURCE_ACCEPTED_STATE = 1, /**< Accepted state. */
  SS_DEFLATION_PERSISTENT_SOURCE_SNAPSHOT_POD = 2    /**< Retained POD mode. */
} SsDeflationPersistentSourceKind;

/** Admission, conditioning and compression policy of a store. */
typedef struct {
  size_t dimension_cap;                /**< Largest number of active columns. */
  PetscReal norm_tolerance;            /**< Relative norm below which a candidate
                                            is negligible. */
  PetscReal pivot_tolerance;           /**< Smallest relative pivot of `E`. */
  PetscReal condition_limit;           /**< Largest condition estimate of `E`. */
  PetscReal invariant_tolerance;       /**< Bound of the projector identities. */
  uint32_t reorthogonalization_passes; /**< Gram-Schmidt passes at admission. */
  SsDeflationProjector projector;      /**< Projector formulation. */
  SsDeflationCompression compression;  /**< Compression at commit. */
  size_t compression_dimension;        /**< Columns retained by compression. */
  /** Exponential weight applied to the prior compressed POD spectrum before
   * admitting the newest snapshots. One is cumulative POD; values below one
   * gradually discount stale tangent information. */
  PetscReal compression_history_decay;
} SsDeflationPolicy;

/** Outcome of one candidate admission. */
typedef struct {
  int admitted;                      /**< Nonzero when the candidate was added. */
  SsDeflationDropReason drop_reason; /**< Reason when not admitted. */
  PetscReal left_norm;               /**< Norm of the left candidate. */
  PetscReal right_norm;              /**< Norm of the right candidate. */
  PetscReal relative_pivot;          /**< Relative pivot after orthogonalization. */
  PetscReal condition_estimate;      /**< Condition estimate after admission. */
  size_t persistent_count;           /**< Persistent columns. */
  size_t provisional_count;          /**< Provisional columns. */
  size_t active_count;               /**< Active columns. */
  uint64_t global_reductions;        /**< Reductions spent on the admission. */
} SsDeflationAdmissionRecord;

/** Counters and conditioning of a store. */
typedef struct {
  size_t persistent_count;                /**< Committed columns. */
  size_t provisional_count;               /**< Columns of the open transaction. */
  size_t active_count;                    /**< Columns used by the projectors. */
  size_t cached_az_count;                 /**< Cached `A Z` columns. */
  PetscObjectState operator_state;        /**< PETSc state of the refreshed operator. */
  uint64_t operator_generation;           /**< Generation of the refreshed operator. */
  uint64_t accepted_generation;           /**< Generation of the last commit. */
  uint64_t transaction_generation;        /**< Generation of the open transaction. */
  uint64_t global_reductions;             /**< Global reductions so far. */
  uint64_t refresh_count;                 /**< Refreshes so far. */
  uint64_t q_apply_count;                 /**< Coarse-correction applications. */
  uint64_t projected_apply_count;         /**< Projected-operator applications. */
  uint64_t ill_conditioned_drop_count;    /**< Ill-conditioned drops. */
  SsDeflationDropReason last_drop_reason; /**< Most recent drop reason. */
  int invariants_validated;               /**< Nonzero when identities were checked. */
  PetscReal minimum_relative_pivot;       /**< Smallest relative pivot of `E`. */
  PetscReal condition_estimate;           /**< Condition estimate of `E`. */
  PetscReal maximum_invariant_error;      /**< Largest projector-identity error. */
  uint64_t compression_count;             /**< Compressions so far. */
  size_t last_compression_input_count;    /**< Columns entering the last compression. */
  size_t last_compression_protected_count; /**< Protected columns of it. */
  size_t last_compression_retained_count;  /**< Columns it retained. */
  /** Fraction of snapshot energy retained by the last compression. */
  PetscReal last_compression_information_fraction;
} SsDeflationDiagnostics;

/** Opaque deflation store; see ss_deflation_create(). */
typedef struct SsDeflation SsDeflation;

/** Create an empty store; collective.
 * @param[in] communicator Communicator of the vectors; duplicated.
 * @param[in] policy Admission and compression policy; copied.
 * @param[out] result New store, released with ss_deflation_destroy().
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, ::SS_ERROR_OUT_OF_MEMORY or
 *   ::SS_ERROR_INTERNAL, agreed on all ranks. */
SsStatus ss_deflation_create(MPI_Comm communicator, const SsDeflationPolicy *policy,
                             SsDeflation **result);
/** Estimate the memory of a policy before a store exists; local.
 *
 * The estimate is the peak aggregate vector payload and the per-rank dense
 * scratch at the configured basis cap; PETSc object and allocator overhead
 * are excluded.
 *
 * @param[in] policy Policy.
 * @param[in] global_rows Global vector length.
 * @param[out] vector_bytes Aggregate vector payload over all ranks.
 * @param[out] dense_bytes_per_rank Dense coarse scratch per rank.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_OVERFLOW. */
SsStatus ss_deflation_policy_memory_estimate(const SsDeflationPolicy *policy,
                                             PetscInt global_rows, size_t *vector_bytes,
                                             size_t *dense_bytes_per_rank);

/** Estimate the memory of an existing store; local.
 * @param[in] deflation Store.
 * @param[in] global_rows Global vector length.
 * @param[out] vector_bytes Aggregate vector payload over all ranks.
 * @param[out] dense_bytes_per_rank Dense coarse scratch per rank.
 * @return As ss_deflation_policy_memory_estimate(). */
SsStatus ss_deflation_memory_estimate(const SsDeflation *deflation,
                                      PetscInt global_rows, size_t *vector_bytes,
                                      size_t *dense_bytes_per_rank);

/** Destroy a store and every vector it owns; collective.
 * @param[in,out] deflation Store pointer, set to NULL; `*deflation` may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_deflation_destroy(SsDeflation **deflation);

/** Open an analysis transaction; later candidates are provisional.
 * @param[in,out] deflation Store without an open transaction.
 * @param[in] transaction_generation Generation of the attempt.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE, ::SS_ERROR_INVALID_ARGUMENT,
 *   ::SS_ERROR_OUT_OF_MEMORY or ::SS_ERROR_INTERNAL. */
SsStatus ss_deflation_begin(SsDeflation *deflation, uint64_t transaction_generation);
/** Admit a candidate pair with unit source weight.
 * @param[in,out] deflation Store with an open transaction.
 * @param[in] left Left candidate (`W` column); copied.
 * @param[in] right Right candidate (`Z` column); copied.
 * @param[in] provenance Short description of the source.
 * @param[out] record Admission outcome.
 * @return As ss_deflation_add_weighted_candidate(). */
SsStatus ss_deflation_add_candidate(SsDeflation *deflation, Vec left, Vec right,
                                    const char *provenance,
                                    SsDeflationAdmissionRecord *record);
/** Admit a candidate pair after normalization and orthogonalization.
 *
 * A dropped candidate is not an error; @p record gives the reason.
 *
 * @param[in,out] deflation Store with an open transaction.
 * @param[in] left Left candidate; copied.
 * @param[in] right Right candidate; copied.
 * @param[in] source_weight Weight of the source in snapshot compression.
 * @param[in] provenance Short description of the source.
 * @param[out] record Admission outcome.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE without a transaction,
 *   ::SS_ERROR_TYPE_MISMATCH for an incompatible layout, ::SS_ERROR_LIMIT,
 *   ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_deflation_add_weighted_candidate(SsDeflation *deflation, Vec left,
                                             Vec right, PetscReal source_weight,
                                             const char *provenance,
                                             SsDeflationAdmissionRecord *record);
/** Drop the provisional columns of the open transaction; local bookkeeping.
 * @param[in,out] deflation Store.
 * @return ::SS_OK or ::SS_ERROR_INVALID_STATE. */
SsStatus ss_deflation_discard_provisional(SsDeflation *deflation);
/** Recompute `A Z` and factorize `E` for a new operator generation, then
 * validate the projector identities.
 * @param[in,out] deflation Store.
 * @param[in] operator_handle Current operator `A`.
 * @param[in] operator_generation Generation of @p operator_handle.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY when the identities fail,
 *   ::SS_ERROR_OVERFLOW, ::SS_ERROR_OUT_OF_MEMORY or ::SS_ERROR_INTERNAL. */
SsStatus ss_deflation_refresh(SsDeflation *deflation, Mat operator_handle,
                              uint64_t operator_generation);
/** Close the transaction at an accepted state: make the provisional columns
 * persistent and apply the compression.
 * @param[in,out] deflation Store with an open transaction.
 * @param[in] accepted_generation Generation of the accepted state.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_OUT_OF_MEMORY. */
SsStatus ss_deflation_commit(SsDeflation *deflation, uint64_t accepted_generation);
/** Close the transaction of a rejected attempt and restore the basis that
 * existed when it was opened.
 * @param[in,out] deflation Store with an open transaction.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_deflation_rollback(SsDeflation *deflation);

/** Apply the coarse correction `output = Q input`.
 * @param[in,out] deflation Refreshed store.
 * @param[in] input Input vector.
 * @param[out] output Output vector.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_deflation_apply_q(SsDeflation *deflation, Vec input, Vec output);
/** Apply the right projector with the cached `A Z` (A-orthogonal projector).
 * @param[in,out] deflation Refreshed store.
 * @param[in] input Input vector.
 * @param[out] output Output vector.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_deflation_apply_cached_right(SsDeflation *deflation, Vec input, Vec output);
/** A-orthogonal projector: set `solution = Q rhs` and
 * `projected_rhs = rhs - A solution` with the cached `A Z`.
 * @param[in,out] deflation Refreshed store with the A-orthogonal projector.
 * @param[in] rhs Right-hand side.
 * @param[out] solution Coarse initial guess.
 * @param[out] projected_rhs Residual of the initial guess.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_deflation_coarse_initial_guess_and_project_rhs(SsDeflation *deflation,
                                                           Vec rhs, Vec solution,
                                                           Vec projected_rhs);
/** Apply the right projector `output = P_R input`.
 * @param[in,out] deflation Refreshed store.
 * @param[in] operator_handle Refreshed operator.
 * @param[in] input Input vector.
 * @param[out] output Output vector.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_deflation_apply_right(SsDeflation *deflation, Mat operator_handle,
                                  Vec input, Vec output);
/** Apply the left projector `output = P_L input`.
 * @param[in,out] deflation Refreshed store.
 * @param[in] operator_handle Refreshed operator.
 * @param[in] input Input vector.
 * @param[out] output Output vector.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_deflation_apply_left(SsDeflation *deflation, Mat operator_handle, Vec input,
                                 Vec output);
/** Apply the projected operator `output = A P_R input`.
 * @param[in,out] deflation Refreshed store.
 * @param[in] operator_handle Refreshed operator.
 * @param[in] input Input vector.
 * @param[out] output Output vector.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_deflation_apply_projected_operator(SsDeflation *deflation,
                                               Mat operator_handle, Vec input,
                                               Vec output);
/** Apply the prepared right projector inside a preconditioner without status
 * agreement.
 *
 * Collective only through its vector reductions; a rank-local failure is
 * returned locally and the caller must make it visible to the Krylov method.
 *
 * @param[in,out] deflation Refreshed store.
 * @param[in] input Input vector.
 * @param[out] output Output vector.
 * @return ::SS_OK or ::SS_ERROR_INTERNAL on this rank. */
SsStatus ss_deflation_apply_cached_right_local(SsDeflation *deflation, Vec input,
                                               Vec output);
/** Apply the prepared projected operator with only its numerical reduction.
 *
 * The caller must flag its output on a rank-local failure.
 *
 * @param[in,out] deflation Refreshed store.
 * @param[in] operator_handle Refreshed operator.
 * @param[in] input Input vector.
 * @param[out] output Output vector.
 * @return ::SS_OK or a failure on this rank. */
SsStatus ss_deflation_apply_projected_operator_local(SsDeflation *deflation,
                                                     Mat operator_handle, Vec input,
                                                     Vec output);
/** Project a right-hand side, `projected_rhs = P_L rhs`.
 * @param[in,out] deflation Refreshed store.
 * @param[in] operator_handle Refreshed operator.
 * @param[in] rhs Right-hand side.
 * @param[out] projected_rhs Projected right-hand side.
 * @return As ss_deflation_apply_left(). */
SsStatus ss_deflation_project_rhs(SsDeflation *deflation, Mat operator_handle, Vec rhs,
                                  Vec projected_rhs);
/** Assemble the deflated solution `solution = Q rhs + P_R complement`.
 * @param[in,out] deflation Refreshed store.
 * @param[in] operator_handle Refreshed operator.
 * @param[in] rhs Original right-hand side.
 * @param[in] complement Solution of the projected system.
 * @param[out] solution Full solution.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_deflation_final_correction(SsDeflation *deflation, Mat operator_handle,
                                       Vec rhs, Vec complement, Vec solution);
/** Measure the projector identities with full distributed applications; an
 * exhaustive oracle for tests and diagnostics.
 * @param[in,out] deflation Refreshed store.
 * @param[in] operator_handle Refreshed operator.
 * @param[out] maximum_error Largest identity error.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_deflation_invariant_error(SsDeflation *deflation, Mat operator_handle,
                                      PetscReal *maximum_error);
/** Layout-independent digest of the persistent basis; collective.
 * @param[in,out] deflation Store.
 * @param[out] output Hexadecimal digest, identical on all ranks.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_deflation_persistent_digest(SsDeflation *deflation,
                                        char output[SS_SHA256_HEX_CAPACITY]);
/** Number of persistent columns; local.
 * @param[in] deflation Store.
 * @param[out] count Persistent columns.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_deflation_persistent_count(const SsDeflation *deflation, size_t *count);
/** Borrow one persistent column and its provenance; local.
 *
 * The vectors stay valid until the next basis mutation, including a refresh
 * that reorthogonalizes the legacy basis.
 *
 * @param[in] deflation Store.
 * @param[in] index Column below ss_deflation_persistent_count().
 * @param[out] left Borrowed left column.
 * @param[out] right Borrowed right column.
 * @param[out] provenance Borrowed provenance string.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTEGRITY. */
SsStatus ss_deflation_persistent_entry(const SsDeflation *deflation, size_t index,
                                       Vec *left, Vec *right, const char **provenance);
/** Borrow one persistent column with the compression state that a restart
 * needs; local.
 * @param[in] deflation Store.
 * @param[in] index Column below ss_deflation_persistent_count().
 * @param[out] left Borrowed left column.
 * @param[out] right Borrowed right column.
 * @param[out] source Borrowed compression source vector, or NULL.
 * @param[out] source_weight Compression weight of the source.
 * @param[out] source_kind Provenance class of the source.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTEGRITY. */
SsStatus ss_deflation_persistent_state_entry(
    const SsDeflation *deflation, size_t index, Vec *left, Vec *right, Vec *source,
    PetscReal *source_weight, SsDeflationPersistentSourceKind *source_kind);
/** Borrow one column of the basis currently used by the projectors; local.
 *
 * Unlike the checkpoint accessors, this is valid while a transaction is open
 * and therefore includes provisional columns.
 *
 * @param[in] deflation Store.
 * @param[in] index Column below the active count.
 * @param[out] left Borrowed left column.
 * @param[out] right Borrowed right column.
 * @param[out] provenance Borrowed provenance string.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTEGRITY. */
SsStatus ss_deflation_active_entry(const SsDeflation *deflation, size_t index,
                                   Vec *left, Vec *right, const char **provenance);
/** Replace the basis with checkpointed columns and refresh it.
 * @param[in,out] deflation Store without an open transaction.
 * @param[in] left @p count left columns; copied.
 * @param[in] right @p count right columns; copied.
 * @param[in] count Number of columns.
 * @param[in] provenance Provenance recorded for every column.
 * @param[in] operator_handle Operator to refresh with.
 * @param[in] operator_generation Generation of @p operator_handle.
 * @param[in] accepted_generation Accepted generation of the checkpoint.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_deflation_restore_persistent(SsDeflation *deflation, const Vec *left,
                                         const Vec *right, size_t count,
                                         const char *provenance, Mat operator_handle,
                                         uint64_t operator_generation,
                                         uint64_t accepted_generation);
/** Replace the basis and its compression state with checkpointed columns and
 * refresh it.
 * @param[in,out] deflation Store without an open transaction.
 * @param[in] left @p count left columns; copied.
 * @param[in] right @p count right columns; copied.
 * @param[in] source @p count compression sources (entries may be NULL).
 * @param[in] source_weights @p count compression weights.
 * @param[in] source_kinds @p count provenance classes.
 * @param[in] count Number of columns.
 * @param[in] operator_handle Operator to refresh with.
 * @param[in] operator_generation Generation of @p operator_handle.
 * @param[in] accepted_generation Accepted generation of the checkpoint.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_deflation_restore_persistent_state(
    SsDeflation *deflation, const Vec *left, const Vec *right, const Vec *source,
    const PetscReal *source_weights,
    const SsDeflationPersistentSourceKind *source_kinds, size_t count,
    Mat operator_handle, uint64_t operator_generation, uint64_t accepted_generation);
/** Copy the counters and conditioning of a store; local.
 * @param[in] deflation Store.
 * @param[out] diagnostics Diagnostics.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_deflation_diagnostics(const SsDeflation *deflation,
                                  SsDeflationDiagnostics *diagnostics);
/** Projector formulation of a store; local.
 * @param[in] deflation Store.
 * @param[out] projector Projector formulation.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_deflation_projector(const SsDeflation *deflation,
                                SsDeflationProjector *projector);
/** Compression of a store; local.
 * @param[in] deflation Store.
 * @param[out] compression Compression.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_deflation_compression(const SsDeflation *deflation,
                                  SsDeflationCompression *compression);

#endif
