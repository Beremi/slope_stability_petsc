/* Private deflation basis, workspaces and module interfaces.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_DEFLATION_DEFLATION_INTERNAL_H
#define SS_DEFLATION_DEFLATION_INTERNAL_H

#include "slope_internal/core/collective.h"
#include "slope_internal/deflation/deflation.h"

#include <petsclog.h>

typedef struct {
  Vec left;
  Vec right;
  Vec az;
  Vec atz;
  /* Immutable compression representation.  The legacy projector is allowed
   * to overwrite left/right on every tangent; source and source_weight carry
   * the Euclidean POD/SVD information into the next incremental update. */
  Vec source;
  PetscReal source_weight;
  char provenance[SS_DEFLATION_PROVENANCE_CAPACITY];
} SsDeflationEntry;

struct SsDeflation {
  MPI_Comm communicator;
  SsDeflationPolicy policy;
  Mat operator_handle;
  SsDeflationEntry *persistent;
  SsDeflationEntry *provisional;
  SsDeflationEntry *transaction_base_persistent;
  size_t persistent_count;
  size_t provisional_count;
  size_t transaction_base_persistent_count;
  size_t active_count;
  size_t active_skip_extra;
  size_t refresh_retry_depth;
  int transaction_open;
  int transaction_basis_restored;
  int factors_ready;
  int invariants_validated;
  PetscScalar *coarse;
  PetscScalar *lu;
  PetscScalar *dots;
  PetscScalar *coefficients;
  PetscInt *pivots;
  Vec *active_left;
  Vec *active_right;
  Vec *active_az;
  Vec *active_atz;
  Vec work[4];
  uint64_t operator_generation;
  uint64_t accepted_generation;
  uint64_t transaction_generation;
  uint64_t transaction_base_operator_generation;
  PetscObjectState transaction_base_operator_state;
  Mat transaction_base_operator_handle;
  uint64_t global_reductions;
  uint64_t refresh_count;
  uint64_t q_apply_count;
  uint64_t projected_apply_count;
  uint64_t ill_conditioned_drop_count;
  SsDeflationDropReason last_drop_reason;
  PetscObjectState operator_state;
  PetscReal minimum_relative_pivot;
  PetscReal condition_estimate;
  PetscReal maximum_invariant_error;
  uint64_t compression_count;
  size_t last_compression_input_count;
  size_t last_compression_protected_count;
  size_t last_compression_retained_count;
  PetscReal last_compression_information_fraction;
};

/* PETSc events */
extern PetscLogEvent ss_deflation_compression_event;

/** Agree a PETSc status before any rank enters the next phase. */
static inline SsStatus ss_deflation_petsc_status(const SsDeflation *deflation,
                                                 PetscErrorCode ierr) {
  return ss_status_allreduce_max(deflation->communicator,
                                 ierr == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL);
}

/** Duplicate and copy a vector with collective status checks. */
SsStatus ss_deflation_vector_copy(SsDeflation *deflation, Vec source, Vec *destination);

/* Active and candidate basis storage and admission */
/** Destroy the vectors of a basis entry and zero it. */
void ss_deflation_entry_clear(SsDeflationEntry *entry);
/** Deep-copy a basis entry; collective (vector duplication). */
SsStatus ss_deflation_entry_clone(const SsDeflationEntry *source,
                                  SsDeflationEntry *destination);
/** Clear and free an array of basis entries. */
void ss_deflation_entry_array_destroy(SsDeflationEntry **entries, size_t capacity);
/** Entry of the persistent-then-provisional sequence. */
SsDeflationEntry *ss_deflation_combined_entry(SsDeflation *deflation, size_t index);
/** Read-only entry of the persistent-then-provisional sequence. */
const SsDeflationEntry *ss_deflation_const_combined_entry(const SsDeflation *deflation,
                                                          size_t index);
/** Leading combined entries outside the active window. */
size_t ss_deflation_active_skip(const SsDeflation *deflation);
/** Active entry by index. */
SsDeflationEntry *ss_deflation_entry(SsDeflation *deflation, size_t index);
/** Read-only active entry by index. */
const SsDeflationEntry *ss_deflation_const_entry(const SsDeflation *deflation,
                                                 size_t index);
/** Recompute the active window from the stored entries and the cap. */
void ss_deflation_update_active_count(SsDeflation *deflation);
/** Drop the newest provisional entry with a reason. */
int ss_deflation_drop_newest_provisional(SsDeflation *deflation,
                                         SsDeflationDropReason reason);
/** Drop the oldest active entry from the view with a reason. */
int ss_deflation_drop_oldest_active_view(SsDeflation *deflation,
                                         SsDeflationDropReason reason);
/** Remove one combined entry with a reason. */
int ss_deflation_remove_combined_entry(SsDeflation *deflation, size_t index,
                                       SsDeflationDropReason reason);
/** Destroy the work vectors of a store. */
void ss_deflation_destroy_work(SsDeflation *deflation);

/** Form the column-major Gram matrix of two distributed vector banks. */
SsStatus ss_deflation_gram(SsDeflation *deflation, const Vec *right, const Vec *left,
                           size_t dimension, PetscScalar *gram);

/* Deflation coarse factorization and left/right projector actions */
/** Drop the coarse factorization and cached `A Z`. */
void ss_deflation_clear_factorization(SsDeflation *deflation);
/** Copy the current counts into an admission record. */
void ss_deflation_fill_record(const SsDeflation *deflation,
                              SsDeflationAdmissionRecord *record);
/** Form and LU-factorize `E = W^T A Z` with pivot checks; collective. */
SsStatus ss_deflation_factorize(SsDeflation *deflation);
/** Apply the factored coarse inverse in place. */
SsStatus ss_deflation_lu_solve(const SsDeflation *deflation, PetscScalar *values);
/** Projector identity residuals through the coarse space; collective. */
SsStatus ss_deflation_coarse_invariant_error(SsDeflation *deflation,
                                             PetscReal *maximum_error);
/** Whether the store was refreshed for this operator and state. */
SsStatus ss_deflation_operator_ready(const SsDeflation *deflation, Mat operator_handle);

/* Legacy A-orthogonal projector, cache, and transaction restore */
/** Restore the A-orthogonal basis saved at the transaction start; collective. */
SsStatus ss_deflation_restore_legacy_transaction_basis(SsDeflation *deflation);
/** Refresh the A-orthogonal basis and its cached images; collective. */
SsStatus ss_deflation_refresh_legacy_aorth(SsDeflation *deflation, Mat operator_handle,
                                           PetscObjectState current_state,
                                           uint64_t operator_generation);

/* Snapshot selection and compression for deflation recycling */
/** Compress the basis with the configured method at commit; collective. */
SsStatus ss_deflation_compress(SsDeflation *deflation, Mat operator_handle);

#endif
