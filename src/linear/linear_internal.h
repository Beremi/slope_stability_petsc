/* Private linear solver state and workspace interfaces.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_LINEAR_INTERNAL_H
#define SS_LINEAR_INTERNAL_H

#include "slope_internal/core/collective.h"
#include "slope_internal/linear/linear.h"

#include <petsclog.h>

typedef struct {
  SsDeflation *deflation;
  Mat operator_handle;
  int failed;
} SsProjectedOperatorContext;

typedef struct {
  Vec *vectors;
  size_t count;
  size_t allocated;
  int enabled;
} SsLinearKrylovCaptureBuffer;

typedef struct {
  SsPmg *pmg;
  SsDeflation *deflation;
  Vec work;
  SsLinearKrylovCaptureBuffer *capture;
} SsProjectedPmgPcContext;

struct SsLinearRecycle {
  MPI_Comm communicator;
  Mat space;
  Mat snapshot;
  uint64_t update_count;
  uint64_t snapshot_update_count;
  uint64_t transaction_generation;
  uint64_t accepted_generation;
  int transaction_open;
};

struct SsLinearSolver {
  MPI_Comm communicator;
  Mat operator_handle;
  Mat preconditioner_operator_handle;
  Mat projected_operator;
  KSP ksp;
  SsLinearCancellation cancelled;
  SsLinearObserver observer;
  PetscInt last_observed_iteration;
  void *cancellation_context;
  Vec projected_rhs;
  Vec complement;
  Vec residual;
  SsDeflation *deflation;
  SsPmg *pmg;
  SsLinearRecycle *krylov_recycle;
  SsLinearKrylovCapture krylov_capture;
  SsLinearKrylovCaptureBuffer captured_krylov;
  PetscInt recycled_dimension_supplied;
  size_t deflation_dimension_supplied;
  uint64_t deflation_refresh_count;
  int solve_since_setup;
  SsProjectedOperatorContext projected_context;
  SsProjectedPmgPcContext projected_pc_context;
  SsLinearPolicy selected_policy;
  SsLinearOperatorProperties properties;
  SsLinearPolicyChain policy_chain;
  SsLinearSetupRecord last_setup_record;
  char *options_snapshot;
  PetscObjectState operator_state;
  PetscObjectState preconditioner_operator_state;
  PetscObjectState operator_nonzero_state;
  PetscObjectState preconditioner_nonzero_state;
  uint64_t structure_generation;
  uint64_t value_generation;
  uint64_t setup_count;
  uint64_t solve_count;
  int setup_complete;
};

extern PetscLogEvent ss_linear_setup_event;
extern PetscLogEvent ss_linear_solve_event;
extern PetscLogEvent ss_linear_deflation_refresh_event;
extern PetscLogEvent ss_linear_pmg_setup_event;
extern PetscLogEvent ss_linear_initial_guess_event;
extern PetscLogEvent ss_linear_krylov_recycle_event;
extern PetscLogEvent ss_linear_full_krylov_event;

/** Agree a PETSc return code before entering another collective phase. */
static inline SsStatus ss_linear_petsc_status(MPI_Comm comm, PetscErrorCode ierr) {
  return ss_status_allreduce_max(comm,
                                 ierr == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL);
}

/** PETSc monitor that dispatches the current solve's local observer. */
PetscErrorCode ss_linear_krylov_monitor(KSP ksp, PetscInt iteration,
                                        PetscReal residual_norm, void *context);

/** Select the PETSc norm matching the configured stopping metric. */
KSPNormType ss_linear_norm_type(const SsLinearPolicy *policy);

/** Save the HPDDM recycle space after a completed solve. */
SsStatus ss_linear_capture_krylov_recycle(SsLinearSolver *solver);
/** Reset the current solve capture. */
void ss_linear_krylov_capture_clear(SsLinearKrylovCaptureBuffer *capture);
/** Release all retained capture storage. */
void ss_linear_krylov_capture_destroy(SsLinearKrylovCaptureBuffer *capture);
/** Store a copy of a flexible Krylov correction for promotion. */
PetscErrorCode ss_linear_krylov_capture_append(SsLinearKrylovCaptureBuffer *capture,
                                               Vec direction);

#endif
