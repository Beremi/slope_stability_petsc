/* Transactional HPDDM recycle spaces.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "linear_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

SsStatus ss_linear_recycle_create(MPI_Comm communicator, SsLinearRecycle **result) {
  SsLinearRecycle *recycle = NULL;
  if (communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  SsStatus status = SS_OK;
  if (result == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    *result = NULL;
    recycle = calloc(1U, sizeof(*recycle));
    if (recycle == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(recycle);
    return status;
  }
  recycle->communicator = MPI_COMM_NULL;
  if (MPI_Comm_dup(communicator, &recycle->communicator) != MPI_SUCCESS) {
    free(recycle);
    return SS_ERROR_INTERNAL;
  }
  *result = recycle;
  return SS_OK;
}

SsStatus ss_linear_recycle_begin(SsLinearRecycle *recycle,
                                 uint64_t transaction_generation) {
  if (recycle == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  SsStatus status = ss_status_allreduce_max(recycle->communicator,
                                            transaction_generation == 0U ||
                                                    recycle->transaction_open != 0 ||
                                                    recycle->snapshot != NULL
                                                ? SS_ERROR_INVALID_STATE
                                                : SS_OK);
  if (status != SS_OK) {
    return status;
  }
  if (recycle->space != NULL) {
    if (PetscObjectReference((PetscObject)recycle->space) == PETSC_SUCCESS) {
      recycle->snapshot = recycle->space;
    } else {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(recycle->communicator, status);
  if (status != SS_OK) {
    (void)MatDestroy(&recycle->snapshot);
    return status;
  }
  recycle->transaction_generation = transaction_generation;
  recycle->snapshot_update_count = recycle->update_count;
  recycle->transaction_open = 1;
  return SS_OK;
}

SsStatus ss_linear_recycle_commit(SsLinearRecycle *recycle,
                                  uint64_t accepted_generation) {
  if (recycle == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  SsStatus status = ss_status_allreduce_max(
      recycle->communicator, accepted_generation == 0U || recycle->transaction_open == 0
                                 ? SS_ERROR_INVALID_STATE
                                 : SS_OK);
  if (status != SS_OK) {
    return status;
  }
  status =
      ss_linear_petsc_status(recycle->communicator, MatDestroy(&recycle->snapshot));
  if (status != SS_OK) {
    return status;
  }
  recycle->accepted_generation = accepted_generation;
  recycle->transaction_generation = 0U;
  recycle->snapshot_update_count = 0U;
  recycle->transaction_open = 0;
  return SS_OK;
}

SsStatus ss_linear_recycle_rollback(SsLinearRecycle *recycle) {
  if (recycle == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  SsStatus status = ss_status_allreduce_max(
      recycle->communicator,
      recycle->transaction_open == 0 ? SS_ERROR_INVALID_STATE : SS_OK);
  if (status != SS_OK) {
    return status;
  }
  status = ss_linear_petsc_status(recycle->communicator, MatDestroy(&recycle->space));
  if (status != SS_OK) {
    return status;
  }
  recycle->space = recycle->snapshot;
  recycle->snapshot = NULL;
  recycle->update_count = recycle->snapshot_update_count;
  recycle->snapshot_update_count = 0U;
  recycle->transaction_generation = 0U;
  recycle->transaction_open = 0;
  return SS_OK;
}

SsStatus ss_linear_recycle_diagnostics(const SsLinearRecycle *recycle,
                                       SsLinearRecycleDiagnostics *diagnostics) {
  PetscInt dimension = 0;
  if (recycle == NULL || diagnostics == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (recycle->space != NULL &&
      MatGetSize(recycle->space, NULL, &dimension) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  memset(diagnostics, 0, sizeof(*diagnostics));
  diagnostics->dimension = dimension;
  diagnostics->update_count = recycle->update_count;
  diagnostics->transaction_generation = recycle->transaction_generation;
  diagnostics->accepted_generation = recycle->accepted_generation;
  diagnostics->transaction_open = recycle->transaction_open;
  return SS_OK;
}

SsStatus ss_linear_recycle_destroy(SsLinearRecycle **recycle_pointer) {
  SsLinearRecycle *recycle;
  SsStatus status = SS_OK;
  if (recycle_pointer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  recycle = *recycle_pointer;
  if (recycle == NULL) {
    return SS_OK;
  }
  if (MatDestroy(&recycle->space) != PETSC_SUCCESS ||
      MatDestroy(&recycle->snapshot) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (recycle->communicator != MPI_COMM_NULL &&
      MPI_Comm_free(&recycle->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free(recycle);
  *recycle_pointer = NULL;
  return status;
}

SsStatus ss_linear_capture_krylov_recycle(SsLinearSolver *solver) {
#if defined(PETSC_HAVE_HPDDM)
  Mat updated = NULL;
  PetscInt rows = 0;
  PetscInt columns = 0;
  SsStatus status = SS_OK;
  if (solver == NULL || solver->ksp == NULL || solver->krylov_recycle == NULL ||
      solver->selected_policy.method != SS_LINEAR_METHOD_HPDDM_GCRODR ||
      solver->solve_since_setup == 0) {
    return SS_OK;
  }
  (void)PetscLogEventBegin(ss_linear_krylov_recycle_event, 0, 0, 0, 0);
  if (KSPHPDDMGetDeflationMat(solver->ksp, &updated) != PETSC_SUCCESS ||
      updated == NULL || MatGetSize(updated, &rows, &columns) != PETSC_SUCCESS ||
      rows <= 0 || columns <= 0 ||
      columns > solver->selected_policy.recycle_dimension) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status != SS_OK) {
    (void)MatDestroy(&updated);
    (void)PetscLogEventEnd(ss_linear_krylov_recycle_event, 0, 0, 0, 0);
    return status;
  }
  if (columns < solver->selected_policy.recycle_dimension) {
    /* HPDDM reduces k to the first solve's Arnoldi dimension when that solve
     * converges before k steps.  Promoting such an undersized space would
     * permanently cap every later tangent at that smaller k.  Leave the
     * accepted full-sized space unchanged, or leave recycling empty until a
     * sufficiently rich Krylov cycle has been observed. */
  } else if (MatDestroy(&solver->krylov_recycle->space) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else {
    solver->krylov_recycle->space = updated;
    updated = NULL;
    ++solver->krylov_recycle->update_count;
  }
  (void)MatDestroy(&updated);
  if (PetscLogEventEnd(ss_linear_krylov_recycle_event, 0, 0, 0, 0) != PETSC_SUCCESS &&
      status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  solver->solve_since_setup = 0;
  return status;
#else
  (void)solver;
  return SS_OK;
#endif
}
