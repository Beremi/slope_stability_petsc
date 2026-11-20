/* Persistent PETSc storage for Newton and continuation attempts.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "nonlinear_internal.h"

#include <stdlib.h>

SsStatus ss_nonlinear_workspace_create(MPI_Comm communicator,
                                       SsNonlinearWorkspace **workspace) {
  SsNonlinearWorkspace *work = NULL;
  SsStatus status = SS_OK;
  if (communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (workspace == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    *workspace = NULL;
    work = calloc(1U, sizeof(*work));
    if (work == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(work);
    return status;
  }
  work->communicator = communicator;
  *workspace = work;
  return SS_OK;
}

static SsStatus ss_nonlinear_workspace_clear(SsNonlinearWorkspace *work) {
  SsStatus status = ss_linear_destroy(&work->linear);
  for (size_t i = 0U; i < SS_NL_VECTOR_COUNT; ++i) {
    if (VecDestroy(&work->work[i]) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  if (MatDestroy(&work->tangent) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (!work->matrix_free && MatDestroy(&work->regularized) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  work->regularized = NULL;
  if (MatDestroy(&work->elastic_template) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  work->allocated_vectors = 0U;
  work->structure_generation = 0U;
  return ss_status_allreduce_max(work->communicator, status);
}

SsStatus ss_nonlinear_workspace_destroy(SsNonlinearWorkspace **workspace) {
  if (workspace == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*workspace == NULL) {
    return SS_OK;
  }
  const SsStatus status = ss_nonlinear_workspace_clear(*workspace);
  free(*workspace);
  *workspace = NULL;
  return status;
}

SsStatus ss_nonlinear_workspace_prepare(const SsNonlinearProblem *problem,
                                        Vec prototype, int matrix_free,
                                        size_t vector_count,
                                        SsNonlinearWorkspace **result,
                                        SsNonlinearWorkspace **temporary) {
  SsNonlinearWorkspace *work = problem->workspace;
  SsStatus status = SS_OK;
  const MPI_Comm comm = problem->communicator;
  if (work == NULL) {
    status = ss_nonlinear_workspace_create(comm, temporary);
    if (status != SS_OK) {
      return status;
    }
    work = *temporary;
  }
  int rebuild = work->elastic_template != problem->elastic_reference ||
                work->structure_generation != problem->structure_generation ||
                work->matrix_free != matrix_free;
  if (!rebuild) {
    PetscLayout input_layout = NULL, workspace_layout = NULL;
    PetscBool compatible = PETSC_FALSE;
    if (VecGetLayout(prototype, &input_layout) != PETSC_SUCCESS ||
        VecGetLayout(work->work[SS_NL_WORKING], &workspace_layout) != PETSC_SUCCESS ||
        PetscLayoutCompare(input_layout, workspace_layout, &compatible) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      rebuild = !compatible;
    }
  }
  status = ss_status_allreduce_max(comm, status);
  int any_rebuild = 0;
  if (status == SS_OK &&
      MPI_Allreduce(&rebuild, &any_rebuild, 1, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status != SS_OK) {
    return status;
  }
  if (any_rebuild) {
    status = ss_nonlinear_workspace_clear(work);
    if (status == SS_OK) {
      const PetscErrorCode ref =
          PetscObjectReference((PetscObject)problem->elastic_reference);
      if (ref == PETSC_SUCCESS) {
        work->elastic_template = problem->elastic_reference;
      }
      status = ss_nonlinear_petsc_status(comm, ref);
    }
    if (status == SS_OK && !matrix_free) {
      status = ss_nonlinear_petsc_status(comm, MatDuplicate(problem->elastic_reference,
                                                            MAT_DO_NOT_COPY_VALUES,
                                                            &work->tangent));
      if (status == SS_OK) {
        status = ss_nonlinear_petsc_status(
            comm, MatDuplicate(problem->elastic_reference, MAT_DO_NOT_COPY_VALUES,
                               &work->regularized));
      }
    } else if (status == SS_OK) {
      work->regularized = work->elastic_template;
    }
    work->matrix_free = matrix_free;
    if (status == SS_OK) {
      status = ss_linear_create(comm, &work->linear);
    }
    if (status == SS_OK) {
      work->structure_generation = problem->structure_generation;
    }
  }
  for (size_t i = work->allocated_vectors; status == SS_OK && i < vector_count; ++i) {
    status = ss_nonlinear_petsc_status(comm, VecDuplicate(prototype, &work->work[i]));
    if (status == SS_OK) {
      ++work->allocated_vectors;
    }
  }
  if (status != SS_OK) {
    (void)ss_nonlinear_workspace_clear(work);
    return status;
  }
  *result = work;
  return SS_OK;
}
