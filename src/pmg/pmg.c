/* PMG lifecycle, configured operators and diagnostics.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "pmg_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"

#include <petscmat.h>
#include <petscsys.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void ss_pmg_setup_clear(SsPmg *pmg) {
  size_t level;
  for (level = 0U; level < pmg->hierarchy.level_count; ++level) {
    ss_pmg_level_runtime_clear(&pmg->levels[level]);
    (void)VecDestroy(&pmg->rhs[level]);
    (void)VecDestroy(&pmg->solution[level]);
    (void)VecDestroy(&pmg->residual[level]);
    (void)VecDestroy(&pmg->delta[level]);
    (void)MatDestroy(&pmg->operators[level]);
  }
  (void)MatDestroy(&pmg->observed_fine_operator);
  pmg->observed_fine_state = 0;
  pmg->structure_generation = 0U;
  pmg->requested_value_generation = 0U;
  pmg->preconditioner_value_generation = 0U;
  pmg->setup_complete = 0;
  pmg->diagnostics.current_matrix_objects = 0U;
  pmg->diagnostics.current_vector_objects = 0U;
  pmg->diagnostics.admitted_memory_bytes = 0U;
  memset(&pmg->last_setup, 0, sizeof(pmg->last_setup));
}

SsStatus ss_pmg_create(MPI_Comm communicator, const SsPmgSpecification *specification,
                       SsPmg **result) {
  SsPmg *pmg = NULL;
  size_t level;
  SsStatus status;
  if (communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = SS_OK;
  if (result == NULL || specification == NULL || specification->orders == NULL ||
      specification->level_policies == NULL ||
      specification->transfer_generation == 0U ||
      specification->numeric_lag_interval == 0U ||
      specification->memory_cap_bytes == 0U ||
      (specification->level_count > 1U && specification->prolongations == NULL) ||
      specification->injected_failure > SS_PMG_FAIL_POST_SMOOTH ||
      (specification->injected_failure != SS_PMG_FAIL_NONE &&
       specification->injected_failure_level >= specification->level_count)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    *result = NULL;
    pmg = calloc(1U, sizeof(*pmg));
    if (pmg == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(pmg);
    return status;
  }
  pmg->communicator = MPI_COMM_NULL;
  if (MPI_Comm_dup(communicator, &pmg->communicator) != MPI_SUCCESS ||
      MPI_Comm_rank(pmg->communicator, &pmg->world_rank) != MPI_SUCCESS ||
      MPI_Comm_size(pmg->communicator, &pmg->world_size) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  pmg->fgmres_full_iterate_post = specification->fgmres_full_iterate_post != 0;
  pmg->ordered_galerkin = specification->ordered_galerkin != 0;
  status =
      ss_pmg_hierarchy_validate(specification->fine_order, specification->level_count,
                                specification->orders, &pmg->hierarchy);
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  pmg->transfer_generation = specification->transfer_generation;
  pmg->numeric_lag_interval = specification->numeric_lag_interval;
  pmg->memory_cap_bytes = specification->memory_cap_bytes;
  pmg->injected_failure = specification->injected_failure;
  pmg->injected_failure_level = specification->injected_failure_level;
  for (level = 0U; level < pmg->hierarchy.level_count; ++level) {
    status = ss_pmg_policy_copy(pmg, level, &specification->level_policies[level]);
    if (status == SS_OK) {
      status = ss_pmg_nullspace_storage_copy(
          &pmg->nullspace_inputs[level], specification->near_nullspaces == NULL
                                             ? NULL
                                             : &specification->near_nullspaces[level]);
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status != SS_OK) {
      goto cleanup;
    }
    /* All ranks must participate in the same nullspace construction even
     * when some have no resident nodes. */
    const SsPmgNearNullspaceInput *input = &pmg->nullspace_inputs[level].input;
    const PetscReal local[] = {(PetscReal)input->field,
                               -(PetscReal)input->field,
                               (PetscReal)input->dimension,
                               -(PetscReal)input->dimension,
                               (PetscReal)input->hydraulic_pure_neumann,
                               -(PetscReal)input->hydraulic_pure_neumann,
                               input->dependence_tolerance,
                               -input->dependence_tolerance};
    PetscReal extremes[8];
    if (MPI_Allreduce(local, extremes, 8, MPIU_REAL, MPI_MAX, pmg->communicator) !=
        MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
    for (size_t pair = 0; pair < 8U; pair += 2U) {
      if (extremes[pair] != -extremes[pair + 1U]) {
        status = SS_ERROR_INTEGRITY;
        goto cleanup;
      }
    }
    if (level + 1U < pmg->hierarchy.level_count) {
      if (specification->prolongations[level] == NULL ||
          PetscObjectReference((PetscObject)specification->prolongations[level]) !=
              PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      } else {
        pmg->prolongations[level] = specification->prolongations[level];
        if (MatGetState(pmg->prolongations[level], &pmg->prolongation_states[level]) !=
            PETSC_SUCCESS) {
          status = SS_ERROR_INTERNAL;
        }
      }
      status = ss_status_allreduce_max(pmg->communicator, status);
      if (status != SS_OK) {
        goto cleanup;
      }
    }
  }
  *result = pmg;
  return SS_OK;

cleanup:
  (void)ss_pmg_destroy(&pmg);
  return status;
}

SsStatus ss_pmg_set_level_operators(SsPmg *pmg, size_t level_count,
                                    const Mat *operators) {
  size_t level;
  SsStatus status = SS_OK;
  if (pmg == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (operators == NULL || level_count != pmg->hierarchy.level_count ||
      pmg->setup_complete != 0 || pmg->external_operator_mode != 0) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  for (level = 0U; level < level_count; ++level) {
    PetscInt rows = 0;
    PetscInt columns = 0;
    MPI_Comm matrix_communicator;
    int comparison = MPI_UNEQUAL;
    PetscBool shell = PETSC_FALSE;
    if (operators[level] == NULL ||
        MatGetSize(operators[level], &rows, &columns) != PETSC_SUCCESS || rows <= 0 ||
        rows != columns ||
        PetscObjectTypeCompare((PetscObject)operators[level], MATSHELL, &shell) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_TYPE_MISMATCH;
      break;
    }
    matrix_communicator = PetscObjectComm((PetscObject)operators[level]);
    if (MPI_Comm_compare(pmg->communicator, matrix_communicator, &comparison) !=
            MPI_SUCCESS ||
        (comparison != MPI_IDENT && comparison != MPI_CONGRUENT)) {
      status = SS_ERROR_TYPE_MISMATCH;
      break;
    }
    /* MatShell levels cannot be redistributed through MatGetRow. */
    if (shell == PETSC_TRUE &&
        pmg->levels[level].policy.active_rank_count != (size_t)pmg->world_size) {
      status = SS_ERROR_INVALID_ARGUMENT;
      break;
    }
    if (shell == PETSC_TRUE) {
      PetscMPIInt rank;
      for (rank = 0; rank < pmg->world_size; ++rank) {
        if (pmg->levels[level].active_ranks[rank] != rank) {
          status = SS_ERROR_INVALID_ARGUMENT;
          break;
        }
      }
      if (status != SS_OK) {
        break;
      }
    }
    if (PetscObjectReference((PetscObject)operators[level]) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    pmg->external_operators[level] = operators[level];
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    for (level = 0U; level < level_count; ++level) {
      (void)MatDestroy(&pmg->external_operators[level]);
    }
    return status;
  }
  pmg->external_operator_mode = 1;
  return SS_OK;
}

SsStatus ss_pmg_destroy(SsPmg **pmg_pointer) {
  SsPmg *pmg;
  size_t level;
  if (pmg_pointer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  pmg = *pmg_pointer;
  if (pmg == NULL) {
    return SS_OK;
  }
  ss_pmg_setup_clear(pmg);
  for (level = 0U; level < pmg->hierarchy.level_count; ++level) {
    free(pmg->levels[level].active_ranks);
    pmg->levels[level].active_ranks = NULL;
    free(pmg->nullspace_inputs[level].coordinates);
    free(pmg->nullspace_inputs[level].global_dof_indices);
    pmg->nullspace_inputs[level].coordinates = NULL;
    pmg->nullspace_inputs[level].global_dof_indices = NULL;
    if (level + 1U < pmg->hierarchy.level_count) {
      (void)MatDestroy(&pmg->prolongations[level]);
    }
    (void)MatDestroy(&pmg->external_operators[level]);
  }
  if (pmg->communicator != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&pmg->communicator);
  }
  free(pmg);
  *pmg_pointer = NULL;
  return SS_OK;
}

SsStatus ss_pmg_level_operator(SsPmg *pmg, size_t level, Mat *operator_handle) {
  if (pmg == NULL || operator_handle == NULL || level >= pmg->hierarchy.level_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *operator_handle = NULL;
  if (pmg->setup_complete == 0 || pmg->operators[level] == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  *operator_handle = pmg->operators[level];
  return SS_OK;
}

SsStatus ss_pmg_active_ranks_get(const SsPmg *pmg, size_t level,
                                 PetscMPIInt *active_ranks, size_t capacity,
                                 size_t *count) {
  size_t active_count;
  if (pmg == NULL || count == NULL || level >= pmg->hierarchy.level_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  active_count = pmg->levels[level].policy.active_rank_count;
  *count = active_count;
  if (active_ranks == NULL || capacity < active_count) {
    return SS_ERROR_RESOURCE;
  }
  memcpy(active_ranks, pmg->levels[level].active_ranks,
         active_count * sizeof(*active_ranks));
  return SS_OK;
}

SsStatus ss_pmg_level_local_ownership(const SsPmg *pmg, size_t level, int *active,
                                      int *owns_ksp, PetscMPIInt *active_rank) {
  if (pmg == NULL || active == NULL || owns_ksp == NULL || active_rank == NULL ||
      level >= pmg->hierarchy.level_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (pmg->setup_complete == 0) {
    return SS_ERROR_INVALID_STATE;
  }
  *active = pmg->levels[level].local_active;
  *owns_ksp = pmg->levels[level].local_owns_ksp;
  *active_rank = pmg->levels[level].local_active_rank;
  return SS_OK;
}

SsStatus ss_pmg_last_apply(const SsPmg *pmg, SsPmgApplyRecord *record) {
  if (pmg == NULL || record == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *record = pmg->last_apply;
  return SS_OK;
}

SsStatus ss_pmg_diagnostics(const SsPmg *pmg, SsPmgDiagnostics *diagnostics) {
  if (pmg == NULL || diagnostics == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *diagnostics = pmg->diagnostics;
  return SS_OK;
}
