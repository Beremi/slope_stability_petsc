/* PMG setup, numeric refresh and memory accounting.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "pmg_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/diagnostics/diagnostics.h"

#include <petscmat.h>
#include <petscsys.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

static SsStatus ss_pmg_size_add(size_t *total, size_t count, size_t item_size) {
  size_t bytes;
  if (count > SIZE_MAX / item_size) {
    return SS_ERROR_OVERFLOW;
  }
  bytes = count * item_size;
  if (*total > SIZE_MAX - bytes) {
    return SS_ERROR_OVERFLOW;
  }
  *total += bytes;
  return SS_OK;
}

static SsStatus ss_pmg_matrix_memory(Mat matrix, size_t *bytes) {
  MatInfo info;
  PetscInt rows = 0;
  PetscInt columns = 0;
  size_t nonzeros;
  PetscBool shell = PETSC_FALSE;
  if (PetscObjectTypeCompare((PetscObject)matrix, MATSHELL, &shell) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (shell == PETSC_TRUE) {
    return SS_OK;
  }
  if (MatGetInfo(matrix, MAT_GLOBAL_SUM, &info) != PETSC_SUCCESS ||
      MatGetSize(matrix, &rows, &columns) != PETSC_SUCCESS || rows < 0 || columns < 0 ||
      !isfinite(info.nz_allocated) || info.nz_allocated < 0.0 ||
      info.nz_allocated > (double)SIZE_MAX) {
    return SS_ERROR_INTERNAL;
  }
  nonzeros = (size_t)ceil(info.nz_allocated);
  if (ss_pmg_size_add(bytes, nonzeros, sizeof(PetscScalar)) != SS_OK ||
      ss_pmg_size_add(bytes, nonzeros, sizeof(PetscInt)) != SS_OK ||
      ss_pmg_size_add(bytes, (size_t)rows + 1U, sizeof(PetscInt)) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  return SS_OK;
}

static SsStatus ss_pmg_memory_audit(SsPmg *pmg, size_t *matrix_objects,
                                    size_t *vector_objects, size_t *admitted_bytes) {
  size_t level;
  size_t bytes = 0U;
  size_t matrices = 0U;
  size_t vectors = 0U;
  for (level = 0U; level < pmg->hierarchy.level_count; ++level) {
    PetscInt size = 0;
    SsStatus status = ss_pmg_matrix_memory(pmg->operators[level], &bytes);
    if (status != SS_OK ||
        MatGetSize(pmg->operators[level], &size, NULL) != PETSC_SUCCESS || size <= 0 ||
        ss_pmg_size_add(&bytes, (size_t)size, 6U * sizeof(PetscScalar)) != SS_OK) {
      return status != SS_OK ? status : SS_ERROR_OVERFLOW;
    }
    ++matrices;
    vectors += 6U;
    if (pmg->levels[level].identity_world_layout == 0) {
      if (ss_pmg_size_add(&bytes, (size_t)size, 2U * sizeof(PetscScalar)) != SS_OK) {
        return SS_ERROR_OVERFLOW;
      }
      status = ss_pmg_matrix_memory(pmg->operators[level], &bytes);
      if (status != SS_OK) {
        return status;
      }
      ++matrices;
    }
    if (level + 1U < pmg->hierarchy.level_count) {
      if (level >= SS_PMG_MAX_LEVELS - 1U) {
        return SS_ERROR_INTEGRITY;
      }
      status = ss_pmg_matrix_memory(pmg->prolongations[level], &bytes);
      if (status != SS_OK) {
        return status;
      }
      ++matrices;
    }
  }
  if (bytes > pmg->memory_cap_bytes) {
    return SS_ERROR_RESOURCE;
  }
  *matrix_objects = matrices;
  *vector_objects = vectors;
  *admitted_bytes = bytes;
  return SS_OK;
}

static SsStatus ss_pmg_observe_operator(SsPmg *pmg, Mat fine_operator,
                                        PetscObjectState state) {
  if (pmg->observed_fine_operator != fine_operator) {
    if (PetscObjectReference((PetscObject)fine_operator) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    (void)MatDestroy(&pmg->observed_fine_operator);
    pmg->observed_fine_operator = fine_operator;
  }
  pmg->observed_fine_state = state;
  return SS_OK;
}

static SsStatus ss_pmg_transfers_unchanged(SsPmg *pmg) {
  size_t level;
  for (level = 0U; level + 1U < pmg->hierarchy.level_count; ++level) {
    PetscObjectState state = 0;
    if (MatGetState(pmg->prolongations[level], &state) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (state != pmg->prolongation_states[level]) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

static SsStatus ss_pmg_setup_record_fill(SsPmg *pmg, uint64_t requested_generation,
                                         int hierarchy_reused, int transfer_reused,
                                         int pattern_reused, int numeric_refreshed,
                                         int numeric_lagged, SsPmgSetupRecord *record) {
  size_t level;
  size_t matrix_objects = 0U;
  size_t vector_objects = 0U;
  size_t admitted_bytes = 0U;
  if (hierarchy_reused) {
    matrix_objects = pmg->last_setup.matrix_object_count;
    vector_objects = pmg->last_setup.vector_object_count;
    admitted_bytes = pmg->last_setup.admitted_memory_bytes;
  } else {
    SsStatus status =
        ss_pmg_memory_audit(pmg, &matrix_objects, &vector_objects, &admitted_bytes);
    if (status != SS_OK) {
      return status;
    }
  }
  memset(record, 0, sizeof(*record));
  record->hierarchy = pmg->hierarchy;
  record->transfer_generation = pmg->transfer_generation;
  record->structure_generation = pmg->structure_generation;
  record->requested_value_generation = requested_generation;
  record->preconditioner_value_generation = pmg->preconditioner_value_generation;
  record->hierarchy_build_count = pmg->hierarchy_build_count;
  record->transfer_setup_count = pmg->transfer_setup_count;
  record->galerkin_symbolic_count = pmg->galerkin_symbolic_count;
  record->galerkin_numeric_count = pmg->galerkin_numeric_count;
  record->level_solver_setup_count = pmg->level_solver_setup_count;
  record->hierarchy_reused = hierarchy_reused;
  record->transfer_reused = transfer_reused;
  record->galerkin_pattern_reused = pattern_reused;
  record->numeric_refreshed = numeric_refreshed;
  record->numeric_lagged = numeric_lagged;
  record->matrix_object_count = matrix_objects;
  record->vector_object_count = vector_objects;
  record->admitted_memory_bytes = admitted_bytes;
  for (level = 0U; level < pmg->hierarchy.level_count; ++level) {
    SsPmgLevelSetupRecord *level_record = &record->levels[level];
    PetscInt rows = 0;
    if (MatGetSize(pmg->operators[level], &rows, NULL) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    level_record->order = pmg->hierarchy.orders[level];
    level_record->global_dofs = rows;
    level_record->method = pmg->levels[level].policy.method;
    level_record->redundant_solve_count =
        pmg->levels[level].policy.redundant_solve_count;
    level_record->active_rank_count = pmg->levels[level].policy.active_rank_count;
    level_record->local_rank_active = pmg->levels[level].local_active;
    level_record->local_owns_ksp = pmg->levels[level].local_owns_ksp;
    level_record->local_active_rank = pmg->levels[level].local_active_rank;
    level_record->repeated_factorization_reuses_row_permutation =
        pmg->levels[level].repeated_factorization_reuses_row_permutation;
    level_record->variable_point_block_count =
        pmg->levels[level].variable_point_block_count;
    level_record->retained_near_nullspace_modes =
        pmg->levels[level].nullspace_record.retained_mode_count;
    level_record->dropped_near_nullspace_modes =
        pmg->levels[level].nullspace_record.dropped_mode_count;
    level_record->solver_setup_count = pmg->levels[level].solver_setup_count;
  }
  pmg->diagnostics.current_matrix_objects = matrix_objects;
  pmg->diagnostics.current_vector_objects = vector_objects;
  if (matrix_objects > pmg->diagnostics.peak_matrix_objects) {
    pmg->diagnostics.peak_matrix_objects = matrix_objects;
  }
  if (vector_objects > pmg->diagnostics.peak_vector_objects) {
    pmg->diagnostics.peak_vector_objects = vector_objects;
  }
  pmg->diagnostics.admitted_memory_bytes = admitted_bytes;
  pmg->last_setup = *record;
  return SS_OK;
}

SsStatus ss_pmg_setup(SsPmg *pmg, Mat fine_operator, uint64_t structure_generation,
                      uint64_t value_generation, SsPmgSetupRecord *record) {
  PetscObjectState fine_state = 0;
  MPI_Comm matrix_communicator;
  int communicator_comparison = MPI_UNEQUAL;
  SsStatus status;
  if (pmg == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (fine_operator == NULL || record == NULL || structure_generation == 0U ||
      value_generation == 0U) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    matrix_communicator = PetscObjectComm((PetscObject)fine_operator);
    if (MPI_Comm_compare(pmg->communicator, matrix_communicator,
                         &communicator_comparison) != MPI_SUCCESS ||
        (communicator_comparison != MPI_IDENT &&
         communicator_comparison != MPI_CONGRUENT) ||
        MatGetState(fine_operator, &fine_state) != PETSC_SUCCESS) {
      status = SS_ERROR_TYPE_MISMATCH;
    } else {
      status = ss_pmg_transfers_unchanged(pmg);
    }
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  ++pmg->setup_calls;
  pmg->diagnostics.setup_calls = pmg->setup_calls;
  if (pmg->setup_complete != 0 && structure_generation == pmg->structure_generation) {
    if (value_generation < pmg->requested_value_generation) {
      return SS_ERROR_VERSION_MISMATCH;
    }
    if (value_generation == pmg->requested_value_generation) {
      status = ss_status_allreduce_max(pmg->communicator,
                                       fine_operator == pmg->observed_fine_operator &&
                                               fine_state == pmg->observed_fine_state
                                           ? SS_OK
                                           : SS_ERROR_INTEGRITY);
      if (status == SS_OK) {
        status = ss_pmg_setup_record_fill(
            pmg, value_generation, 1, 1, 1, 0,
            pmg->preconditioner_value_generation != value_generation, record);
      }
      return ss_status_allreduce_max(pmg->communicator, status);
    }
    status = ss_status_allreduce_max(
        pmg->communicator, ss_pmg_observe_operator(pmg, fine_operator, fine_state));
    if (status != SS_OK) {
      return status;
    }
    pmg->requested_value_generation = value_generation;
    if (value_generation - pmg->preconditioner_value_generation <
        pmg->numeric_lag_interval) {
      return ss_status_allreduce_max(
          pmg->communicator,
          ss_pmg_setup_record_fill(pmg, value_generation, 1, 1, 1, 0, 1, record));
    }
    status = ss_pmg_world_hierarchy_refresh(pmg, fine_operator);
#ifndef NDEBUG
    if (status == SS_OK) {
      for (size_t level = 0U; level < pmg->hierarchy.level_count; ++level) {
        ss_debug_stage_matrix("pmg-refresh-matrix", level, pmg->operators[level]);
      }
    }
#endif
    if (status == SS_OK) {
      status = ss_pmg_active_levels_refresh(pmg);
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status != SS_OK) {
      ss_pmg_setup_clear(pmg);
      return status;
    }
    pmg->preconditioner_value_generation = value_generation;
    status = ss_status_allreduce_max(
        pmg->communicator,
        ss_pmg_setup_record_fill(pmg, value_generation, 1, 1, 1, 1, 0, record));
    if (status != SS_OK) {
      ss_pmg_setup_clear(pmg);
    }
    return status;
  }
  ss_pmg_setup_clear(pmg);
  status = ss_pmg_world_hierarchy_build(pmg, fine_operator);
#ifndef NDEBUG
  if (status == SS_OK) {
    for (size_t level = 0U; level < pmg->hierarchy.level_count; ++level) {
      ss_debug_stage_matrix("pmg-world-matrix", level, pmg->operators[level]);
      if (level + 1U < pmg->hierarchy.level_count) {
        ss_debug_stage_matrix("pmg-transfer", level, pmg->prolongations[level]);
      }
    }
  }
#endif
  if (status == SS_OK) {
    status = ss_pmg_near_nullspaces_build(pmg);
  }
  if (status == SS_OK) {
    status = ss_pmg_active_levels_build(pmg);
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    ss_pmg_setup_clear(pmg);
    return status;
  }
  ++pmg->hierarchy_build_count;
  pmg->structure_generation = structure_generation;
  pmg->requested_value_generation = value_generation;
  pmg->preconditioner_value_generation = value_generation;
  status = ss_status_allreduce_max(
      pmg->communicator, ss_pmg_observe_operator(pmg, fine_operator, fine_state));
  if (status != SS_OK) {
    ss_pmg_setup_clear(pmg);
    return status;
  }
  pmg->setup_complete = 1;
  status = ss_status_allreduce_max(
      pmg->communicator,
      ss_pmg_setup_record_fill(pmg, value_generation, 0, 0, 0, 1, 0, record));
  if (status != SS_OK) {
    ss_pmg_setup_clear(pmg);
  }
  return status;
}
