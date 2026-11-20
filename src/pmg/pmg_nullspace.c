/* Rigid-mode and hydraulic near-nullspace construction.
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

static size_t ss_pmg_nullspace_components(const SsPmgNearNullspaceInput *input) {
  return input->field == SS_PMG_FIELD_MECHANICS ? (size_t)input->dimension : 1U;
}

static size_t ss_pmg_nullspace_raw_modes(const SsPmgNearNullspaceInput *input) {
  if (input->field == SS_PMG_FIELD_MECHANICS) {
    return input->dimension == 2U ? 3U : 6U;
  }
  return input->hydraulic_pure_neumann != 0 ? 1U : 0U;
}

static SsStatus ss_pmg_nullspace_input_validate(const SsPmgNearNullspaceInput *input) {
  const size_t components = input == NULL ? 0U : ss_pmg_nullspace_components(input);
  size_t coordinate;
  if (input == NULL ||
      (input->field != SS_PMG_FIELD_MECHANICS &&
       input->field != SS_PMG_FIELD_HYDRAULIC) ||
      (input->dimension != 2U && input->dimension != 3U) ||
      (input->field == SS_PMG_FIELD_MECHANICS && input->hydraulic_pure_neumann != 0) ||
      (input->field == SS_PMG_FIELD_HYDRAULIC && input->hydraulic_pure_neumann != 0 &&
       input->hydraulic_pure_neumann != 1) ||
      !ss_is_finite(input->dependence_tolerance) ||
      input->dependence_tolerance <= 0.0 ||
      (input->owned_node_count != 0U &&
       (input->coordinates == NULL || input->global_dof_indices == NULL)) ||
      (input->owned_node_count > SIZE_MAX / (size_t)input->dimension) ||
      (input->owned_node_count > SIZE_MAX / components)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (coordinate = 0U; coordinate < input->owned_node_count * (size_t)input->dimension;
       ++coordinate) {
    if (!ss_is_finite(input->coordinates[coordinate])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  return SS_OK;
}

static PetscScalar ss_pmg_rigid_value(uint32_t dimension, size_t mode, size_t component,
                                      const PetscReal *coordinate) {
  if (mode < (size_t)dimension) {
    return mode == component ? 1.0 : 0.0;
  }
  if (dimension == 2U) {
    return component == 0U ? (PetscScalar)(-coordinate[1]) : (PetscScalar)coordinate[0];
  }
  switch (mode - 3U) {
  case 0U:
    if (component == 1U) {
      return (PetscScalar)(-coordinate[2]);
    }
    return component == 2U ? (PetscScalar)coordinate[1] : 0.0;
  case 1U:
    if (component == 0U) {
      return (PetscScalar)coordinate[2];
    }
    return component == 2U ? (PetscScalar)(-coordinate[0]) : 0.0;
  case 2U:
    if (component == 0U) {
      return (PetscScalar)(-coordinate[1]);
    }
    return component == 1U ? (PetscScalar)coordinate[0] : 0.0;
  default:
    return 0.0;
  }
}

SsStatus ss_pmg_near_nullspace_create(Mat operator_handle,
                                      const SsPmgNearNullspaceInput *input,
                                      MatNullSpace *near_nullspace,
                                      SsPmgNullspaceRecord *record) {
  MPI_Comm communicator;
  PetscInt global_rows = 0;
  PetscInt global_columns = 0;
  PetscInt owned_start = 0, owned_end = 0;
  Vec raw[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
  Vec retained[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
  size_t raw_count;
  size_t retained_count = 0U;
  size_t mode;
  const size_t components = input == NULL ? 0U : ss_pmg_nullspace_components(input);
  SsStatus status;
  if (operator_handle == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  communicator = PetscObjectComm((PetscObject)operator_handle);
  status = near_nullspace == NULL || record == NULL
               ? SS_ERROR_INVALID_ARGUMENT
               : ss_pmg_nullspace_input_validate(input);
  if (status == SS_OK &&
      (MatGetSize(operator_handle, &global_rows, &global_columns) != PETSC_SUCCESS ||
       MatGetOwnershipRangeColumn(operator_handle, &owned_start, &owned_end) !=
           PETSC_SUCCESS ||
       global_rows <= 0 || global_rows != global_columns)) {
    status = SS_ERROR_TYPE_MISMATCH;
  }
  for (size_t index = 0U;
       status == SS_OK && index < input->owned_node_count * components; ++index) {
    const PetscInt dof = input->global_dof_indices[index];
    if (dof >= 0 && (dof < owned_start || dof >= owned_end)) {
      status = SS_ERROR_INVALID_ARGUMENT;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    return status;
  }
  *near_nullspace = NULL;
  memset(record, 0, sizeof(*record));
  raw_count = ss_pmg_nullspace_raw_modes(input);
  record->field = input->field;
  record->raw_mode_count = raw_count;
  if (raw_count == 0U) {
    return SS_OK;
  }
  for (mode = 0U; mode < raw_count; ++mode) {
    size_t node;
    PetscReal raw_norm = 0.0;
    PetscReal residual_norm = 0.0;
    PetscScalar *values = NULL;
    if (MatCreateVecs(operator_handle, &raw[mode], NULL) != PETSC_SUCCESS) {
      status = SS_ERROR_OUT_OF_MEMORY;
    } else if (VecGetArray(raw[mode], &values) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      for (PetscInt offset = 0; offset < owned_end - owned_start; ++offset) {
        values[offset] = 0.0;
      }
      for (node = 0U; node < input->owned_node_count; ++node) {
        for (size_t component = 0U; component < components; ++component) {
          const PetscInt dof = input->global_dof_indices[node * components + component];
          if (dof >= 0) {
            values[dof - owned_start] =
                input->field == SS_PMG_FIELD_HYDRAULIC
                    ? 1.0
                    : ss_pmg_rigid_value(
                          input->dimension, mode, component,
                          &input->coordinates[node * (size_t)input->dimension]);
          }
        }
      }
      if (VecRestoreArray(raw[mode], &values) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
    }
    /* Only resident entries are written, including an empty array on ranks
     * with no coarse rows. Array restore invalidates the norm on every rank. */
    status = ss_status_allreduce_max(communicator, status);
    if (status != SS_OK) {
      goto cleanup;
    }
    status = ss_status_allreduce_max(
        communicator, VecNorm(raw[mode], NORM_2, &raw_norm) == PETSC_SUCCESS
                          ? SS_OK
                          : SS_ERROR_INTERNAL);
    if (status != SS_OK) {
      goto cleanup;
    }
    {
      size_t pass;
      for (pass = 0U; pass < 2U; ++pass) {
        size_t previous;
        for (previous = 0U; previous < retained_count; ++previous) {
          PetscScalar coefficient = 0.0;
          status = ss_status_allreduce_max(
              communicator,
              VecDot(retained[previous], raw[mode], &coefficient) == PETSC_SUCCESS &&
                      VecAXPY(raw[mode], -coefficient, retained[previous]) ==
                          PETSC_SUCCESS
                  ? SS_OK
                  : SS_ERROR_INTERNAL);
          if (status != SS_OK) {
            goto cleanup;
          }
        }
      }
    }
    status = ss_status_allreduce_max(
        communicator, VecNorm(raw[mode], NORM_2, &residual_norm) == PETSC_SUCCESS
                          ? SS_OK
                          : SS_ERROR_INTERNAL);
    if (status != SS_OK) {
      goto cleanup;
    }
    if (residual_norm <=
        input->dependence_tolerance * (raw_norm > 1.0 ? raw_norm : 1.0)) {
      status = ss_status_allreduce_max(
          communicator,
          VecDestroy(&raw[mode]) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL);
      if (status != SS_OK) {
        goto cleanup;
      }
      ++record->dropped_mode_count;
      continue;
    }
    status = ss_status_allreduce_max(
        communicator, VecScale(raw[mode], 1.0 / residual_norm) == PETSC_SUCCESS
                          ? SS_OK
                          : SS_ERROR_INTERNAL);
    if (status != SS_OK) {
      goto cleanup;
    }
    retained[retained_count] = raw[mode];
    raw[mode] = NULL;
    ++retained_count;
  }
  for (mode = 0U; mode < retained_count; ++mode) {
    size_t other;
    for (other = 0U; other < retained_count; ++other) {
      PetscScalar dot = 0.0;
      PetscReal defect;
      status = ss_status_allreduce_max(
          communicator, VecDot(retained[mode], retained[other], &dot) == PETSC_SUCCESS
                            ? SS_OK
                            : SS_ERROR_INTERNAL);
      if (status != SS_OK) {
        goto cleanup;
      }
      defect = PetscAbsScalar(dot - (mode == other ? 1.0 : 0.0));
      if (defect > record->orthogonality_defect) {
        record->orthogonality_defect = defect;
      }
    }
  }
  if (MatNullSpaceCreate(communicator, PETSC_FALSE, (PetscInt)retained_count, retained,
                         near_nullspace) != PETSC_SUCCESS ||
      PetscObjectSetName((PetscObject)*near_nullspace,
                         input->field == SS_PMG_FIELD_MECHANICS
                             ? "pmg-mechanics-near-nullspace"
                             : "pmg-hydraulic-near-nullspace") != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  record->retained_mode_count = retained_count;

cleanup:
  status = ss_status_allreduce_max(communicator, status);
  for (mode = 0U; mode < 6U; ++mode) {
    (void)VecDestroy(&raw[mode]);
    (void)VecDestroy(&retained[mode]);
  }
  if (status != SS_OK) {
    (void)MatNullSpaceDestroy(near_nullspace);
  }
  return status;
}

SsStatus ss_pmg_nullspace_storage_copy(SsPmgNearNullspaceStorage *storage,
                                       const SsPmgNearNullspaceInput *source) {
  size_t coordinate_count;
  size_t dof_count;
  size_t components;
  SsStatus status;
  memset(storage, 0, sizeof(*storage));
  if (source == NULL || source->field == SS_PMG_FIELD_NONE) {
    storage->input.field = SS_PMG_FIELD_NONE;
    return SS_OK;
  }
  status = ss_pmg_nullspace_input_validate(source);
  if (status != SS_OK) {
    return status;
  }
  components = ss_pmg_nullspace_components(source);
  coordinate_count = source->owned_node_count * (size_t)source->dimension;
  dof_count = source->owned_node_count * components;
  if (coordinate_count != 0U) {
    storage->coordinates =
        (PetscReal *)malloc(coordinate_count * sizeof(*storage->coordinates));
    storage->global_dof_indices =
        (PetscInt *)malloc(dof_count * sizeof(*storage->global_dof_indices));
    if (storage->coordinates == NULL || storage->global_dof_indices == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
    memcpy(storage->coordinates, source->coordinates,
           coordinate_count * sizeof(*storage->coordinates));
    memcpy(storage->global_dof_indices, source->global_dof_indices,
           dof_count * sizeof(*storage->global_dof_indices));
  }
  storage->input = *source;
  storage->input.coordinates = storage->coordinates;
  storage->input.global_dof_indices = storage->global_dof_indices;
  return SS_OK;
}

SsStatus ss_pmg_active_near_nullspace_create(SsPmg *pmg, size_t level_index) {
  SsPmgLevel *level = &pmg->levels[level_index];
  PetscBool has_constant = PETSC_FALSE;
  PetscInt vector_count = 0;
  const Vec *world_vectors = NULL;
  Vec active_vectors[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
  PetscInt vector;
  SsStatus status = SS_OK;
  if (level->world_near_nullspace == NULL || level->identity_world_layout != 0) {
    return SS_OK;
  }
  if (MatNullSpaceGetVecs(level->world_near_nullspace, &has_constant, &vector_count,
                          &world_vectors) != PETSC_SUCCESS ||
      vector_count < 0 || vector_count > 6) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  for (vector = 0; vector < vector_count; ++vector) {
    if (level->local_active != 0 &&
        MatCreateVecs(level->active_operator, &active_vectors[vector], NULL) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status != SS_OK) {
      goto cleanup;
    }
    status = ss_status_allreduce_max(
        pmg->communicator,
        ss_pmg_bridge_to_active(pmg, level_index, world_vectors[vector],
                                active_vectors[vector], 0));
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  if (level->local_active != 0) {
    if (MatNullSpaceCreate(level->active_communicator, has_constant, vector_count,
                           active_vectors,
                           &level->active_near_nullspace) != PETSC_SUCCESS ||
        MatSetNearNullSpace(level->active_operator, level->active_near_nullspace) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(pmg->communicator, status);

cleanup:
  for (vector = 0; vector < 6; ++vector) {
    (void)VecDestroy(&active_vectors[vector]);
  }
  return status;
}

SsStatus ss_pmg_near_nullspaces_build(SsPmg *pmg) {
  size_t level;
  for (level = 0U; level < pmg->hierarchy.level_count; ++level) {
    SsPmgNearNullspaceStorage *storage = &pmg->nullspace_inputs[level];
    SsStatus status = SS_OK;
    if (storage->input.field == SS_PMG_FIELD_NONE) {
      PetscInt block_count = 0;
      const PetscInt *block_sizes = NULL;
      /* External diagnostic hierarchies may carry a valid nodal partition on
       * the matrix without owning coordinate data for a second nullspace
       * construction.  Preserve that partition so VPBJacobi does not silently
       * degrade to point Jacobi. */
      if (MatGetVariableBlockSizes(pmg->operators[level], &block_count, &block_sizes) !=
              PETSC_SUCCESS ||
          block_count < 0 || (block_count != 0 && block_sizes == NULL)) {
        status = SS_ERROR_INTERNAL;
      }
      pmg->levels[level].variable_point_block_count = (size_t)block_count;
      status = ss_status_allreduce_max(pmg->communicator, status);
      if (status != SS_OK) {
        return status;
      }
      continue;
    }
    status = ss_status_allreduce_max(pmg->communicator,
                                     ss_pmg_variable_point_blocks_set(pmg, level));
    if (status == SS_OK) {
      status = ss_pmg_near_nullspace_create(pmg->operators[level], &storage->input,
                                            &pmg->levels[level].world_near_nullspace,
                                            &pmg->levels[level].nullspace_record);
    }
    if (status != SS_OK) {
      return status;
    }
    if (pmg->levels[level].world_near_nullspace != NULL &&
        MatSetNearNullSpace(pmg->operators[level],
                            pmg->levels[level].world_near_nullspace) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}
