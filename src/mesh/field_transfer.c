/* Field lifting, broadcast and inter-order vector transfer.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"

#include "slope_internal/fe/fe.h"

#include <petscdm.h>
#include <petscsection.h>
#include <petscsf.h>
#include <petscvec.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

SsStatus ss_field_space_broadcast_owned_cell_values(
    const SsFieldSpace *space, size_t values_per_cell, const SsEntityId *owned_cell_ids,
    const double *owned_values, size_t owned_cell_count, SsEntityId *incident_cell_ids,
    double *incident_values, size_t incident_cell_count) {
  PetscSection section = NULL;
  PetscSF point_sf = NULL;
  PetscSF value_sf = NULL;
  PetscInt *remote_offsets = NULL;
  double *values = NULL;
  PetscInt storage_size = 0;
  size_t entry_index;
  size_t cell_index;
  SsStatus local_status = SS_OK;
  SsStatus status;
  if (space == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (values_per_cell == 0U || values_per_cell > (size_t)PETSC_MAX_INT ||
      owned_cell_count != space->owned_cell_plan_count ||
      incident_cell_count != space->incident_cell_plan_count ||
      (owned_cell_count != 0U && (owned_cell_ids == NULL || owned_values == NULL)) ||
      (incident_cell_count != 0U &&
       (incident_cell_ids == NULL || incident_values == NULL)) ||
      owned_cell_count > SIZE_MAX / values_per_cell ||
      incident_cell_count > SIZE_MAX / values_per_cell) {
    local_status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, local_status);
  if (status != SS_OK) {
    return status;
  }
  if (PetscSectionCreate(space->mesh->communicator, &section) != PETSC_SUCCESS ||
      PetscSectionSetChart(section, space->mesh->cache_chart_start,
                           space->mesh->cache_chart_end) != PETSC_SUCCESS) {
    local_status = SS_ERROR_INTERNAL;
  }
  for (entry_index = 0U;
       local_status == SS_OK && entry_index < space->mesh->point_count; ++entry_index) {
    const SsPointEntry *entry = &space->mesh->points[entry_index];
    if (entry->kind == SS_MESH_ENTITY_CELL &&
        PetscSectionSetDof(section, entry->point, (PetscInt)values_per_cell) !=
            PETSC_SUCCESS) {
      local_status = SS_ERROR_INTERNAL;
    }
  }
  if (local_status == SS_OK &&
      (PetscSectionSetUp(section) != PETSC_SUCCESS ||
       PetscSectionGetStorageSize(section, &storage_size) != PETSC_SUCCESS ||
       storage_size < 0)) {
    local_status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  values =
      (double *)calloc(storage_size == 0 ? 1U : (size_t)storage_size, sizeof(*values));
  local_status = values == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  status = ss_status_allreduce_max(space->mesh->communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (cell_index = 0U; local_status == SS_OK && cell_index < owned_cell_count;
       ++cell_index) {
    const SsCellPlan *cell =
        ss_field_owned_cell_plan_find(space, owned_cell_ids[cell_index]);
    PetscInt point = -1;
    PetscInt offset = -1;
    if (cell == NULL || cell->cell_owned == 0U ||
        (cell_index != 0U &&
         owned_cell_ids[cell_index - 1U] >= owned_cell_ids[cell_index]) ||
        ss_mesh_cell_point(space->mesh, cell->stable_cell_id, &point) != SS_OK ||
        PetscSectionGetOffset(section, point, &offset) != PETSC_SUCCESS || offset < 0 ||
        (size_t)offset > (size_t)storage_size ||
        values_per_cell > (size_t)storage_size - (size_t)offset) {
      local_status = SS_ERROR_INTEGRITY;
      break;
    }
    memcpy(values + (size_t)offset, owned_values + cell_index * values_per_cell,
           values_per_cell * sizeof(*values));
  }
  status = ss_status_allreduce_max(space->mesh->communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (DMGetPointSF(space->mesh->dm, &point_sf) != PETSC_SUCCESS || point_sf == NULL ||
      PetscSFCreateRemoteOffsets(point_sf, section, section, &remote_offsets) !=
          PETSC_SUCCESS ||
      PetscSFCreateSectionSF(point_sf, section, remote_offsets, section, &value_sf) !=
          PETSC_SUCCESS) {
    local_status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (PetscSFBcastBegin(value_sf, MPI_DOUBLE, values, values, MPI_REPLACE) !=
          PETSC_SUCCESS ||
      PetscSFBcastEnd(value_sf, MPI_DOUBLE, values, values, MPI_REPLACE) !=
          PETSC_SUCCESS) {
    local_status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (cell_index = 0U; local_status == SS_OK && cell_index < incident_cell_count;
       ++cell_index) {
    const SsCellPlan *cell = &space->incident_cell_plans[cell_index];
    PetscInt point = -1;
    PetscInt offset = -1;
    if (ss_mesh_cell_point(space->mesh, cell->stable_cell_id, &point) != SS_OK ||
        PetscSectionGetOffset(section, point, &offset) != PETSC_SUCCESS || offset < 0 ||
        (size_t)offset > (size_t)storage_size ||
        values_per_cell > (size_t)storage_size - (size_t)offset) {
      local_status = SS_ERROR_INTEGRITY;
      break;
    }
    incident_cell_ids[cell_index] = cell->stable_cell_id;
    memcpy(incident_values + cell_index * values_per_cell, values + (size_t)offset,
           values_per_cell * sizeof(*incident_values));
  }
  status = ss_status_allreduce_max(space->mesh->communicator, local_status);

cleanup:
  if (PetscSFDestroy(&value_sf) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (PetscFree(remote_offsets) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (PetscSectionDestroy(&section) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  free(values);
  return status;
}

SsStatus ss_field_space_apply_cell_lifting(
    const SsFieldSpace *space, SsEntityId stable_cell_id, const double *element_matrix,
    const double *element_rhs, const double *trial_values, size_t closure_count,
    double *complete_values, size_t *free_slots, double *free_rhs, size_t *free_count) {
  const SsCellPlan *plan = ss_field_owned_cell_plan_find(space, stable_cell_id);
  size_t expected_count;
  size_t node;
  size_t free_index = 0U;
  if (space == NULL || element_matrix == NULL || element_rhs == NULL ||
      trial_values == NULL || complete_values == NULL || free_slots == NULL ||
      free_rhs == NULL || free_count == NULL || plan == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  expected_count = plan->closure_count;
  if (closure_count != expected_count || expected_count > SS_FIELD_MAX_CLOSURE) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (node = 0U; node < closure_count; ++node) {
    complete_values[node] = plan->constrained[node] == 0U
                                ? trial_values[node]
                                : plan->prescribed_values[node];
  }
  for (node = 0U; node < closure_count; ++node) {
    if (plan->constrained[node] == 0U) {
      size_t column;
      double lifted = element_rhs[node];
      for (column = 0U; column < closure_count; ++column) {
        if (plan->constrained[column] != 0U) {
          lifted -= element_matrix[node * closure_count + column] *
                    plan->prescribed_values[column];
        }
      }
      free_slots[free_index] = node;
      free_rhs[free_index] = lifted;
      ++free_index;
    }
  }
  *free_count = free_index;
  return SS_OK;
}

SsStatus ss_field_space_create_global_vector(const SsFieldSpace *space, Vec *vector) {
  PetscInt local_size = 0;
  PetscInt global_size = 0;
  SsStatus status = SS_OK;
  if (space == NULL || vector == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *vector = NULL;
  if (ss_checked_u64_to_petsc(space->summary.owned_free_dof_count, &local_size) !=
          SS_OK ||
      ss_checked_u64_to_petsc(space->summary.global_free_dof_count, &global_size) !=
          SS_OK) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (VecCreate(space->mesh->communicator, vector) != PETSC_SUCCESS ||
      VecSetSizes(*vector, local_size, global_size) != PETSC_SUCCESS ||
      VecSetFromOptions(*vector) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    (void)VecDestroy(vector);
    return status;
  }
  if (PetscObjectSetName((PetscObject)*vector, "field-free-values") != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    (void)VecDestroy(vector);
  }
  return status;
}

SsStatus ss_field_space_injection_create(const SsFieldSpace *coarse,
                                         const SsFieldSpace *fine, Mat *injection) {
  Mat created = NULL;
  PetscInt local_rows = 0;
  PetscInt global_rows = 0;
  PetscInt local_columns = 0;
  PetscInt global_columns = 0;
  PetscInt *rows = NULL;
  PetscInt *columns = NULL;
  PetscScalar *values = NULL;
  unsigned char *seen = NULL;
  size_t plan_index;
  SsStatus status = SS_OK;
  if (coarse == NULL || fine == NULL || injection == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *injection = NULL;
  if (coarse->mesh != fine->mesh ||
      coarse->summary.components != fine->summary.components ||
      coarse->summary.order >= fine->summary.order ||
      ss_checked_u64_to_petsc(coarse->summary.owned_free_dof_count, &local_rows) !=
          SS_OK ||
      ss_checked_u64_to_petsc(coarse->summary.global_free_dof_count, &global_rows) !=
          SS_OK ||
      ss_checked_u64_to_petsc(fine->summary.owned_free_dof_count, &local_columns) !=
          SS_OK ||
      ss_checked_u64_to_petsc(fine->summary.global_free_dof_count, &global_columns) !=
          SS_OK) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  if (status == SS_OK && local_rows != 0) {
    rows = (PetscInt *)calloc((size_t)local_rows, sizeof(*rows));
    columns = (PetscInt *)calloc((size_t)local_rows, sizeof(*columns));
    values = (PetscScalar *)calloc((size_t)local_rows, sizeof(*values));
    seen = (unsigned char *)calloc((size_t)local_rows, sizeof(*seen));
    if (rows == NULL || columns == NULL || values == NULL || seen == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(coarse->mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (status == SS_OK &&
      (MatCreate(coarse->mesh->communicator, &created) != PETSC_SUCCESS ||
       MatSetSizes(created, local_rows, local_columns, global_rows, global_columns) !=
           PETSC_SUCCESS ||
       MatSetType(created, MATAIJ) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    PetscInt row_start = 0;
    PetscInt row_end = 0;
    if (MatGetOwnershipRange(created, &row_start, &row_end) != PETSC_SUCCESS ||
        row_end - row_start != local_rows) {
      status = SS_ERROR_INTERNAL;
    } else {
      if (local_rows != 0) {
        memset(seen, 0, (size_t)local_rows * sizeof(*seen));
      }
      for (plan_index = 0U;
           status == SS_OK && plan_index < coarse->owned_cell_plan_count;
           ++plan_index) {
        const SsCellPlan *coarse_pack = &coarse->owned_cell_plans[plan_index];
        const SsCellPlan *fine_pack =
            ss_field_incident_cell_plan_find(fine, coarse_pack->stable_cell_id);
        size_t coarse_slot;
        if (fine_pack == NULL) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        for (coarse_slot = 0U; coarse_slot < coarse_pack->closure_count;
             ++coarse_slot) {
          const PetscInt coarse_global = coarse_pack->global_indices[coarse_slot];
          PetscInt slot;
          size_t fine_slot;
          if (coarse_pack->owned[coarse_slot] == 0U ||
              coarse_pack->constrained[coarse_slot] != 0U) {
            continue;
          }
          slot = coarse_global - row_start;
          if (slot < 0 || slot >= local_rows) {
            status = SS_ERROR_INTEGRITY;
            break;
          }
          if (seen[slot] != 0U) {
            continue;
          }
          for (fine_slot = 0U; fine_slot < fine_pack->closure_count; ++fine_slot) {
            if (fine_pack->field_entity_ids[fine_slot] ==
                    coarse_pack->field_entity_ids[coarse_slot] &&
                fine_pack->field_entity_slots[fine_slot] ==
                    coarse_pack->field_entity_slots[coarse_slot] &&
                fine_pack->field_components[fine_slot] ==
                    coarse_pack->field_components[coarse_slot]) {
              break;
            }
          }
          if (fine_slot == fine_pack->closure_count ||
              fine_pack->constrained[fine_slot] != 0U ||
              fine_pack->global_indices[fine_slot] < 0) {
            status = SS_ERROR_INTEGRITY;
            break;
          }
          rows[slot] = coarse_global;
          columns[slot] = fine_pack->global_indices[fine_slot];
          values[slot] = 1.0;
          seen[slot] = 1U;
        }
      }
      for (PetscInt slot = 0; status == SS_OK && slot < local_rows; ++slot) {
        if (seen[slot] == 0U) {
          status = SS_ERROR_INTEGRITY;
        }
      }
    }
  }
  status = ss_status_allreduce_max(coarse->mesh->communicator, status);
  if (status == SS_OK &&
      (MatSetPreallocationCOO(created, (PetscCount)local_rows, rows, columns) !=
           PETSC_SUCCESS ||
       MatSetValuesCOO(created, values, INSERT_VALUES) != PETSC_SUCCESS ||
       PetscObjectSetName((PetscObject)created, "field-nodal-injection") !=
           PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(coarse->mesh->communicator, status);
  if (status == SS_OK) {
    *injection = created;
    created = NULL;
  }

cleanup:
  free(seen);
  free(values);
  free(columns);
  free(rows);
  (void)MatDestroy(&created);
  return status;
}

SsStatus ss_field_space_transfer_create(const SsFieldSpace *coarse,
                                        const SsFieldSpace *fine, Mat *prolongation) {
  double transfer[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
  unsigned char *seen = NULL;
  Mat created = NULL;
  PetscInt local_rows = 0;
  PetscInt global_rows = 0;
  PetscInt local_columns = 0;
  PetscInt global_columns = 0;
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  size_t transfer_rows = 0U;
  size_t transfer_columns = 0U;
  size_t plan_index;
  SsStatus status = SS_OK;
  if (coarse == NULL || fine == NULL || prolongation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *prolongation = NULL;
  if (coarse->mesh != fine->mesh ||
      coarse->summary.components != fine->summary.components ||
      coarse->summary.order >= fine->summary.order ||
      fine->summary.order % coarse->summary.order != 0U) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  if (status == SS_OK &&
      (ss_checked_u64_to_petsc(fine->summary.owned_free_dof_count, &local_rows) !=
           SS_OK ||
       ss_checked_u64_to_petsc(fine->summary.global_free_dof_count, &global_rows) !=
           SS_OK ||
       ss_checked_u64_to_petsc(coarse->summary.owned_free_dof_count, &local_columns) !=
           SS_OK ||
       ss_checked_u64_to_petsc(coarse->summary.global_free_dof_count,
                               &global_columns) != SS_OK ||
       ss_fe_transfer_build(coarse->descriptor, fine->descriptor, transfer,
                            SS_FE_MAX_NODES * SS_FE_MAX_NODES, &transfer_rows,
                            &transfer_columns) != SS_OK)) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK && (transfer_rows != fine->descriptor->node_count ||
                          transfer_columns != coarse->descriptor->node_count)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    seen = (unsigned char *)calloc(local_rows == 0 ? 1U : (size_t)local_rows,
                                   sizeof(*seen));
    if (seen == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(fine->mesh->communicator, status);
  if (status != SS_OK) {
    free(seen);
    return status;
  }
  if (MatCreate(fine->mesh->communicator, &created) != PETSC_SUCCESS ||
      MatSetSizes(created, local_rows, local_columns, global_rows, global_columns) !=
          PETSC_SUCCESS ||
      MatSetFromOptions(created) != PETSC_SUCCESS ||
      (fine->mesh->size == 1
           ? MatSeqAIJSetPreallocation(created,
                                       (PetscInt)coarse->descriptor->node_count, NULL)
           : MatMPIAIJSetPreallocation(
                 created, (PetscInt)coarse->descriptor->node_count, NULL,
                 (PetscInt)coarse->descriptor->node_count, NULL)) != PETSC_SUCCESS ||
      MatSetOption(created, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) !=
          PETSC_SUCCESS ||
      MatGetOwnershipRange(created, &row_start, &row_end) != PETSC_SUCCESS ||
      row_end - row_start != local_rows) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(fine->mesh->communicator, status);
  if (status != SS_OK) {
    free(seen);
    (void)MatDestroy(&created);
    return status;
  }
  for (plan_index = 0U; plan_index < fine->owned_cell_plan_count; ++plan_index) {
    const SsCellPlan *fine_pack = &fine->owned_cell_plans[plan_index];
    const SsCellPlan *coarse_pack =
        ss_field_owned_cell_plan_find(coarse, fine_pack->stable_cell_id);
    size_t fine_node;
    if (coarse_pack == NULL) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (fine_node = 0U; fine_node < fine->descriptor->node_count; ++fine_node) {
      uint32_t component;
      for (component = 0U; component < fine->summary.components; ++component) {
        const size_t fine_closure =
            fine_node * (size_t)fine->summary.components + component;
        const PetscInt row = fine_pack->global_indices[fine_closure];
        PetscInt columns[SS_FE_MAX_NODES];
        PetscScalar values[SS_FE_MAX_NODES];
        PetscInt column_count = 0;
        size_t coarse_node;
        if (fine_pack->constrained[fine_closure] != 0U ||
            fine_pack->owned[fine_closure] == 0U) {
          continue;
        }
        if (row < row_start || row >= row_end) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        if (seen[row - row_start] != 0U) {
          continue;
        }
        for (coarse_node = 0U; coarse_node < coarse->descriptor->node_count;
             ++coarse_node) {
          const size_t coarse_closure =
              coarse_node * (size_t)coarse->summary.components + component;
          const double value = transfer[fine_node * transfer_columns + coarse_node];
          if (coarse_pack->constrained[coarse_closure] == 0U && value != 0.0) {
            columns[column_count] = coarse_pack->global_indices[coarse_closure];
            values[column_count] = (PetscScalar)value;
            ++column_count;
          }
        }
        /*
         * Constraint unions can eliminate every coarse basis function that
         * supports an otherwise free high-order DOF.  The correct
         * free-space prolongation row is then identically zero: the fine
         * smoother owns that mode and coarse correction contributes nothing.
         */
        if (column_count != 0 && MatSetValues(created, 1, &row, column_count, columns,
                                              values, INSERT_VALUES) != PETSC_SUCCESS) {
          status = SS_ERROR_INTERNAL;
          break;
        }
        seen[row - row_start] = 1U;
      }
      if (status != SS_OK) {
        break;
      }
    }
    if (status != SS_OK) {
      break;
    }
  }
  if (status == SS_OK) {
    PetscInt row;
    for (row = 0; row < local_rows; ++row) {
      if (seen[row] == 0U) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
    }
  }
  status = ss_status_allreduce_max(fine->mesh->communicator, status);
  if (status == SS_OK) {
    if (MatAssemblyBegin(created, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    if (MatAssemblyEnd(created, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(fine->mesh->communicator, status);
  free(seen);
  if (status != SS_OK) {
    (void)MatDestroy(&created);
    return status;
  }
  if (PetscObjectSetName((PetscObject)created, "field-prolongation") != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(fine->mesh->communicator, status);
  if (status != SS_OK) {
    (void)MatDestroy(&created);
    return status;
  }
  *prolongation = created;
  return SS_OK;
}
