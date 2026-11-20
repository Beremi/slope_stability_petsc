/* Size and allocate workspaces for assembled and matrix-free operators.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "operator_internal.h"

#include "slope_internal/build/config.h"

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <stdlib.h>

SsStatus ss_operator_checked_bytes(size_t count, size_t item_size, uint64_t *bytes) {
  if (bytes == NULL || (item_size != 0U && count > UINT64_MAX / item_size)) {
    return SS_ERROR_OVERFLOW;
  }
  *bytes = (uint64_t)count * (uint64_t)item_size;
  return SS_OK;
}

SsStatus ss_operator_mechanics_block_size(const SsOperator *operator_handle, size_t row,
                                          size_t *block_size) {
  const SsOwnedRowPlan *first;
  uint32_t current_slot;
  uint32_t previous_component;
  size_t node_count = 1U;
  size_t size = 1U;
  const size_t node_cap =
      operator_handle == NULL ? 0U : operator_handle->matrix_free_block_node_cap;
  if (operator_handle == NULL || block_size == NULL ||
      row >= operator_handle->plan->row_count || node_cap == 0U ||
      node_cap > SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_NODES) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  first = &operator_handle->plan->rows[row];
  current_slot = first->field_entity_slot;
  previous_component = first->component;
  if (first->global_row != operator_handle->plan->row_start + (PetscInt)row) {
    return SS_ERROR_INTEGRITY;
  }
  while (row + size < operator_handle->plan->row_count) {
    const SsOwnedRowPlan *candidate = &operator_handle->plan->rows[row + size];
    if (candidate->field_entity_id != first->field_entity_id) {
      break;
    }
    if (candidate->field_entity_slot != current_slot) {
      if (node_count == node_cap) {
        break;
      }
      ++node_count;
      current_slot = candidate->field_entity_slot;
      previous_component = UINT32_MAX;
    }
    if (candidate->global_row !=
            operator_handle->plan->row_start + (PetscInt)(row + size) ||
        (previous_component != UINT32_MAX &&
         candidate->component <= previous_component)) {
      return SS_ERROR_INTEGRITY;
    }
    previous_component = candidate->component;
    ++size;
  }
  if (size == 0U ||
      size > (size_t)operator_handle->plan->field_summary.components * node_cap ||
      size > SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS) {
    return SS_ERROR_INTEGRITY;
  }
  *block_size = size;
  return SS_OK;
}

SsStatus ss_operator_mechanics_block_requirements(const SsOperator *operator_handle,
                                                  size_t *block_count,
                                                  size_t *entry_count) {
  size_t blocks = 0U;
  size_t entries = 0U;
  size_t row = 0U;
  if (operator_handle == NULL || block_count == NULL || entry_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *block_count = 0U;
  *entry_count = 0U;
  while (row < operator_handle->plan->row_count) {
    size_t block_size = 0U;
    const SsStatus block_status =
        ss_operator_mechanics_block_size(operator_handle, row, &block_size);
    if (block_status != SS_OK) {
      return block_status;
    }
    if (block_size > SIZE_MAX / block_size ||
        entries > SIZE_MAX - block_size * block_size || blocks == SIZE_MAX) {
      return SS_ERROR_OVERFLOW;
    }
    entries += block_size * block_size;
    ++blocks;
    row += block_size;
  }
  if (blocks > (size_t)PETSC_MAX_INT) {
    return SS_ERROR_OVERFLOW;
  }
  *block_count = blocks;
  *entry_count = entries;
  return SS_OK;
}

SsStatus ss_operator_matrix_free_block_workspace_requirements(
    const SsOperator *operator_handle, size_t *stride, size_t *entry_count) {
  size_t maximum_points = 0U;
  size_t maximum_strains = 0U;
  size_t maximum_block_dofs;
  size_t cell_index;
  if (operator_handle == NULL || stride == NULL || entry_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *stride = 0U;
  *entry_count = 0U;
  if (operator_handle->spec.matrix_free == 0 ||
      (operator_handle->matrix_free_block_node_cap <= 1U &&
       operator_handle->spec.determinism != SS_ASSEMBLY_PERFORMANCE)) {
    return SS_OK;
  }
  maximum_block_dofs = operator_handle->matrix_free_block_node_cap *
                       (size_t)operator_handle->plan->field_summary.components;
  if (maximum_block_dofs == 0U ||
      maximum_block_dofs > SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS) {
    return SS_ERROR_INTEGRITY;
  }
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    const SsOperatorCell *metadata = &operator_handle->cells[cell_index];
    const size_t strains = cell->dimension == 2U ? 3U : 6U;
    if (metadata->quadrature->point_count > maximum_points) {
      maximum_points = metadata->quadrature->point_count;
    }
    if (strains > maximum_strains) {
      maximum_strains = strains;
    }
  }
  if (maximum_points == 0U || maximum_strains == 0U ||
      maximum_points > SIZE_MAX / maximum_strains ||
      maximum_points * maximum_strains > SIZE_MAX / maximum_block_dofs ||
      maximum_points * maximum_strains * maximum_block_dofs > SIZE_MAX / 2U) {
    return SS_ERROR_OVERFLOW;
  }
  *stride = 2U * maximum_points * maximum_strains * maximum_block_dofs;
  if ((size_t)operator_handle->spec.requested_threads > SIZE_MAX / *stride) {
    return SS_ERROR_OVERFLOW;
  }
  *entry_count = (size_t)operator_handle->spec.requested_threads * *stride;
  return SS_OK;
}

SsStatus ss_operator_matrix_free_reserve_block_workspace(SsOperator *operator_handle,
                                                         size_t point_count,
                                                         size_t strain_count) {
  size_t maximum_block_dofs;
  size_t required_stride;
  size_t old_entries;
  size_t required_entries;
  uint64_t old_bytes;
  uint64_t required_bytes;
  uint64_t added_bytes;
  uint64_t memory_cap;
  PetscScalar *resized;
  if (operator_handle == NULL || point_count == 0U || strain_count == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (operator_handle->matrix_free_block_node_cap <= 1U &&
      operator_handle->spec.determinism != SS_ASSEMBLY_PERFORMANCE) {
    return SS_OK;
  }
  maximum_block_dofs = operator_handle->matrix_free_block_node_cap *
                       (size_t)operator_handle->plan->field_summary.components;
  if (maximum_block_dofs == 0U ||
      maximum_block_dofs > SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS ||
      point_count > SIZE_MAX / strain_count ||
      point_count * strain_count > SIZE_MAX / maximum_block_dofs ||
      point_count * strain_count * maximum_block_dofs > SIZE_MAX / 2U) {
    return SS_ERROR_OVERFLOW;
  }
  required_stride = 2U * point_count * strain_count * maximum_block_dofs;
  if (required_stride <= operator_handle->matrix_free_block_workspace_stride) {
    return SS_OK;
  }
  if ((size_t)operator_handle->spec.requested_threads > SIZE_MAX / required_stride ||
      (operator_handle->matrix_free_block_workspace_stride != 0U &&
       (size_t)operator_handle->spec.requested_threads >
           SIZE_MAX / operator_handle->matrix_free_block_workspace_stride)) {
    return SS_ERROR_OVERFLOW;
  }
  old_entries = (size_t)operator_handle->spec.requested_threads *
                operator_handle->matrix_free_block_workspace_stride;
  required_entries = (size_t)operator_handle->spec.requested_threads * required_stride;
  if (ss_operator_checked_bytes(old_entries, sizeof(PetscScalar), &old_bytes) !=
          SS_OK ||
      ss_operator_checked_bytes(required_entries, sizeof(PetscScalar),
                                &required_bytes) != SS_OK ||
      required_bytes < old_bytes) {
    return SS_ERROR_OVERFLOW;
  }
  added_bytes = required_bytes - old_bytes;
  memory_cap = operator_handle->spec.memory_cap_bytes == 0U
                   ? SS_ASSEMBLY_DEFAULT_MEMORY_CAP
                   : (uint64_t)operator_handle->spec.memory_cap_bytes;
  if (operator_handle->diagnostics.admitted_peak_bytes > memory_cap ||
      added_bytes > memory_cap - operator_handle->diagnostics.admitted_peak_bytes) {
    return SS_ERROR_RESOURCE;
  }
  resized = (PetscScalar *)realloc(operator_handle->matrix_free_block_workspace,
                                   (required_entries == 0U ? 1U : required_entries) *
                                       sizeof(*resized));
  if (resized == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  operator_handle->matrix_free_block_workspace = resized;
  operator_handle->matrix_free_block_workspace_stride = required_stride;
  operator_handle->diagnostics.matrix_free_bytes += added_bytes;
  operator_handle->diagnostics.admitted_peak_bytes += added_bytes;
  return SS_OK;
}

SsStatus ss_operator_matrix_free_action_workspace_requirements(
    const SsOperator *operator_handle, size_t *stride, size_t *entry_count) {
  size_t maximum_points = 0U;
  size_t maximum_nodes = 0U;
  size_t cell_index;
  size_t columns;
  size_t stacked_points;
  size_t element_entries;
  size_t quadrature_entries;
  if (operator_handle == NULL || stride == NULL || entry_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *stride = 0U;
  *entry_count = 0U;
  if (operator_handle->spec.matrix_free == 0 ||
      operator_handle->matrix_free_action_batch_cells <= 1U) {
    return SS_OK;
  }
  if (operator_handle->matrix_free_action_batch_cells >
      SS_OPERATOR_MATRIX_FREE_MAX_ACTION_BATCH) {
    return SS_ERROR_INTEGRITY;
  }
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsOperatorCell *metadata = &operator_handle->cells[cell_index];
    if (metadata->quadrature->point_count > maximum_points) {
      maximum_points = metadata->quadrature->point_count;
    }
    if (metadata->field_descriptor->node_count > maximum_nodes) {
      maximum_nodes = metadata->field_descriptor->node_count;
    }
  }
  columns = 3U * operator_handle->matrix_free_action_batch_cells;
  if (maximum_points == 0U || maximum_nodes == 0U || maximum_points > SIZE_MAX / 3U ||
      3U * maximum_points > SIZE_MAX / columns || maximum_nodes > SIZE_MAX / columns) {
    return SS_ERROR_OVERFLOW;
  }
  stacked_points = 3U * maximum_points;
  element_entries = maximum_nodes * columns;
  quadrature_entries = stacked_points * columns;
  /* P-by-(3B) input/output plus (3Q)-by-(3B) derivative/flux pairs. */
  if (element_entries > SIZE_MAX / 2U || quadrature_entries > SIZE_MAX / 2U ||
      2U * element_entries > SIZE_MAX - 2U * quadrature_entries) {
    return SS_ERROR_OVERFLOW;
  }
  *stride = 2U * element_entries + 2U * quadrature_entries;
  if ((size_t)operator_handle->spec.requested_threads > SIZE_MAX / *stride) {
    return SS_ERROR_OVERFLOW;
  }
  *entry_count = (size_t)operator_handle->spec.requested_threads * *stride;
  return SS_OK;
}

SsStatus ss_operator_matrix_free_storage_requirements(const SsOperator *operator_handle,
                                                      size_t *tangent_count,
                                                      size_t *action_count,
                                                      uint64_t *storage_bytes) {
  uint64_t total = 0U;
  uint64_t bytes = 0U;
  size_t tangents = 0U;
  size_t actions;
  size_t block_count = 0U;
  size_t block_entry_count = 0U;
  size_t block_workspace_stride = 0U;
  size_t block_workspace_entries = 0U;
  size_t action_workspace_stride = 0U;
  size_t action_workspace_entries = 0U;
  size_t cell_index;
  SsStatus block_status;
  if (operator_handle == NULL || tangent_count == NULL || action_count == NULL ||
      storage_bytes == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *tangent_count = 0U;
  *action_count = 0U;
  *storage_bytes = 0U;
  if (operator_handle->spec.matrix_free == 0) {
    return SS_OK;
  }
  if (operator_handle->plan->cell_count == SIZE_MAX ||
      (operator_handle->maximum_closure != 0U &&
       operator_handle->plan->cell_count >
           SIZE_MAX / operator_handle->maximum_closure)) {
    return SS_ERROR_OVERFLOW;
  }
  actions = operator_handle->plan->cell_count * operator_handle->maximum_closure;
  block_status = ss_operator_mechanics_block_requirements(operator_handle, &block_count,
                                                          &block_entry_count);
  if (block_status != SS_OK) {
    return block_status;
  }
  block_status = ss_operator_matrix_free_action_workspace_requirements(
      operator_handle, &action_workspace_stride, &action_workspace_entries);
  (void)action_workspace_stride;
  if (block_status != SS_OK) {
    return block_status;
  }
  block_status = ss_operator_matrix_free_block_workspace_requirements(
      operator_handle, &block_workspace_stride, &block_workspace_entries);
  (void)block_workspace_stride;
  if (block_status != SS_OK) {
    return block_status;
  }
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    const SsOperatorCell *metadata = &operator_handle->cells[cell_index];
    const size_t strain_count = cell->dimension == 2U ? 3U : 6U;
    const size_t point_count = metadata->quadrature->point_count;
    size_t cell_tangents;
    if (strain_count > SIZE_MAX / strain_count ||
        point_count > SIZE_MAX / (strain_count * strain_count)) {
      return SS_ERROR_OVERFLOW;
    }
    cell_tangents = point_count * strain_count * strain_count;
    if (tangents > SIZE_MAX - cell_tangents) {
      return SS_ERROR_OVERFLOW;
    }
    tangents += cell_tangents;
  }
#define SS_OPERATOR_ADD_STORAGE(count_, type_)                                         \
  do {                                                                                 \
    if (ss_operator_checked_bytes((count_), sizeof(type_), &bytes) != SS_OK ||         \
        total > UINT64_MAX - bytes) {                                                  \
      return SS_ERROR_OVERFLOW;                                                        \
    }                                                                                  \
    total += bytes;                                                                    \
  } while (0)
  SS_OPERATOR_ADD_STORAGE(operator_handle->plan->cell_count + 1U, size_t);
  SS_OPERATOR_ADD_STORAGE(tangents, double);
  SS_OPERATOR_ADD_STORAGE(actions, double);
  SS_OPERATOR_ADD_STORAGE(operator_handle->plan->row_count, double);
  SS_OPERATOR_ADD_STORAGE(block_count, PetscInt);
  SS_OPERATOR_ADD_STORAGE(block_count + 1U, size_t);
  SS_OPERATOR_ADD_STORAGE(block_count + 1U, size_t);
  SS_OPERATOR_ADD_STORAGE(block_entry_count, PetscScalar);
  SS_OPERATOR_ADD_STORAGE(block_workspace_entries, PetscScalar);
  SS_OPERATOR_ADD_STORAGE(action_workspace_entries, PetscScalar);
#undef SS_OPERATOR_ADD_STORAGE
  *tangent_count = tangents;
  *action_count = actions;
  *storage_bytes = total;
  return SS_OK;
}

SsStatus ss_operator_allocate_work(SsOperator *operator_handle) {
  const int assembled_jacobian = operator_handle->spec.matrix_free == 0;
  const uint64_t memory_cap = operator_handle->spec.memory_cap_bytes == 0U
                                  ? SS_ASSEMBLY_DEFAULT_MEMORY_CAP
                                  : (uint64_t)operator_handle->spec.memory_cap_bytes;
  uint64_t packed_bytes = 0U;
  uint64_t shadow_residual_bytes = 0U;
  uint64_t shadow_matrix_bytes = 0U;
  uint64_t natural_metadata_bytes = 0U;
  uint64_t staging_estimate = 0U;
  uint64_t matrix_free_bytes = 0U;
  uint64_t fixed_bytes;
  uint64_t per_cell_values;
  uint64_t per_cell_bytes;
  uint64_t available;
  uint64_t capacity;
  size_t row;
  size_t matrix_free_tangent_count = 0U;
  size_t matrix_free_action_count = 0U;
  size_t nnz = assembled_jacobian != 0
                   ? (size_t)operator_handle->plan->summary.local_nonzeros
                   : 0U;
  size_t packed_count;
  if ((operator_handle->maximum_closure != 0U &&
       operator_handle->plan->cell_count >
           SIZE_MAX / operator_handle->maximum_closure) ||
      (assembled_jacobian != 0 && operator_handle->maximum_closure != 0U &&
       operator_handle->maximum_closure >
           SIZE_MAX / operator_handle->maximum_closure)) {
    return SS_ERROR_OVERFLOW;
  }
  packed_count = operator_handle->plan->cell_count * operator_handle->maximum_closure;
  operator_handle->matrix_stride =
      assembled_jacobian != 0
          ? operator_handle->maximum_closure * operator_handle->maximum_closure
          : 0U;
  if (ss_operator_matrix_free_storage_requirements(
          operator_handle, &matrix_free_tangent_count, &matrix_free_action_count,
          &matrix_free_bytes) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  (void)matrix_free_tangent_count;
  (void)matrix_free_action_count;
  if (ss_operator_checked_bytes(packed_count, sizeof(double), &packed_bytes) != SS_OK ||
      ss_operator_checked_bytes(operator_handle->plan->row_count, sizeof(double),
                                &shadow_residual_bytes) != SS_OK ||
      ss_operator_checked_bytes(nnz, sizeof(double), &shadow_matrix_bytes) != SS_OK ||
      ss_operator_checked_bytes(operator_handle->natural_facet_count,
                                sizeof(*operator_handle->natural_facets),
                                &natural_metadata_bytes) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  staging_estimate = shadow_residual_bytes;
  if (assembled_jacobian != 0) {
    if (shadow_matrix_bytes > UINT64_MAX - staging_estimate ||
        (uint64_t)nnz > (UINT64_MAX - staging_estimate - shadow_matrix_bytes) /
                            (2U * (uint64_t)sizeof(PetscInt))) {
      return SS_ERROR_OVERFLOW;
    }
    staging_estimate +=
        shadow_matrix_bytes + (uint64_t)nnz * 2U * (uint64_t)sizeof(PetscInt);
  }
  if (operator_handle->plan->summary.plan_bytes > UINT64_MAX - packed_bytes ||
      operator_handle->plan->summary.plan_bytes + packed_bytes >
          UINT64_MAX - shadow_residual_bytes ||
      operator_handle->plan->summary.plan_bytes + packed_bytes + shadow_residual_bytes >
          UINT64_MAX - shadow_matrix_bytes ||
      operator_handle->plan->summary.plan_bytes + packed_bytes + shadow_residual_bytes +
              shadow_matrix_bytes >
          UINT64_MAX - staging_estimate ||
      operator_handle->plan->summary.plan_bytes + packed_bytes + shadow_residual_bytes +
              shadow_matrix_bytes + staging_estimate >
          UINT64_MAX - natural_metadata_bytes ||
      operator_handle->plan->summary.plan_bytes + packed_bytes + shadow_residual_bytes +
              shadow_matrix_bytes + staging_estimate + natural_metadata_bytes >
          UINT64_MAX - operator_handle->cell_cache_bytes ||
      operator_handle->plan->summary.plan_bytes + packed_bytes + shadow_residual_bytes +
              shadow_matrix_bytes + staging_estimate + natural_metadata_bytes +
              operator_handle->cell_cache_bytes >
          UINT64_MAX - matrix_free_bytes) {
    return SS_ERROR_OVERFLOW;
  }
  fixed_bytes = operator_handle->plan->summary.plan_bytes + packed_bytes +
                shadow_residual_bytes + shadow_matrix_bytes + staging_estimate +
                natural_metadata_bytes + operator_handle->cell_cache_bytes +
                matrix_free_bytes;
  per_cell_values = (uint64_t)operator_handle->maximum_closure +
                    (uint64_t)operator_handle->matrix_stride;
  /* Each tile slot also holds its potential and its natural-load work. */
  if (per_cell_values > UINT64_MAX / sizeof(double) ||
      per_cell_values * sizeof(double) >
          UINT64_MAX - sizeof(SsErrorRecord) - 2U * sizeof(double)) {
    return SS_ERROR_OVERFLOW;
  }
  per_cell_bytes =
      per_cell_values * sizeof(double) + sizeof(SsErrorRecord) + 2U * sizeof(double);
  if (memory_cap <= fixed_bytes || memory_cap - fixed_bytes < per_cell_bytes) {
    return SS_ERROR_RESOURCE;
  }
  available = memory_cap - fixed_bytes;
  capacity = available / per_cell_bytes;
  if (capacity > operator_handle->plan->cell_count) {
    capacity = operator_handle->plan->cell_count;
  }
  if (capacity == 0U && operator_handle->plan->cell_count == 0U) {
    capacity = 1U;
  }
  if (capacity == 0U || capacity > SIZE_MAX) {
    return capacity == 0U ? SS_ERROR_RESOURCE : SS_ERROR_OVERFLOW;
  }
  operator_handle->tile_capacity = (size_t)capacity;
  if ((operator_handle->maximum_closure != 0U &&
       operator_handle->tile_capacity > SIZE_MAX / operator_handle->maximum_closure) ||
      (operator_handle->matrix_stride != 0U &&
       operator_handle->tile_capacity > SIZE_MAX / operator_handle->matrix_stride)) {
    return SS_ERROR_OVERFLOW;
  }
  operator_handle->element_residual = (double *)calloc(
      operator_handle->tile_capacity * operator_handle->maximum_closure == 0U
          ? 1U
          : operator_handle->tile_capacity * operator_handle->maximum_closure,
      sizeof(*operator_handle->element_residual));
  operator_handle->element_jacobian = (double *)calloc(
      operator_handle->tile_capacity * operator_handle->matrix_stride == 0U
          ? 1U
          : operator_handle->tile_capacity * operator_handle->matrix_stride,
      sizeof(*operator_handle->element_jacobian));
  operator_handle->element_potential = (double *)calloc(
      operator_handle->tile_capacity, sizeof(*operator_handle->element_potential));
  operator_handle->element_natural_work = (double *)calloc(
      operator_handle->tile_capacity, sizeof(*operator_handle->element_natural_work));
  operator_handle->element_errors = (SsErrorRecord *)calloc(
      operator_handle->tile_capacity, sizeof(*operator_handle->element_errors));
  operator_handle->row_offsets = (size_t *)calloc(
      operator_handle->plan->row_count + 1U, sizeof(*operator_handle->row_offsets));
  operator_handle->shadow_residual = (double *)calloc(
      operator_handle->plan->row_count == 0U ? 1U : operator_handle->plan->row_count,
      sizeof(*operator_handle->shadow_residual));
  operator_handle->shadow_jacobian =
      (double *)calloc(nnz == 0U ? 1U : nnz, sizeof(*operator_handle->shadow_jacobian));
  operator_handle->row_complete = (unsigned char *)calloc(
      operator_handle->plan->row_count == 0U ? 1U : operator_handle->plan->row_count,
      sizeof(*operator_handle->row_complete));
  operator_handle->row_writer_violation = (unsigned char *)calloc(
      operator_handle->plan->row_count == 0U ? 1U : operator_handle->plan->row_count,
      sizeof(*operator_handle->row_writer_violation));
  operator_handle->row_writer = (int *)malloc(
      (operator_handle->plan->row_count == 0U ? 1U : operator_handle->plan->row_count) *
      sizeof(*operator_handle->row_writer));
  operator_handle->worker_cpus =
      (int *)malloc((size_t)operator_handle->spec.requested_threads *
                    sizeof(*operator_handle->worker_cpus));
  if (operator_handle->element_residual == NULL ||
      operator_handle->element_jacobian == NULL ||
      operator_handle->element_potential == NULL ||
      operator_handle->element_natural_work == NULL ||
      operator_handle->element_errors == NULL || operator_handle->row_offsets == NULL ||
      operator_handle->shadow_residual == NULL ||
      operator_handle->shadow_jacobian == NULL ||
      operator_handle->row_complete == NULL ||
      operator_handle->row_writer_violation == NULL ||
      operator_handle->row_writer == NULL || operator_handle->worker_cpus == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (row = 0U; row < operator_handle->plan->row_count; ++row) {
    operator_handle->row_offsets[row + 1U] =
        assembled_jacobian != 0 ? operator_handle->row_offsets[row] +
                                      operator_handle->plan->rows[row].column_count
                                : 0U;
    operator_handle->row_writer[row] = -1;
  }
  for (row = 0U; row < (size_t)operator_handle->spec.requested_threads; ++row) {
    operator_handle->worker_cpus[row] = -1;
  }
  if (operator_handle->row_offsets[operator_handle->plan->row_count] != nnz) {
    return SS_ERROR_INTEGRITY;
  }
  operator_handle->diagnostics.packed_bytes = packed_bytes;
  operator_handle->diagnostics.shadow_bytes =
      shadow_residual_bytes + shadow_matrix_bytes;
  operator_handle->diagnostics.scratch_peak_bytes = capacity * per_cell_bytes;
  operator_handle->diagnostics.matrix_free_bytes = matrix_free_bytes;
  operator_handle->diagnostics.admitted_peak_bytes =
      fixed_bytes + capacity * per_cell_bytes;
  operator_handle->diagnostics.tile_capacity = capacity;
  return operator_handle->diagnostics.admitted_peak_bytes <= memory_cap
             ? SS_OK
             : SS_ERROR_RESOURCE;
}
