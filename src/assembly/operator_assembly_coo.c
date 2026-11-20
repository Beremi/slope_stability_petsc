/* Prepare and accumulate the production COO matrix staging.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "operator_internal.h"

#include "slope_internal/build/config.h"

#include <petsclog.h>
#include <petscvec.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <stdlib.h>

size_t ss_operator_find_row_column(const SsOwnedRowPlan *row, PetscInt global_column) {
  size_t left = 0U;
  size_t right = row->column_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (row->columns[middle] < global_column) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  return left < row->column_count && row->columns[left] == global_column ? left
                                                                         : SIZE_MAX;
}

static size_t ss_operator_free_count(size_t closure_count, const uint8_t *constrained) {
  size_t slot;
  size_t count = 0U;
  for (slot = 0U; slot < closure_count; ++slot) {
    if (constrained[slot] == 0U) {
      ++count;
    }
  }
  return count;
}

SsStatus ss_operator_prepare_performance_coo(SsOperator *operator_handle) {
  const int jacobian_enabled =
      operator_handle != NULL && operator_handle->spec.matrix_free == 0;
  uint64_t memory_cap;
  PetscInt *vector_indices = NULL;
  PetscInt *matrix_rows = NULL;
  PetscInt *matrix_columns = NULL;
  size_t vector_count = 0U;
  size_t matrix_count = 0U;
  size_t vector_offset = 0U;
  size_t matrix_offset = 0U;
  size_t entity_count;
  uint64_t coo_bytes = 0U;
  uint64_t vector_bytes = 0U;
  uint64_t matrix_bytes = 0U;
  uint64_t offset_bytes = 0U;
  size_t entity;
  int local_eligible = 1;
  int globally_eligible = 0;
  SsStatus status = SS_OK;
  if (operator_handle == NULL || operator_handle->plan == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memory_cap = operator_handle->spec.memory_cap_bytes == 0U
                   ? SS_ASSEMBLY_DEFAULT_MEMORY_CAP
                   : (uint64_t)operator_handle->spec.memory_cap_bytes;
  if (operator_handle->staging_residual == NULL ||
      (jacobian_enabled != 0 && operator_handle->staging_jacobian == NULL)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  if (operator_handle->plan->cell_count >
      SIZE_MAX - operator_handle->natural_facet_count) {
    status = SS_ERROR_OVERFLOW;
  }
  entity_count = status == SS_OK ? operator_handle->plan->cell_count +
                                       operator_handle->natural_facet_count
                                 : 0U;
  for (entity = 0U; status == SS_OK && entity < operator_handle->plan->cell_count;
       ++entity) {
    const SsCellPlan *cell = &operator_handle->plan->cells[entity];
    size_t free_count;
    if (cell->cell_owned == 0U) {
      continue;
    }
    free_count = ss_operator_free_count(cell->closure_count, cell->constrained);
    if (vector_count > SIZE_MAX - free_count ||
        (jacobian_enabled != 0 &&
         ((free_count != 0U && free_count > SIZE_MAX / free_count) ||
          matrix_count > SIZE_MAX - free_count * free_count))) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    vector_count += free_count;
    if (jacobian_enabled != 0) {
      matrix_count += free_count * free_count;
    }
  }
  for (entity = 0U; status == SS_OK && entity < operator_handle->natural_facet_count;
       ++entity) {
    const SsFieldFacetPack *facet = &operator_handle->natural_facets[entity].pack;
    size_t free_count;
    if (facet->geometry.facet_owned == 0U) {
      continue;
    }
    free_count = ss_operator_free_count(facet->closure_count, facet->constrained);
    if (vector_count > SIZE_MAX - free_count ||
        (jacobian_enabled != 0 &&
         ((free_count != 0U && free_count > SIZE_MAX / free_count) ||
          matrix_count > SIZE_MAX - free_count * free_count))) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    vector_count += free_count;
    if (jacobian_enabled != 0) {
      matrix_count += free_count * free_count;
    }
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    return status;
  }
  if (vector_count > (size_t)PETSC_MAX_INT || matrix_count > (size_t)PETSC_MAX_INT ||
      entity_count == SIZE_MAX ||
      vector_count > UINT64_MAX / (sizeof(PetscScalar) + sizeof(PetscInt)) ||
      matrix_count > UINT64_MAX / (sizeof(PetscScalar) + 2U * sizeof(PetscInt))) {
    local_eligible = 0;
  }
  if (local_eligible != 0 &&
      entity_count + 1U >
          UINT64_MAX / ((jacobian_enabled != 0 ? 2U : 1U) * sizeof(size_t))) {
    local_eligible = 0;
  }
  if (local_eligible != 0) {
    offset_bytes = (uint64_t)(entity_count + 1U) *
                   (uint64_t)(jacobian_enabled != 0 ? 2U : 1U) *
                   (uint64_t)sizeof(size_t);
    vector_bytes =
        (uint64_t)vector_count * (uint64_t)(sizeof(PetscScalar) + sizeof(PetscInt));
    matrix_bytes = (uint64_t)matrix_count *
                   (uint64_t)(sizeof(PetscScalar) + 2U * sizeof(PetscInt));
    if (vector_bytes > UINT64_MAX - matrix_bytes ||
        vector_bytes + matrix_bytes > UINT64_MAX - offset_bytes) {
      local_eligible = 0;
    } else {
      coo_bytes = vector_bytes + matrix_bytes + offset_bytes;
    }
  }
  if (local_eligible != 0) {
    if (operator_handle->diagnostics.admitted_peak_bytes > memory_cap ||
        coo_bytes > memory_cap - operator_handle->diagnostics.admitted_peak_bytes) {
      local_eligible = 0;
    }
  }
  /*
   * VecSetPreallocationCOO() and MatSetPreallocationCOO() are collective.  A
   * rank-local memory-cap decision must therefore select the COO path only
   * when every rank can enter it; otherwise every rank uses the preallocated
   * insertion fallback.
   */
  if (MPI_Allreduce(&local_eligible, &globally_eligible, 1, MPI_INT, MPI_MIN,
                    operator_handle->plan->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (globally_eligible == 0) {
    return SS_OK;
  }
  vector_indices = (PetscInt *)malloc((vector_count == 0U ? 1U : vector_count) *
                                      sizeof(*vector_indices));
  if (jacobian_enabled != 0) {
    matrix_rows = (PetscInt *)malloc((matrix_count == 0U ? 1U : matrix_count) *
                                     sizeof(*matrix_rows));
    matrix_columns = (PetscInt *)malloc((matrix_count == 0U ? 1U : matrix_count) *
                                        sizeof(*matrix_columns));
  }
  operator_handle->coo_residual_values =
      (PetscScalar *)calloc(vector_count == 0U ? 1U : vector_count,
                            sizeof(*operator_handle->coo_residual_values));
  if (jacobian_enabled != 0) {
    operator_handle->coo_jacobian_values =
        (PetscScalar *)calloc(matrix_count == 0U ? 1U : matrix_count,
                              sizeof(*operator_handle->coo_jacobian_values));
  }
  operator_handle->coo_residual_offsets = (size_t *)calloc(
      entity_count + 1U, sizeof(*operator_handle->coo_residual_offsets));
  if (jacobian_enabled != 0) {
    operator_handle->coo_jacobian_offsets = (size_t *)calloc(
        entity_count + 1U, sizeof(*operator_handle->coo_jacobian_offsets));
  }
  if (vector_indices == NULL ||
      (jacobian_enabled != 0 && (matrix_rows == NULL || matrix_columns == NULL)) ||
      operator_handle->coo_residual_values == NULL ||
      (jacobian_enabled != 0 && operator_handle->coo_jacobian_values == NULL) ||
      operator_handle->coo_residual_offsets == NULL ||
      (jacobian_enabled != 0 && operator_handle->coo_jacobian_offsets == NULL)) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    goto cleanup;
  }
#define SS_OPERATOR_APPEND_COO(pack_)                                                  \
  do {                                                                                 \
    const size_t closure_count_ = (pack_)->closure_count;                              \
    size_t row_;                                                                       \
    for (row_ = 0U; row_ < closure_count_; ++row_) {                                   \
      size_t column_;                                                                  \
      if ((pack_)->constrained[row_] != 0U) {                                          \
        continue;                                                                      \
      }                                                                                \
      vector_indices[vector_offset++] = (pack_)->global_indices[row_];                 \
      if (jacobian_enabled != 0) {                                                     \
        for (column_ = 0U; column_ < closure_count_; ++column_) {                      \
          if ((pack_)->constrained[column_] != 0U) {                                   \
            continue;                                                                  \
          }                                                                            \
          matrix_rows[matrix_offset] = (pack_)->global_indices[row_];                  \
          matrix_columns[matrix_offset] = (pack_)->global_indices[column_];            \
          ++matrix_offset;                                                             \
        }                                                                              \
      }                                                                                \
    }                                                                                  \
  } while (0)
  for (entity = 0U; entity < operator_handle->plan->cell_count; ++entity) {
    const SsCellPlan *cell = &operator_handle->plan->cells[entity];
    operator_handle->coo_residual_offsets[entity] = vector_offset;
    if (jacobian_enabled != 0) {
      operator_handle->coo_jacobian_offsets[entity] = matrix_offset;
    }
    if (cell->cell_owned != 0U) {
      SS_OPERATOR_APPEND_COO(cell);
    }
  }
  for (entity = 0U; entity < operator_handle->natural_facet_count; ++entity) {
    const SsFieldFacetPack *facet = &operator_handle->natural_facets[entity].pack;
    const size_t combined_entity = operator_handle->plan->cell_count + entity;
    operator_handle->coo_residual_offsets[combined_entity] = vector_offset;
    if (jacobian_enabled != 0) {
      operator_handle->coo_jacobian_offsets[combined_entity] = matrix_offset;
    }
    if (facet->geometry.facet_owned != 0U) {
      SS_OPERATOR_APPEND_COO(facet);
    }
  }
#undef SS_OPERATOR_APPEND_COO
  operator_handle->coo_residual_offsets[entity_count] = vector_offset;
  if (jacobian_enabled != 0) {
    operator_handle->coo_jacobian_offsets[entity_count] = matrix_offset;
  }
  if (vector_offset != vector_count || matrix_offset != matrix_count) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (VecSetPreallocationCOO(operator_handle->staging_residual,
                             (PetscCount)vector_count,
                             vector_indices) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (jacobian_enabled != 0) {
    if (MatSetPreallocationCOO(operator_handle->staging_jacobian,
                               (PetscCount)matrix_count, matrix_rows,
                               matrix_columns) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                   status);
    if (status != SS_OK) {
      goto cleanup;
    }
    if (MatSetOption(operator_handle->staging_jacobian, MAT_NEW_NONZERO_ALLOCATION_ERR,
                     PETSC_TRUE) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                   status);
    if (status != SS_OK) {
      goto cleanup;
    }
    if (MatSetOption(operator_handle->staging_jacobian, MAT_KEEP_NONZERO_PATTERN,
                     PETSC_TRUE) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                   status);
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  operator_handle->coo_residual_count = vector_count;
  operator_handle->coo_jacobian_count = matrix_count;
  operator_handle->coo_entity_count = entity_count;
  operator_handle->performance_coo = 1;
  operator_handle->diagnostics.admitted_peak_bytes += coo_bytes;

cleanup:
  free(matrix_columns);
  free(matrix_rows);
  free(vector_indices);
  if (status != SS_OK) {
    free(operator_handle->coo_jacobian_values);
    free(operator_handle->coo_residual_values);
    free(operator_handle->coo_jacobian_offsets);
    free(operator_handle->coo_residual_offsets);
    operator_handle->coo_jacobian_values = NULL;
    operator_handle->coo_residual_values = NULL;
    operator_handle->coo_jacobian_offsets = NULL;
    operator_handle->coo_residual_offsets = NULL;
  }
  return status;
}

static SsStatus ss_operator_stage_element_values(
    SsOperator *operator_handle, uint32_t request, size_t closure_count,
    const PetscInt *global_indices, const uint8_t *constrained,
    const double *element_residual, const double *element_jacobian, int contribute,
    size_t coo_entity, SsEntityId entity, SsErrorRecord *error) {
  PetscInt free_indices[SS_FIELD_MAX_CLOSURE];
  PetscScalar residual_values[SS_FIELD_MAX_CLOSURE];
  PetscScalar *matrix_values = NULL;
  size_t free_slots[SS_FIELD_MAX_CLOSURE];
  size_t free_count = 0U;
  size_t residual_start = 0U;
  size_t jacobian_start = 0U;
  size_t row;
  size_t column;
  if (closure_count > SS_FIELD_MAX_CLOSURE || global_indices == NULL ||
      constrained == NULL ||
      ((request & SS_ASSEMBLY_RESIDUAL) != 0U && element_residual == NULL) ||
      ((request & SS_ASSEMBLY_JACOBIAN) != 0U && element_jacobian == NULL)) {
    ss_assembly_error_private(error, SS_ERROR_INVALID_ARGUMENT,
                              "ASM_PERFORMANCE_STAGE_ARGUMENT", entity,
                              operator_handle->plan->rank);
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (row = 0U; row < closure_count; ++row) {
    if (constrained[row] == 0U) {
      free_slots[free_count] = row;
      if (operator_handle->performance_coo == 0) {
        free_indices[free_count] = global_indices[row];
      }
      if (operator_handle->performance_coo == 0 && contribute != 0 &&
          (request & SS_ASSEMBLY_RESIDUAL) != 0U) {
        if (!isfinite(element_residual[row])) {
          ss_assembly_error_private(error, SS_ERROR_INTEGRITY, "ASM_NONFINITE_ELEMENT",
                                    entity, operator_handle->plan->rank);
          return SS_ERROR_INTEGRITY;
        }
        residual_values[free_count] = (PetscScalar)element_residual[row];
      }
      ++free_count;
    }
  }
  if (operator_handle->performance_coo != 0) {
    if (coo_entity >= operator_handle->coo_entity_count ||
        ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
         operator_handle->coo_residual_offsets == NULL) ||
        ((request & SS_ASSEMBLY_JACOBIAN) != 0U &&
         operator_handle->coo_jacobian_offsets == NULL)) {
      ss_assembly_error_private(error, SS_ERROR_INTEGRITY, "ASM_PERFORMANCE_COO_OFFSET",
                                entity, operator_handle->plan->rank);
      return SS_ERROR_INTEGRITY;
    }
    if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
      residual_start = operator_handle->coo_residual_offsets[coo_entity];
    }
    if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
      jacobian_start = operator_handle->coo_jacobian_offsets[coo_entity];
    }
    if (((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
         (operator_handle->coo_residual_offsets[coo_entity + 1U] - residual_start !=
              free_count ||
          free_count > operator_handle->coo_residual_count ||
          residual_start > operator_handle->coo_residual_count - free_count)) ||
        ((request & SS_ASSEMBLY_JACOBIAN) != 0U &&
         ((free_count != 0U && free_count > SIZE_MAX / free_count) ||
          operator_handle->coo_jacobian_offsets[coo_entity + 1U] - jacobian_start !=
              free_count * free_count ||
          free_count * free_count > operator_handle->coo_jacobian_count ||
          jacobian_start >
              operator_handle->coo_jacobian_count - free_count * free_count))) {
      ss_assembly_error_private(error, SS_ERROR_INTEGRITY, "ASM_PERFORMANCE_COO_OFFSET",
                                entity, operator_handle->plan->rank);
      return SS_ERROR_INTEGRITY;
    }
  }
  if (contribute == 0) {
    return SS_OK;
  }
  if (operator_handle->performance_coo != 0) {
    for (row = 0U; row < free_count; ++row) {
      if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
        const double value = element_residual[free_slots[row]];
        if (!isfinite(value)) {
          ss_assembly_error_private(error, SS_ERROR_INTEGRITY, "ASM_NONFINITE_ELEMENT",
                                    entity, operator_handle->plan->rank);
          return SS_ERROR_INTEGRITY;
        }
        operator_handle->coo_residual_values[residual_start + row] = (PetscScalar)value;
      }
      if ((request & SS_ASSEMBLY_JACOBIAN) == 0U) {
        continue;
      }
      for (column = 0U; column < free_count; ++column) {
        const double value =
            element_jacobian[free_slots[row] * closure_count + free_slots[column]];
        if (!isfinite(value)) {
          ss_assembly_error_private(error, SS_ERROR_INTEGRITY, "ASM_NONFINITE_ELEMENT",
                                    entity, operator_handle->plan->rank);
          return SS_ERROR_INTEGRITY;
        }
        operator_handle
            ->coo_jacobian_values[jacobian_start + row * free_count + column] =
            (PetscScalar)value;
      }
    }
    return SS_OK;
  }
  if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
    /* The insertion fallback runs on the master thread only, so one operator-
     * owned block replaces a maximum-closure-squared stack array (1.9 MiB per
     * call frame at P8). */
    if (operator_handle->stage_matrix_scratch == NULL) {
      operator_handle->stage_matrix_scratch = (PetscScalar *)malloc(
          (operator_handle->maximum_closure * operator_handle->maximum_closure == 0U
               ? 1U
               : operator_handle->maximum_closure * operator_handle->maximum_closure) *
          sizeof(*operator_handle->stage_matrix_scratch));
      if (operator_handle->stage_matrix_scratch == NULL) {
        ss_assembly_error_private(error, SS_ERROR_OUT_OF_MEMORY,
                                  "ASM_PERFORMANCE_STAGE_SCRATCH", entity,
                                  operator_handle->plan->rank);
        return SS_ERROR_OUT_OF_MEMORY;
      }
    }
    matrix_values = operator_handle->stage_matrix_scratch;
    for (row = 0U; row < free_count; ++row) {
      for (column = 0U; column < free_count; ++column) {
        const double value =
            element_jacobian[free_slots[row] * closure_count + free_slots[column]];
        if (!isfinite(value)) {
          ss_assembly_error_private(error, SS_ERROR_INTEGRITY, "ASM_NONFINITE_ELEMENT",
                                    entity, operator_handle->plan->rank);
          return SS_ERROR_INTEGRITY;
        }
        matrix_values[row * free_count + column] = (PetscScalar)value;
      }
    }
  }
  if (free_count == 0U) {
    return SS_OK;
  }
  if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK ||
      ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
       VecSetValues(operator_handle->staging_residual, (PetscInt)free_count,
                    free_indices, residual_values, ADD_VALUES) != PETSC_SUCCESS) ||
      ((request & SS_ASSEMBLY_JACOBIAN) != 0U &&
       MatSetValues(operator_handle->staging_jacobian, (PetscInt)free_count,
                    free_indices, (PetscInt)free_count, free_indices, matrix_values,
                    ADD_VALUES) != PETSC_SUCCESS)) {
    ss_assembly_error_private(error, SS_ERROR_INTERNAL, "ASM_PERFORMANCE_STAGE", entity,
                              operator_handle->plan->rank);
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

void ss_operator_stage_cell(SsOperator *operator_handle, uint32_t request,
                            size_t cell_index, size_t tile_slot) {
  const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
  SsErrorRecord stage_error = {0};
  int contribute = operator_handle->element_errors[tile_slot].valid == 0 ? 1 : 0;
  if (cell->cell_owned == 0U) {
    return;
  }
  if (contribute != 0 && (request & SS_ASSEMBLY_POTENTIAL) != 0U &&
      !isfinite(operator_handle->element_potential[tile_slot])) {
    ss_assembly_error_private(&stage_error, SS_ERROR_INTEGRITY, "ASM_NONFINITE_ELEMENT",
                              cell->stable_cell_id, operator_handle->plan->rank);
    contribute = 0;
  }
  if ((request & (SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN)) != 0U) {
    (void)ss_operator_stage_element_values(
        operator_handle, request, cell->closure_count, cell->global_indices,
        cell->constrained,
        operator_handle->element_residual +
            tile_slot * operator_handle->maximum_closure,
        operator_handle->element_jacobian + tile_slot * operator_handle->matrix_stride,
        contribute, cell_index, cell->stable_cell_id, &stage_error);
  }
  if (stage_error.valid != 0) {
    operator_handle->element_errors[tile_slot] = stage_error;
  }
}

void ss_operator_stage_natural(SsOperator *operator_handle, uint32_t request,
                               size_t facet_index, size_t tile_slot) {
  const SsOperatorNaturalFacet *facet = &operator_handle->natural_facets[facet_index];
  SsErrorRecord stage_error = {0};
  const int contribute = operator_handle->element_errors[tile_slot].valid == 0 ? 1 : 0;
  if (facet->pack.geometry.facet_owned == 0U) {
    return;
  }
  (void)ss_operator_stage_element_values(
      operator_handle, request, facet->pack.closure_count, facet->pack.global_indices,
      facet->pack.constrained,
      operator_handle->element_residual + tile_slot * operator_handle->maximum_closure,
      operator_handle->element_jacobian + tile_slot * operator_handle->matrix_stride,
      contribute, operator_handle->plan->cell_count + facet_index,
      facet->pack.geometry.stable_facet_id, &stage_error);
  if (stage_error.valid != 0) {
    operator_handle->element_errors[tile_slot] = stage_error;
  }
}

SsStatus ss_operator_prepare_performance_staging(SsOperator *operator_handle,
                                                 uint32_t request) {
  if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK ||
      ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
       VecZeroEntries(operator_handle->staging_residual) != PETSC_SUCCESS) ||
      ((request & SS_ASSEMBLY_JACOBIAN) != 0U &&
       MatZeroEntries(operator_handle->staging_jacobian) != PETSC_SUCCESS)) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

SsStatus ss_operator_finish_performance_staging(SsOperator *operator_handle,
                                                uint32_t request) {
  SsStatus status = SS_OK;
  if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK) {
    return SS_ERROR_THREAD;
  }
  if (operator_handle->performance_coo != 0) {
    MatInfo info = {0};
    if ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
        VecSetValuesCOO(operator_handle->staging_residual,
                        operator_handle->coo_residual_values,
                        ADD_VALUES) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    if ((request & SS_ASSEMBLY_JACOBIAN) != 0U &&
        (MatSetValuesCOO(operator_handle->staging_jacobian,
                         operator_handle->coo_jacobian_values,
                         ADD_VALUES) != PETSC_SUCCESS ||
         MatGetInfo(operator_handle->staging_jacobian, MAT_LOCAL, &info) !=
             PETSC_SUCCESS)) {
      status = SS_ERROR_INTERNAL;
    } else if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
      operator_handle->diagnostics.petsc_new_nonzero_allocations =
          info.mallocs < 0.0 ? UINT64_MAX : (uint64_t)info.mallocs;
      if (info.mallocs != 0.0) {
        status = SS_ERROR_INTEGRITY;
      }
    }
    return status;
  }
  if ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
      (VecAssemblyBegin(operator_handle->staging_residual) != PETSC_SUCCESS ||
       VecAssemblyEnd(operator_handle->staging_residual) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
    MatInfo info = {0};
    if (MatAssemblyBegin(operator_handle->staging_jacobian, MAT_FINAL_ASSEMBLY) !=
            PETSC_SUCCESS ||
        MatAssemblyEnd(operator_handle->staging_jacobian, MAT_FINAL_ASSEMBLY) !=
            PETSC_SUCCESS ||
        MatGetInfo(operator_handle->staging_jacobian, MAT_LOCAL, &info) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      operator_handle->diagnostics.petsc_new_nonzero_allocations =
          info.mallocs < 0.0 ? UINT64_MAX : (uint64_t)info.mallocs;
      if (info.mallocs != 0.0) {
        status = SS_ERROR_INTEGRITY;
      }
    }
  }
  return status;
}
