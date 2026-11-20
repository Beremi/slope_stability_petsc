/* Create matrix-free PETSc shells and update their linearization.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "operator_internal.h"
#include "voigt.h"

#include "slope_internal/build/config.h"

#include <petsclog.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

static SsStatus ss_operator_matrix_free_allocate(SsOperator *operator_handle) {
  size_t tangent_count = 0U;
  size_t action_count = 0U;
  size_t block_count = 0U;
  size_t block_entry_count = 0U;
  size_t block_workspace_stride = 0U;
  size_t block_workspace_entries = 0U;
  size_t action_workspace_stride = 0U;
  size_t action_workspace_entries = 0U;
  size_t tangent_offset = 0U;
  size_t block_entry_offset = 0U;
  size_t row = 0U;
  uint64_t storage_bytes = 0U;
  size_t cell_index;
  size_t block;
  if (operator_handle == NULL ||
      operator_handle->spec.physics != SS_ASSEMBLY_MECHANICS ||
      operator_handle->spec.matrix_free == 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (operator_handle->matrix_free_tangent_offsets != NULL) {
    return SS_OK;
  }
  if (ss_operator_matrix_free_storage_requirements(
          operator_handle, &tangent_count, &action_count, &storage_bytes) != SS_OK ||
      ss_operator_mechanics_block_requirements(operator_handle, &block_count,
                                               &block_entry_count) != SS_OK ||
      ss_operator_matrix_free_block_workspace_requirements(
          operator_handle, &block_workspace_stride, &block_workspace_entries) !=
          SS_OK ||
      ss_operator_matrix_free_action_workspace_requirements(
          operator_handle, &action_workspace_stride, &action_workspace_entries) !=
          SS_OK ||
      storage_bytes != operator_handle->diagnostics.matrix_free_bytes) {
    return SS_ERROR_OVERFLOW;
  }
  operator_handle->matrix_free_tangent_offsets =
      (size_t *)calloc(operator_handle->plan->cell_count + 1U,
                       sizeof(*operator_handle->matrix_free_tangent_offsets));
  if (operator_handle->matrix_free_tangent_offsets == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    const SsOperatorCell *metadata = &operator_handle->cells[cell_index];
    const size_t strain_count = cell->dimension == 2U ? 3U : 6U;
    const size_t point_count = metadata->quadrature->point_count;
    const size_t cell_tangent_count = point_count * strain_count * strain_count;
    operator_handle->matrix_free_tangent_offsets[cell_index] = tangent_offset;
    tangent_offset += cell_tangent_count;
  }
  operator_handle->matrix_free_tangent_offsets[operator_handle->plan->cell_count] =
      tangent_offset;
  if (tangent_offset != tangent_count) {
    return SS_ERROR_INTEGRITY;
  }
  if (tangent_count > SIZE_MAX / sizeof(*operator_handle->matrix_free_tangents) ||
      action_count > SIZE_MAX / sizeof(*operator_handle->matrix_free_element_action) ||
      operator_handle->plan->row_count >
          SIZE_MAX / sizeof(*operator_handle->matrix_free_diagonal)) {
    return SS_ERROR_OVERFLOW;
  }
  operator_handle->matrix_free_tangents =
      (double *)calloc(tangent_count == 0U ? 1U : tangent_count,
                       sizeof(*operator_handle->matrix_free_tangents));
  operator_handle->matrix_free_element_action =
      (double *)calloc(action_count == 0U ? 1U : action_count,
                       sizeof(*operator_handle->matrix_free_element_action));
  operator_handle->matrix_free_diagonal = (double *)calloc(
      operator_handle->plan->row_count == 0U ? 1U : operator_handle->plan->row_count,
      sizeof(*operator_handle->matrix_free_diagonal));
  operator_handle->matrix_free_block_sizes =
      (PetscInt *)calloc(block_count == 0U ? 1U : block_count,
                         sizeof(*operator_handle->matrix_free_block_sizes));
  operator_handle->matrix_free_block_offsets = (size_t *)calloc(
      block_count + 1U, sizeof(*operator_handle->matrix_free_block_offsets));
  operator_handle->matrix_free_block_row_offsets = (size_t *)calloc(
      block_count + 1U, sizeof(*operator_handle->matrix_free_block_row_offsets));
  operator_handle->matrix_free_inverse_blocks =
      (PetscScalar *)calloc(block_entry_count == 0U ? 1U : block_entry_count,
                            sizeof(*operator_handle->matrix_free_inverse_blocks));
  operator_handle->matrix_free_block_workspace = (PetscScalar *)calloc(
      block_workspace_entries == 0U ? 1U : block_workspace_entries,
      sizeof(*operator_handle->matrix_free_block_workspace));
  operator_handle->matrix_free_action_workspace = (PetscScalar *)calloc(
      action_workspace_entries == 0U ? 1U : action_workspace_entries,
      sizeof(*operator_handle->matrix_free_action_workspace));
  if (operator_handle->matrix_free_tangents == NULL ||
      operator_handle->matrix_free_element_action == NULL ||
      operator_handle->matrix_free_diagonal == NULL ||
      operator_handle->matrix_free_block_sizes == NULL ||
      operator_handle->matrix_free_block_row_offsets == NULL ||
      operator_handle->matrix_free_block_offsets == NULL ||
      operator_handle->matrix_free_inverse_blocks == NULL ||
      operator_handle->matrix_free_block_workspace == NULL ||
      operator_handle->matrix_free_action_workspace == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (block = 0U; block < block_count; ++block) {
    size_t block_size = 0U;
    if (ss_operator_mechanics_block_size(operator_handle, row, &block_size) != SS_OK) {
      return SS_ERROR_INTEGRITY;
    }
    operator_handle->matrix_free_block_sizes[block] = (PetscInt)block_size;
    operator_handle->matrix_free_block_row_offsets[block] = row;
    operator_handle->matrix_free_block_offsets[block] = block_entry_offset;
    block_entry_offset += block_size * block_size;
    row += block_size;
  }
  operator_handle->matrix_free_block_offsets[block_count] = block_entry_offset;
  operator_handle->matrix_free_block_row_offsets[block_count] = row;
  if (row != operator_handle->plan->row_count ||
      block_entry_offset != block_entry_count) {
    return SS_ERROR_INTEGRITY;
  }
  operator_handle->matrix_free_block_count = (PetscInt)block_count;
  operator_handle->matrix_free_block_workspace_stride = block_workspace_stride;
  operator_handle->matrix_free_action_workspace_stride = action_workspace_stride;
  operator_handle->matrix_free_tangent_count = tangent_count;
  return SS_OK;
}

SsStatus ss_operator_matrix_free_create(SsOperator *operator_handle, Mat *tangent,
                                        SsErrorRecord *error) {
  SsStatus status;
  if (tangent != NULL) {
    *tangent = NULL;
  }
  if (operator_handle == NULL || tangent == NULL ||
      operator_handle->spec.physics != SS_ASSEMBLY_MECHANICS ||
      operator_handle->spec.matrix_free == 0) {
    ss_assembly_error_private(
        error, SS_ERROR_INVALID_ARGUMENT, "ASM_MATFREE_CREATE_ARGUMENT", 0U,
        operator_handle == NULL ? 0 : operator_handle->plan->rank);
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_operator_matrix_free_allocate(operator_handle);
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, "ASM_MATFREE_ALLOCATE", 0U,
                              operator_handle->plan->rank);
    return status;
  }
  status = ss_operator_matrix_free_prepare_exchange(operator_handle);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, "ASM_MATFREE_EXCHANGE", 0U,
                              operator_handle->plan->rank);
    return status;
  }
  if (operator_handle->matrix_free_operator == NULL) {
    if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK ||
        MatCreateShell(operator_handle->plan->communicator,
                       (PetscInt)operator_handle->plan->summary.local_owned_rows,
                       (PetscInt)operator_handle->plan->summary.local_owned_rows,
                       (PetscInt)operator_handle->plan->summary.global_rows,
                       (PetscInt)operator_handle->plan->summary.global_rows,
                       operator_handle,
                       &operator_handle->matrix_free_operator) != PETSC_SUCCESS ||
        MatShellSetOperation(operator_handle->matrix_free_operator, MATOP_MULT,
                             (void (*)(void))ss_operator_matrix_free_mult) !=
            PETSC_SUCCESS ||
        MatShellSetOperation(
            operator_handle->matrix_free_operator, MATOP_MULT_TRANSPOSE,
            (void (*)(void))ss_operator_matrix_free_mult_transpose) != PETSC_SUCCESS ||
        MatShellSetOperation(operator_handle->matrix_free_operator, MATOP_GET_DIAGONAL,
                             (void (*)(void))ss_operator_matrix_free_get_diagonal) !=
            PETSC_SUCCESS ||
        MatShellSetOperation(
            operator_handle->matrix_free_operator, MATOP_INVERT_VBLOCK_DIAGONAL,
            (void (*)(void))ss_operator_matrix_free_invert_variable_block_diagonal) !=
            PETSC_SUCCESS ||
        MatShellSetOperation(operator_handle->matrix_free_operator, MATOP_CREATE_VECS,
                             (void (*)(void))ss_operator_matrix_free_create_vecs) !=
            PETSC_SUCCESS ||
        MatSetVariableBlockSizes(operator_handle->matrix_free_operator,
                                 operator_handle->matrix_free_block_count,
                                 operator_handle->matrix_free_block_sizes) !=
            PETSC_SUCCESS ||
        (operator_handle->matrix_free_threaded_block_apply != 0 &&
         PetscObjectComposeFunction(
             (PetscObject)operator_handle->matrix_free_operator,
             "SsMatApplyInverseVariableBlocks_C",
             ss_operator_matrix_free_apply_inverse_variable_blocks) != PETSC_SUCCESS) ||
        PetscObjectSetName((PetscObject)operator_handle->matrix_free_operator,
                           "mechanics_tangent_matfree") != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                   status);
    if (status != SS_OK) {
      ss_assembly_error_private(error, status, "ASM_MATFREE_SHELL", 0U,
                                operator_handle->plan->rank);
      return status;
    }
  }
  if (PetscObjectReference((PetscObject)operator_handle->matrix_free_operator) !=
      PETSC_SUCCESS) {
    ss_assembly_error_private(error, SS_ERROR_INTERNAL, "ASM_MATFREE_REFERENCE", 0U,
                              operator_handle->plan->rank);
    return SS_ERROR_INTERNAL;
  }
  *tangent = operator_handle->matrix_free_operator;
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
  return SS_OK;
}

SsStatus ss_operator_matrix_free_linearize(SsOperator *operator_handle,
                                           const SsOperatorState *state,
                                           double elastic_regularization,
                                           SsErrorRecord *error) {
  double *elastic_tangents = NULL;
  const PetscScalar *gathered = NULL;
  int gathered_open = 0;
  SsErrorRecord selected = {0};
  SsErrorRecord local_error = {0};
  SsErrorRecord natural_error = {0};
  SsStatus status = SS_OK;
  int use_openmp;
  int block_failure = 0;
  size_t block_index;
  size_t cell_index;
  size_t region;
#if !SLOPE_ENABLE_OPENMP
  (void)use_openmp;
#endif
  if (operator_handle == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (state == NULL || state->field == NULL ||
      operator_handle->spec.physics != SS_ASSEMBLY_MECHANICS ||
      operator_handle->spec.matrix_free == 0 ||
      operator_handle->matrix_free_operator == NULL ||
      !isfinite(elastic_regularization) || elastic_regularization < 0.0 ||
      elastic_regularization > 1.0 || !isfinite(state->reduction_factor) ||
      state->reduction_factor <= 0.0 || !isfinite(state->external_load_factor)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (state->cancelled) {
    status = SS_ERROR_CANCELLED;
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status,
                              status == SS_ERROR_CANCELLED
                                  ? "ASM_MATFREE_CANCELLED"
                                  : "ASM_MATFREE_LINEARIZE_ARGUMENT",
                              0U, operator_handle->plan->rank);
    return status;
  }
  operator_handle->matrix_free_linearized = 0;
  use_openmp = ss_operator_use_openmp(operator_handle);
  if (operator_handle->mechanics.region_count > SIZE_MAX / 36U ||
      operator_handle->mechanics.region_count * 36U >
          SIZE_MAX / sizeof(*elastic_tangents)) {
    status = SS_ERROR_OVERFLOW;
  } else {
    elastic_tangents = (double *)calloc(operator_handle->mechanics.region_count * 36U,
                                        sizeof(*elastic_tangents));
    if (elastic_tangents == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status == SS_OK) {
    status =
        ss_operator_prepare_reduced_materials(operator_handle, state->reduction_factor);
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status == SS_OK && operator_handle->plan->cell_count != 0U) {
    const SsMcDimension dimension = operator_handle->properties.dimension == 2U
                                        ? SS_MC_DIMENSION_2
                                        : SS_MC_DIMENSION_3;
    const size_t strain_count = operator_handle->properties.dimension == 2U ? 3U : 6U;
    const double zero_strain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (region = 0U; region < operator_handle->mechanics.region_count; ++region) {
      SsMcResponse response = {0};
      SsMcError material_error = {0};
      status = ss_mc_evaluate_prepared(
          operator_handle->material_regions[region].parameters, dimension,
          &operator_handle->reduced_materials[region], SS_MC_EVALUATE_TANGENT,
          zero_strain, &response, &material_error);
      if (status != SS_OK) {
        ss_assembly_error_private(&selected, status,
                                  material_error.code[0] == '\0' ? "ASM_MATFREE_ELASTIC"
                                                                 : material_error.code,
                                  0U, operator_handle->plan->rank);
        break;
      }
      memcpy(elastic_tangents + region * 36U, response.tangent,
             strain_count * strain_count * sizeof(*response.tangent));
    }
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status == SS_OK) {
    status = ss_operator_gather(operator_handle, state, &gathered);
    gathered_open = status == SS_OK;
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    if (gathered_open) {
      (void)VecRestoreArrayRead(operator_handle->gathered_values, &gathered);
    }
    free(elastic_tangents);
    if (selected.valid != 0 && error != NULL) {
      *error = selected;
    } else {
      ss_assembly_error_private(error, status, "ASM_MATFREE_PREPARE", 0U,
                                operator_handle->plan->rank);
    }
    return status;
  }
  (void)PetscLogEventBegin(ss_operator_matrix_free_tangent_event, 0, 0, 0, 0);
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel num_threads(                                                      \
        operator_handle->spec.requested_threads) if (use_openmp)                       \
    reduction(ss_first_error : local_error)
#endif
  {
    ss_operator_pack_cells(operator_handle, gathered);
#if SLOPE_ENABLE_OPENMP
#pragma omp for schedule(static)
#endif
    for (cell_index = 0U; cell_index < operator_handle->plan->cell_count;
         ++cell_index) {
      const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
      const SsOperatorCell *metadata = &operator_handle->cells[cell_index];
      const size_t strain_count = cell->dimension == 2U ? 3U : 6U;
      size_t quadrature_index;
      for (quadrature_index = 0U; quadrature_index < metadata->quadrature->point_count;
           ++quadrature_index) {
        const double *elastic =
            elastic_tangents + metadata->material_region_index * 36U;
        double *cached = operator_handle->matrix_free_tangents +
                         operator_handle->matrix_free_tangent_offsets[cell_index] +
                         quadrature_index * strain_count * strain_count;
        double strain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        SsMcResponse response = {0};
        SsMcError material_error = {0};
        SsStatus point_status;
        size_t slot;
        size_t entry;
        if (metadata->prepared.physical_gradients == NULL && cell->dimension == 3U) {
          const double *inverse_transpose =
              ss_operator_cell_inverse_transpose(metadata, quadrature_index);
          double reference_displacement_gradient[9] = {0.0, 0.0, 0.0, 0.0, 0.0,
                                                       0.0, 0.0, 0.0, 0.0};
          const size_t node_count = metadata->field_descriptor->node_count;
          const double *reference = operator_handle->reference_gradient_cache +
                                    quadrature_index * node_count * 3U;
          size_t node;
          for (node = 0U; node < node_count; ++node) {
            const double *gradient = reference + 3U * node;
            const double *values = operator_handle->packed_values +
                                   cell_index * operator_handle->maximum_closure +
                                   3U * node;
            size_t component;
            for (component = 0U; component < 3U; ++component) {
              reference_displacement_gradient[3U * component] +=
                  gradient[0] * values[component];
              reference_displacement_gradient[3U * component + 1U] +=
                  gradient[1] * values[component];
              reference_displacement_gradient[3U * component + 2U] +=
                  gradient[2] * values[component];
            }
          }
          {
            double physical[9];
            size_t component;
            for (component = 0U; component < 3U; ++component) {
              size_t coordinate;
              for (coordinate = 0U; coordinate < 3U; ++coordinate) {
                physical[3U * component + coordinate] =
                    inverse_transpose[3U * coordinate] *
                        reference_displacement_gradient[3U * component] +
                    inverse_transpose[3U * coordinate + 1U] *
                        reference_displacement_gradient[3U * component + 1U] +
                    inverse_transpose[3U * coordinate + 2U] *
                        reference_displacement_gradient[3U * component + 2U];
              }
            }
            strain[0] = physical[0];
            strain[1] = physical[4];
            strain[2] = physical[8];
            strain[3] = physical[1] + physical[3];
            strain[4] = physical[5] + physical[7];
            strain[5] = physical[2] + physical[6];
          }
        } else {
          double gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
          ss_operator_mechanics_point_gradients(operator_handle, metadata,
                                                quadrature_index, gradients);
          for (slot = 0U; slot < cell->closure_count; ++slot) {
            const double *gradient =
                gradients + (slot / cell->dimension) * cell->dimension;
            const double value =
                operator_handle
                    ->packed_values[cell_index * operator_handle->maximum_closure +
                                    slot];
            ss_mechanics_accumulate_strain(cell->dimension, gradient,
                                           (uint32_t)(slot % cell->dimension), value,
                                           strain);
          }
        }
        point_status = ss_mc_evaluate_prepared(
            metadata->material,
            cell->dimension == 2U ? SS_MC_DIMENSION_2 : SS_MC_DIMENSION_3,
            metadata->prepared.reduced_material, SS_MC_EVALUATE_TANGENT, strain,
            &response, &material_error);
        if (point_status != SS_OK) {
          SsErrorRecord point_error;
          ss_assembly_error_private(&point_error, point_status,
                                    material_error.code[0] == '\0'
                                        ? "ASM_MATFREE_MATERIAL"
                                        : material_error.code,
                                    cell->stable_cell_id, operator_handle->plan->rank);
          point_error.key.quadrature = (uint64_t)quadrature_index;
          point_error.key.thread = (uint32_t)ss_operator_thread();
          ss_operator_keep_first_error(&local_error, &point_error);
          break;
        }
        for (entry = 0U; entry < strain_count * strain_count; ++entry) {
          cached[entry] = (1.0 - elastic_regularization) * response.tangent[entry] +
                          elastic_regularization * elastic[entry];
        }
      }
    }
  }
  if (use_openmp) {
    ++operator_handle->diagnostics.parallel_pack_calls;
  }
  if (VecRestoreArrayRead(operator_handle->gathered_values, &gathered) !=
      PETSC_SUCCESS) {
    SsErrorRecord array_error;
    ss_assembly_error_private(&array_error, SS_ERROR_INTERNAL, "ASM_PACK_RESTORE", 0U,
                              operator_handle->plan->rank);
    ss_operator_keep_first_error(&local_error, &array_error);
  }
  status = ss_operator_collective_validation(operator_handle, &local_error, &selected);
  (void)PetscLogEventEnd(ss_operator_matrix_free_tangent_event, 0, 0, 0, 0);
  free(elastic_tangents);
  if (status != SS_OK) {
    if (selected.valid != 0 && error != NULL) {
      *error = selected;
    }
    return status;
  }
  /*
   * The application mechanics boundary kernels are dead-load kernels.  Reject
   * a state-dependent natural tangent explicitly instead of silently omitting
   * it from the shell action.
   */
  for (cell_index = 0U; cell_index < operator_handle->natural_facet_count;
       ++cell_index) {
    const SsOperatorNaturalFacet *facet = &operator_handle->natural_facets[cell_index];
    const SsNaturalOperatorSpec *operation =
        &operator_handle->natural_operations[facet->operation_index];
    const SsCellPlan *parent = &operator_handle->plan->cells[facet->parent_cell_index];
    double field_values[SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
    double jacobian[SS_FIELD_MAX_FACET_CLOSURE * SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
    double signed_flux = 0.0;
    SsElementOutput output = {
        0U,  SS_FIELD_MAX_FACET_CLOSURE * SS_FIELD_MAX_FACET_CLOSURE, NULL, jacobian,
        0.0,
    };
    SsNaturalElementInput input = {
        &facet->pack, facet->field_descriptor, facet->quadrature,
        field_values, operation->point_kernel, operation->point_context,
    };
    size_t slot;
    for (slot = 0U; slot < facet->pack.closure_count; ++slot) {
      const size_t parent_slot = facet->pack.parent_closure_slots[slot];
      if (parent_slot >= parent->closure_count) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      field_values[slot] =
          operator_handle->packed_values[facet->parent_cell_index *
                                             operator_handle->maximum_closure +
                                         parent_slot];
    }
    if (status == SS_OK) {
      status = ss_assembly_natural_element(&input, SS_ASSEMBLY_JACOBIAN, &output,
                                           &signed_flux, &natural_error);
    }
    for (slot = 0U; status == SS_OK &&
                    slot < facet->pack.closure_count * facet->pack.closure_count;
         ++slot) {
      if (jacobian[slot] != 0.0) {
        status = SS_ERROR_INVALID_STATE;
      }
    }
    if (status != SS_OK) {
      ss_assembly_error_private(&natural_error, status, "ASM_MATFREE_NATURAL_TANGENT",
                                facet->pack.geometry.stable_facet_id,
                                operator_handle->plan->rank);
      break;
    }
  }
  status = ss_operator_collective_validation(
      operator_handle, natural_error.valid != 0 ? &natural_error : NULL, &selected);
  if (status != SS_OK) {
    if (error != NULL) {
      *error = selected;
    }
    return status;
  }
  (void)PetscLogEventBegin(ss_operator_matrix_free_block_setup_event, 0, 0, 0, 0);
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel for num_threads(                                                  \
        operator_handle->spec.requested_threads) if (use_openmp != 0) schedule(static) \
    reduction(max : block_failure)
#endif
  for (block_index = 0U; block_index < (size_t)operator_handle->matrix_free_block_count;
       ++block_index) {
    const size_t block_row_start =
        operator_handle->matrix_free_block_row_offsets[block_index];
    const PetscInt block_size = operator_handle->matrix_free_block_sizes[block_index];
    PetscScalar *inverse = operator_handle->matrix_free_inverse_blocks +
                           operator_handle->matrix_free_block_offsets[block_index];
    PetscInt block_row;
    int valid = 1;
    if (block_size > 3 ||
        (operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE &&
         operator_handle->matrix_free_block_workspace_stride != 0U)) {
      const size_t thread = (size_t)ss_operator_thread();
      if (thread >= (size_t)operator_handle->spec.requested_threads ||
          operator_handle->matrix_free_block_workspace_stride == 0U) {
        valid = 0;
      } else {
        PetscScalar *workspace =
            operator_handle->matrix_free_block_workspace +
            thread * operator_handle->matrix_free_block_workspace_stride;
        valid = ss_operator_matrix_free_entity_block(operator_handle, block_row_start,
                                                     block_size, inverse, workspace);
      }
    } else {
      for (block_row = 0; valid != 0 && block_row < block_size; ++block_row) {
        const SsOwnedRowPlan *row =
            &operator_handle->plan->rows[block_row_start + (size_t)block_row];
        PetscInt block_column;
        for (block_column = 0; block_column < block_size; ++block_column) {
          const SsOwnedRowPlan *column =
              &operator_handle->plan->rows[block_row_start + (size_t)block_column];
          double entry = 0.0;
          size_t contribution;
          for (contribution = 0U; contribution < row->cell_count; ++contribution) {
            const size_t cell_number = row->cell_slots[contribution].cell_index;
            const SsCellPlan *cell = &operator_handle->plan->cells[cell_number];
            const size_t column_slot = ss_operator_matrix_free_cell_slot(cell, column);
            if (column_slot == SIZE_MAX || cell->constrained[column_slot] != 0U) {
              valid = 0;
              break;
            }
            entry += ss_operator_matrix_free_cell_entry(
                operator_handle, cell_number, row->cell_slots[contribution].row_slot,
                column_slot);
          }
          inverse[block_row + block_column * block_size] = (PetscScalar)entry;
        }
      }
    }
    for (block_row = 0; valid != 0 && block_row < block_size; ++block_row) {
      operator_handle->matrix_free_diagonal[block_row_start + (size_t)block_row] =
          (double)PetscRealPart(inverse[block_row + block_row * block_size]);
    }
    if (valid == 0 || !ss_operator_matrix_free_invert_block(block_size, inverse)) {
      block_failure = 1;
    }
  }
  block_failure =
      block_failure || !ss_operator_matrix_free_inverses_finite(operator_handle);
  (void)PetscLogEventEnd(ss_operator_matrix_free_block_setup_event, 0, 0, 0, 0);
  status = ss_assembly_collective_status_private(
      operator_handle->plan->communicator,
      block_failure == 0 ? SS_OK : SS_ERROR_INTEGRITY);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, "ASM_MATFREE_BLOCK_DIAGONAL", 0U,
                              operator_handle->plan->rank);
    return status;
  }
  operator_handle->matrix_free_linearized = 1;
  ++operator_handle->matrix_free_generation;
  if (MatAssemblyBegin(operator_handle->matrix_free_operator, MAT_FINAL_ASSEMBLY) !=
          PETSC_SUCCESS ||
      MatAssemblyEnd(operator_handle->matrix_free_operator, MAT_FINAL_ASSEMBLY) !=
          PETSC_SUCCESS) {
    operator_handle->matrix_free_linearized = 0;
    ss_assembly_error_private(error, SS_ERROR_INTERNAL, "ASM_MATFREE_STATE", 0U,
                              operator_handle->plan->rank);
    return SS_ERROR_INTERNAL;
  }
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
  return SS_OK;
}
