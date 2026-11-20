/* Project matrix-free tangents into the PMG coarse operators.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "operator_internal.h"
#include "voigt.h"

#include "slope_internal/build/config.h"

#include <petscblaslapack.h>
#include <petsclog.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SS_OPERATOR_P1_MAX_CLOSURE 12U
#define SS_OPERATOR_P1_MAX_GRADIENTS 36U

static SsStatus ss_operator_matrix_free_compute_p1_cell(
    const SsOperator *fine_operator, const SsAssemblyPlan *p1_plan, size_t cell_index,
    PetscScalar *values, PetscScalar *workspace, size_t workspace_count) {
  const SsCellPlan *fine_cell = &fine_operator->plan->cells[cell_index];
  const SsCellPlan *coarse_cell = &p1_plan->cells[cell_index];
  const SsOperatorCell *fine_metadata = &fine_operator->cells[cell_index];
  const SsFeDescriptor *fine_descriptor = fine_metadata->field_descriptor;
  const SsFeDescriptor *coarse_descriptor = NULL;
  const size_t dimension = fine_cell->dimension;
  const size_t strain_count = dimension == 2U ? 3U : 6U;
  size_t slots[SS_OPERATOR_P1_MAX_CLOSURE];
  size_t free_count = 0U;
  size_t coarse_slot;
  size_t quadrature_index;
  if (values == NULL ||
      ss_fe_descriptor_find(coarse_cell->dimension, 1U, &coarse_descriptor) != SS_OK ||
      coarse_descriptor == NULL) {
    return SS_ERROR_INTEGRITY;
  }
  for (coarse_slot = 0U; coarse_slot < coarse_cell->closure_count; ++coarse_slot) {
    if (coarse_cell->constrained[coarse_slot] == 0U &&
        coarse_cell->global_indices[coarse_slot] >= 0) {
      slots[free_count++] = coarse_slot;
    }
  }
  if (dimension == 3U && fine_metadata->affine_geometry != 0 &&
      fine_operator->matrix_free_p1_constraint_pattern_indices != NULL &&
      fine_operator->matrix_free_p1_reference_gradient_blas_cache != NULL &&
      fine_operator->matrix_free_p1_constraint_pattern_count != 0U) {
    const size_t point_count = fine_metadata->quadrature->point_count;
    const size_t stacked_points = dimension * point_count;
    const size_t strain_rows = strain_count * point_count;
    const size_t pattern =
        fine_operator->matrix_free_p1_constraint_pattern_indices[cell_index];
    PetscScalar block[SS_OPERATOR_P1_MAX_CLOSURE * SS_OPERATOR_P1_MAX_CLOSURE];
    PetscScalar *basis = workspace;
    PetscScalar *weighted_product;
    PetscBLASInt rows;
    PetscBLASInt columns;
    const PetscScalar one = 1.0;
    const PetscScalar zero = 0.0;
    size_t free_slot;
    if (workspace == NULL ||
        pattern >= fine_operator->matrix_free_p1_constraint_pattern_count ||
        point_count > (size_t)PETSC_BLAS_INT_MAX || strain_rows == 0U ||
        strain_rows > (size_t)PETSC_BLAS_INT_MAX ||
        free_count > (size_t)PETSC_BLAS_INT_MAX ||
        free_count > SIZE_MAX / strain_rows ||
        free_count * strain_rows > workspace_count / 2U) {
      return SS_ERROR_INTEGRITY;
    }
    weighted_product = basis + free_count * strain_rows;
    for (free_slot = 0U; free_slot < free_count; ++free_slot) {
      const size_t slot = slots[free_slot];
      const size_t coarse_node = slot / dimension;
      const size_t component = slot % dimension;
      const PetscScalar *reference_gradients =
          fine_operator->matrix_free_p1_reference_gradient_blas_cache +
          (pattern * dimension + component) * coarse_descriptor->node_count *
              stacked_points +
          coarse_node * stacked_points;
      for (quadrature_index = 0U; quadrature_index < point_count; ++quadrature_index) {
        const double *tangent = fine_operator->matrix_free_tangents +
                                fine_operator->matrix_free_tangent_offsets[cell_index] +
                                quadrature_index * strain_count * strain_count;
        const double weight = fine_metadata->prepared.weights[quadrature_index];
        double reference[3];
        double physical[3];
        double basis_strain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double product[6];
        size_t coordinate;
        size_t strain;
        for (coordinate = 0U; coordinate < dimension; ++coordinate) {
          reference[coordinate] = (double)PetscRealPart(
              reference_gradients[coordinate * point_count + quadrature_index]);
        }
        for (coordinate = 0U; coordinate < dimension; ++coordinate) {
          physical[coordinate] =
              fine_metadata->inverse_transpose[coordinate * dimension] * reference[0] +
              fine_metadata->inverse_transpose[coordinate * dimension + 1U] *
                  reference[1] +
              fine_metadata->inverse_transpose[coordinate * dimension + 2U] *
                  reference[2];
        }
        ss_mechanics_accumulate_strain((uint32_t)dimension, physical,
                                       (uint32_t)component, 1.0, basis_strain);
        ss_mechanics_tangent_product(tangent, strain_count, basis_strain, 0, product);
        for (strain = 0U; strain < strain_count; ++strain) {
          const size_t offset =
              free_slot * strain_rows + quadrature_index * strain_count + strain;
          basis[offset] = (PetscScalar)basis_strain[strain];
          weighted_product[offset] = (PetscScalar)(product[strain] * weight);
        }
      }
    }
    rows = (PetscBLASInt)strain_rows;
    columns = (PetscBLASInt)free_count;
    BLASgemm_("T", "N", &columns, &columns, &rows, &one, basis, &rows, weighted_product,
              &rows, &zero, block, &columns);
    for (free_slot = 0U; free_slot < free_count; ++free_slot) {
      size_t free_column;
      for (free_column = 0U; free_column < free_count; ++free_column) {
        values[free_slot * free_count + free_column] =
            block[free_slot + free_column * free_count];
      }
    }
    return SS_OK;
  }
  for (quadrature_index = 0U; quadrature_index < fine_metadata->quadrature->point_count;
       ++quadrature_index) {
    const double *tangent = fine_operator->matrix_free_tangents +
                            fine_operator->matrix_free_tangent_offsets[cell_index] +
                            quadrature_index * strain_count * strain_count;
    const double weight = fine_metadata->prepared.weights[quadrature_index];
    double projected[SS_OPERATOR_P1_MAX_GRADIENTS] = {0.0};
    size_t free_slot;
    for (free_slot = 0U; free_slot < free_count; ++free_slot) {
      const size_t slot = slots[free_slot];
      const size_t coarse_node = slot / dimension;
      const size_t component = slot % dimension;
      const uint8_t *coarse_alpha =
          coarse_descriptor->barycentric_indices + coarse_node * (dimension + 1U);
      size_t barycentric = 0U;
      size_t fine_node;
      while (barycentric <= dimension && coarse_alpha[barycentric] != 1U) {
        ++barycentric;
      }
      if (barycentric > dimension) {
        return SS_ERROR_INTEGRITY;
      }
      for (fine_node = 0U; fine_node < fine_descriptor->node_count; ++fine_node) {
        const size_t fine_slot = fine_node * dimension + component;
        const double coefficient =
            (double)fine_descriptor
                ->barycentric_indices[fine_node * (dimension + 1U) + barycentric] /
            (double)fine_descriptor->order;
        size_t coordinate;
        double fine_gradient[SS_FE_MAX_DIMENSION];
        if (fine_cell->constrained[fine_slot] != 0U || coefficient == 0.0) {
          continue;
        }
        if (ss_operator_mechanics_physical_gradient(fine_operator, cell_index,
                                                    quadrature_index, fine_node,
                                                    fine_gradient) == 0) {
          return SS_ERROR_INTEGRITY;
        }
        for (coordinate = 0U; coordinate < dimension; ++coordinate) {
          projected[slot * dimension + coordinate] +=
              coefficient * fine_gradient[coordinate];
        }
      }
    }
    for (free_slot = 0U; free_slot < free_count; ++free_slot) {
      const size_t row_slot = slots[free_slot];
      const uint32_t row_component = (uint32_t)(row_slot % dimension);
      const double *row_gradient = projected + row_slot * dimension;
      size_t free_column;
      for (free_column = 0U; free_column < free_count; ++free_column) {
        const size_t column_slot = slots[free_column];
        const uint32_t column_component = (uint32_t)(column_slot % dimension);
        const double *column_gradient = projected + column_slot * dimension;
        double basis_strain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double product[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        ss_mechanics_accumulate_strain((uint32_t)dimension, column_gradient,
                                       column_component, 1.0, basis_strain);
        ss_mechanics_tangent_product(tangent, strain_count, basis_strain, 0, product);
        values[free_slot * free_count + free_column] +=
            (PetscScalar)(ss_mechanics_b_dot((uint32_t)dimension, row_gradient,
                                             row_component, product, 0) *
                          weight);
      }
    }
  }
  return SS_OK;
}

SsStatus ss_operator_matrix_free_project_p1_jacobian(const SsOperator *fine_operator,
                                                     const SsAssemblyPlan *p1_plan,
                                                     Mat p1_jacobian,
                                                     SsErrorRecord *error) {
  PetscInt matrix_rows = 0;
  PetscInt matrix_columns = 0;
  PetscScalar *cell_values = NULL;
  PetscScalar *projection_workspace = NULL;
  size_t projection_workspace_stride = 0U;
  size_t projection_workspace_count = 0U;
  int communicator_comparison = MPI_UNEQUAL;
  int computation_failed = 0;
  int use_openmp = 0;
  uint64_t value_bytes = 0U;
  uint64_t workspace_bytes = 0U;
  SsStatus status = SS_OK;
  size_t cell_index;
#if !SLOPE_ENABLE_OPENMP
  (void)use_openmp;
#endif
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
  if (fine_operator == NULL || p1_plan == NULL || p1_jacobian == NULL ||
      fine_operator->spec.physics != SS_ASSEMBLY_MECHANICS ||
      fine_operator->spec.matrix_free == 0 ||
      fine_operator->matrix_free_linearized == 0 ||
      fine_operator->matrix_free_tangents == NULL ||
      fine_operator->properties.dimension < 2U ||
      fine_operator->properties.dimension > 3U ||
      MPI_Comm_compare(p1_plan->communicator, fine_operator->plan->communicator,
                       &communicator_comparison) != MPI_SUCCESS ||
      (communicator_comparison != MPI_IDENT &&
       communicator_comparison != MPI_CONGRUENT) ||
      p1_plan->cell_count != fine_operator->plan->cell_count ||
      p1_plan->summary.global_rows == 0U ||
      p1_plan->summary.global_rows > (uint64_t)PETSC_MAX_INT ||
      MatGetSize(p1_jacobian, &matrix_rows, &matrix_columns) != PETSC_SUCCESS ||
      matrix_rows != (PetscInt)p1_plan->summary.global_rows ||
      matrix_columns != (PetscInt)p1_plan->summary.global_rows) {
    ss_assembly_error_private(error, SS_ERROR_INVALID_ARGUMENT,
                              "ASM_MATFREE_P1_ARGUMENT", 0U,
                              fine_operator == NULL ? 0 : fine_operator->plan->rank);
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_operator_register_events() != PETSC_SUCCESS ||
      PetscLogEventBegin(ss_operator_p1_project_event, p1_jacobian, 0, 0, 0) !=
          PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  for (cell_index = 0U; cell_index < p1_plan->cell_count; ++cell_index) {
    const SsCellPlan *fine_cell = &fine_operator->plan->cells[cell_index];
    const SsCellPlan *coarse_cell = &p1_plan->cells[cell_index];
    const SsOperatorCell *fine_metadata = &fine_operator->cells[cell_index];
    const SsFeDescriptor *fine_descriptor = fine_metadata->field_descriptor;
    const SsFeDescriptor *coarse_descriptor = NULL;
    const size_t coarse_nodes = (size_t)coarse_cell->dimension + 1U;
    if (fine_cell->stable_cell_id != coarse_cell->stable_cell_id ||
        fine_cell->dimension != coarse_cell->dimension ||
        fine_cell->components != coarse_cell->components || coarse_cell->order != 1U ||
        coarse_cell->field_node_count != coarse_nodes ||
        coarse_cell->closure_count != coarse_nodes * (size_t)coarse_cell->dimension ||
        coarse_cell->closure_count > SS_OPERATOR_P1_MAX_CLOSURE ||
        fine_cell->field_node_count != fine_descriptor->node_count ||
        fine_cell->closure_count !=
            fine_descriptor->node_count * (size_t)fine_cell->dimension ||
        fine_descriptor->order == 0U ||
        ss_fe_descriptor_find(coarse_cell->dimension, 1U, &coarse_descriptor) !=
            SS_OK ||
        coarse_descriptor == NULL || coarse_descriptor->node_count != coarse_nodes) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
  }
  status = ss_assembly_collective_status_private(p1_plan->communicator, status);
  if (status != SS_OK || MatZeroEntries(p1_jacobian) != PETSC_SUCCESS) {
    if (status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
    goto finish;
  }
  use_openmp = ss_operator_use_openmp(fine_operator);
  if (fine_operator->matrix_free_p1_reference_gradient_blas_cache != NULL) {
    const size_t worker_count =
        use_openmp != 0 ? (size_t)fine_operator->spec.requested_threads : 1U;
    const size_t point_count = fine_operator->cells[0].quadrature->point_count;
    const size_t strain_count = fine_operator->properties.dimension == 2U ? 3U : 6U;
    if (point_count == 0U ||
        point_count > SIZE_MAX / strain_count / SS_OPERATOR_P1_MAX_CLOSURE / 2U) {
      status = SS_ERROR_OVERFLOW;
    } else {
      projection_workspace_stride =
          2U * point_count * strain_count * SS_OPERATOR_P1_MAX_CLOSURE;
      if (worker_count > SIZE_MAX / projection_workspace_stride) {
        status = SS_ERROR_OVERFLOW;
      } else {
        projection_workspace_count = worker_count * projection_workspace_stride;
      }
    }
  }
  if (status == SS_OK && p1_plan->cell_count > SIZE_MAX / sizeof(*cell_values) /
                                                   (SS_OPERATOR_P1_MAX_CLOSURE *
                                                    SS_OPERATOR_P1_MAX_CLOSURE)) {
    status = SS_ERROR_OVERFLOW;
  } else if (status == SS_OK) {
    const size_t value_count =
        p1_plan->cell_count * SS_OPERATOR_P1_MAX_CLOSURE * SS_OPERATOR_P1_MAX_CLOSURE;
    const uint64_t memory_cap = fine_operator->spec.memory_cap_bytes == 0U
                                    ? SS_ASSEMBLY_DEFAULT_MEMORY_CAP
                                    : (uint64_t)fine_operator->spec.memory_cap_bytes;
    if (ss_operator_checked_bytes(value_count, sizeof(*cell_values), &value_bytes) !=
            SS_OK ||
        ss_operator_checked_bytes(projection_workspace_count,
                                  sizeof(*projection_workspace),
                                  &workspace_bytes) != SS_OK ||
        fine_operator->diagnostics.admitted_peak_bytes > memory_cap ||
        value_bytes > memory_cap - fine_operator->diagnostics.admitted_peak_bytes ||
        workspace_bytes >
            memory_cap - fine_operator->diagnostics.admitted_peak_bytes - value_bytes) {
      status = SS_ERROR_RESOURCE;
    } else {
      cell_values = (PetscScalar *)calloc(value_count == 0U ? 1U : value_count,
                                          sizeof(*cell_values));
      projection_workspace = (PetscScalar *)calloc(
          projection_workspace_count == 0U ? 1U : projection_workspace_count,
          sizeof(*projection_workspace));
      if (cell_values == NULL || projection_workspace == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  status = ss_assembly_collective_status_private(p1_plan->communicator, status);
  if (status != SS_OK) {
    goto finish;
  }
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel for num_threads(                                                  \
        fine_operator->spec.requested_threads) if (use_openmp != 0) schedule(static)   \
    reduction(| : computation_failed)
#endif
  for (cell_index = 0U; cell_index < p1_plan->cell_count; ++cell_index) {
    const size_t thread = (size_t)ss_operator_thread();
    PetscScalar *values = cell_values + cell_index * SS_OPERATOR_P1_MAX_CLOSURE *
                                            SS_OPERATOR_P1_MAX_CLOSURE;
    PetscScalar *workspace =
        projection_workspace + thread * projection_workspace_stride;
    if (ss_operator_matrix_free_compute_p1_cell(fine_operator, p1_plan, cell_index,
                                                values, workspace,
                                                projection_workspace_stride) != SS_OK) {
      computation_failed = 1;
    }
  }
  status = ss_assembly_collective_status_private(
      p1_plan->communicator, computation_failed == 0 ? SS_OK : SS_ERROR_INTEGRITY);
  if (status != SS_OK) {
    goto finish;
  }
  for (cell_index = 0U; cell_index < p1_plan->cell_count; ++cell_index) {
    const SsCellPlan *coarse_cell = &p1_plan->cells[cell_index];
    PetscInt indices[SS_OPERATOR_P1_MAX_CLOSURE];
    size_t slots[SS_OPERATOR_P1_MAX_CLOSURE];
    PetscScalar *values = cell_values + cell_index * SS_OPERATOR_P1_MAX_CLOSURE *
                                            SS_OPERATOR_P1_MAX_CLOSURE;
    size_t free_count = 0U;
    size_t coarse_slot;
    for (coarse_slot = 0U; coarse_slot < coarse_cell->closure_count; ++coarse_slot) {
      if (coarse_cell->constrained[coarse_slot] == 0U &&
          coarse_cell->global_indices[coarse_slot] >= 0) {
        indices[free_count] = coarse_cell->global_indices[coarse_slot];
        slots[free_count] = coarse_slot;
        ++free_count;
      }
    }
    {
      size_t free_row;
      for (free_row = 0U; free_row < free_count; ++free_row) {
        const size_t row_slot = slots[free_row];
        if (coarse_cell->owned[row_slot] == 0U) {
          continue;
        }
        if (MatSetValues(p1_jacobian, 1, &indices[free_row], (PetscInt)free_count,
                         indices, values + free_row * free_count,
                         ADD_VALUES) != PETSC_SUCCESS) {
          status = SS_ERROR_INTERNAL;
          break;
        }
      }
    }
    if (status != SS_OK) {
      break;
    }
  }
  status = ss_assembly_collective_status_private(p1_plan->communicator, status);
  if (MatAssemblyBegin(p1_jacobian, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS ||
      MatAssemblyEnd(p1_jacobian, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_assembly_collective_status_private(p1_plan->communicator, status);

finish:
  free(projection_workspace);
  free(cell_values);
  if (PetscLogEventEnd(ss_operator_p1_project_event, p1_jacobian, 0, 0, 0) !=
          PETSC_SUCCESS &&
      status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, "ASM_MATFREE_P1_PROJECT", 0U,
                              p1_plan->rank);
  }
  return status;
}

static double ss_operator_matrix_free_projected_cell_entry(
    const SsOperator *fine_operator, const SsOperator *coarse_operator,
    size_t cell_index, size_t row_slot, size_t column_slot) {
  const SsCellPlan *fine_cell = &fine_operator->plan->cells[cell_index];
  const SsOperatorCell *fine_metadata = &fine_operator->cells[cell_index];
  const size_t dimension = fine_cell->dimension;
  const size_t strain_count = dimension == 2U ? 3U : 6U;
  const uint32_t row_component = (uint32_t)(row_slot % dimension);
  const uint32_t column_component = (uint32_t)(column_slot % dimension);
  double value = 0.0;
  size_t quadrature_index;
  for (quadrature_index = 0U; quadrature_index < fine_metadata->quadrature->point_count;
       ++quadrature_index) {
    double row_gradient[SS_FE_MAX_DIMENSION];
    double column_gradient[SS_FE_MAX_DIMENSION];
    const double *tangent = fine_operator->matrix_free_tangents +
                            fine_operator->matrix_free_tangent_offsets[cell_index] +
                            quadrature_index * strain_count * strain_count;
    double basis_strain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double product[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (ss_operator_matrix_free_projected_physical_gradient(
            fine_operator, coarse_operator, cell_index, quadrature_index, row_slot,
            row_gradient) == 0 ||
        ss_operator_matrix_free_projected_physical_gradient(
            fine_operator, coarse_operator, cell_index, quadrature_index, column_slot,
            column_gradient) == 0) {
      return NAN;
    }
    ss_mechanics_accumulate_strain((uint32_t)dimension, column_gradient,
                                   column_component, 1.0, basis_strain);
    ss_mechanics_tangent_product(tangent, strain_count, basis_strain, 0, product);
    value += ss_mechanics_b_dot((uint32_t)dimension, row_gradient, row_component,
                                product, 0) *
             fine_metadata->prepared.weights[quadrature_index];
  }
  return value;
}

SsStatus ss_operator_matrix_free_project_coarse(const SsOperator *fine_operator,
                                                SsOperator *coarse_operator,
                                                SsErrorRecord *error) {
  double transfer[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
  const SsFeDescriptor *fine_descriptor = NULL;
  const SsFeDescriptor *coarse_descriptor = NULL;
  size_t transfer_rows = 0U;
  size_t transfer_columns = 0U;
  size_t gradient_count = 0U;
  size_t reference_gradient_count = 0U;
  uint64_t projection_bytes = 0U;
  uint64_t offset_bytes = 0U;
  uint64_t gradient_bytes = 0U;
  uint64_t transfer_bytes = 0U;
  uint64_t constrained_bytes = 0U;
  uint64_t constraint_pattern_index_bytes = 0U;
  uint64_t constraint_pattern_representative_bytes = 0U;
  uint64_t constraint_pattern_gradient_bytes = 0U;
  uint64_t physical_pattern_index_bytes = 0U;
  uint64_t physical_pattern_representative_bytes = 0U;
  uint64_t physical_pattern_gradient_bytes = 0U;
  uint64_t reference_gradient_bytes = 0U;
  uint64_t reference_gradient_blas_bytes = 0U;
  size_t cell_index;
  size_t block_index;
  size_t constraint_pattern_count = 0U;
  size_t constraint_pattern_gradient_count = 0U;
  size_t physical_pattern_count = 0U;
  size_t physical_pattern_gradient_count = 0U;
  size_t *new_constraint_pattern_indices = NULL;
  size_t *new_constraint_pattern_representatives = NULL;
  size_t *new_physical_pattern_indices = NULL;
  size_t *new_physical_pattern_representatives = NULL;
  int communicator_comparison = MPI_UNEQUAL;
  int compact_affine_projection = 0;
  int cache_projection_physical_patterns = 0;
  int block_failure = 0;
  SsStatus status = SS_OK;
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
  if (fine_operator == NULL || coarse_operator == NULL ||
      fine_operator->spec.physics != SS_ASSEMBLY_MECHANICS ||
      coarse_operator->spec.physics != SS_ASSEMBLY_MECHANICS ||
      fine_operator->spec.matrix_free == 0 || coarse_operator->spec.matrix_free == 0 ||
      fine_operator->matrix_free_linearized == 0 ||
      fine_operator->matrix_free_tangents == NULL ||
      coarse_operator->matrix_free_operator == NULL ||
      coarse_operator->matrix_free_diagonal == NULL ||
      coarse_operator->matrix_free_inverse_blocks == NULL ||
      MPI_Comm_compare(fine_operator->plan->communicator,
                       coarse_operator->plan->communicator,
                       &communicator_comparison) != MPI_SUCCESS ||
      (communicator_comparison != MPI_IDENT &&
       communicator_comparison != MPI_CONGRUENT) ||
      fine_operator->plan->cell_count != coarse_operator->plan->cell_count ||
      fine_operator->plan->cell_count == 0U) {
    ss_assembly_error_private(error, SS_ERROR_INVALID_ARGUMENT,
                              "ASM_MATFREE_COARSE_ARGUMENT", 0U,
                              fine_operator == NULL ? 0 : fine_operator->plan->rank);
    return SS_ERROR_INVALID_ARGUMENT;
  }
  fine_descriptor = fine_operator->cells[0].field_descriptor;
  coarse_descriptor = coarse_operator->cells[0].field_descriptor;
  if (fine_descriptor == NULL || coarse_descriptor == NULL ||
      coarse_descriptor->dimension != fine_descriptor->dimension ||
      coarse_descriptor->order >= fine_descriptor->order ||
      fine_descriptor->node_count > SS_FE_MAX_NODES ||
      coarse_descriptor->node_count > SS_FE_MAX_NODES ||
      fine_operator->reference_gradient_cache == NULL ||
      ss_fe_transfer_build(coarse_descriptor, fine_descriptor, transfer,
                           SS_FE_MAX_NODES * SS_FE_MAX_NODES, &transfer_rows,
                           &transfer_columns) != SS_OK ||
      transfer_rows != fine_descriptor->node_count ||
      transfer_columns != coarse_descriptor->node_count) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK) {
    const size_t point_count = fine_operator->cells[0].quadrature->point_count;
    const size_t dimension = fine_descriptor->dimension;
    if (point_count > SIZE_MAX / coarse_descriptor->node_count ||
        point_count * coarse_descriptor->node_count > SIZE_MAX / dimension) {
      status = SS_ERROR_OVERFLOW;
    } else {
      reference_gradient_count =
          point_count * coarse_descriptor->node_count * dimension;
      compact_affine_projection = dimension == 3U ? 1 : 0;
    }
  }
  for (cell_index = 0U; status == SS_OK && cell_index < fine_operator->plan->cell_count;
       ++cell_index) {
    const SsCellPlan *fine_cell = &fine_operator->plan->cells[cell_index];
    const SsCellPlan *coarse_cell = &coarse_operator->plan->cells[cell_index];
    if (fine_cell->stable_cell_id != coarse_cell->stable_cell_id ||
        fine_cell->dimension != coarse_cell->dimension ||
        fine_cell->components != coarse_cell->components ||
        fine_operator->cells[cell_index].field_descriptor != fine_descriptor ||
        coarse_operator->cells[cell_index].field_descriptor != coarse_descriptor ||
        coarse_cell->closure_count !=
            coarse_descriptor->node_count * (size_t)coarse_cell->dimension ||
        fine_cell->closure_count !=
            fine_descriptor->node_count * (size_t)fine_cell->dimension ||
        fine_operator->cells[cell_index].quadrature == NULL ||
        fine_operator->cells[cell_index].prepared.weights == NULL) {
      status = SS_ERROR_INTEGRITY;
      continue;
    }
    if (fine_operator->cells[cell_index].affine_geometry == 0) {
      compact_affine_projection = 0;
    }
  }
  {
    int global_compact_affine_projection = 0;
    if (MPI_Allreduce(&compact_affine_projection, &global_compact_affine_projection, 1,
                      MPI_INT, MPI_MIN,
                      fine_operator->plan->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      compact_affine_projection = global_compact_affine_projection;
    }
  }
  for (cell_index = 0U; status == SS_OK && compact_affine_projection == 0 &&
                        cell_index < fine_operator->plan->cell_count;
       ++cell_index) {
    const SsCellPlan *coarse_cell = &coarse_operator->plan->cells[cell_index];
    const size_t point_count = fine_operator->cells[cell_index].quadrature->point_count;
    if (point_count != 0U &&
        coarse_cell->closure_count > (SIZE_MAX - gradient_count) / point_count /
                                         (size_t)coarse_cell->dimension) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    gradient_count +=
        point_count * coarse_cell->closure_count * (size_t)coarse_cell->dimension;
  }
  status =
      ss_assembly_collective_status_private(fine_operator->plan->communicator, status);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, "ASM_MATFREE_COARSE_PROJECT", 0U,
                              fine_operator->plan->rank);
    return status;
  }
  if (coarse_operator->matrix_free_projection_source != NULL &&
      coarse_operator->matrix_free_projection_source != fine_operator) {
    status = SS_ERROR_INVALID_STATE;
  }
  if (status == SS_OK && coarse_operator->matrix_free_projection_source == NULL) {
    const size_t cell_count = fine_operator->plan->cell_count;
    if (cell_count > SIZE_MAX / sizeof(*new_constraint_pattern_indices) ||
        cell_count > SIZE_MAX / sizeof(*new_constraint_pattern_representatives) ||
        cell_count > SIZE_MAX / sizeof(*new_physical_pattern_indices) ||
        cell_count > SIZE_MAX / sizeof(*new_physical_pattern_representatives)) {
      status = SS_ERROR_OVERFLOW;
    } else {
      new_constraint_pattern_indices =
          (size_t *)malloc(cell_count * sizeof(*new_constraint_pattern_indices));
      new_constraint_pattern_representatives = (size_t *)malloc(
          cell_count * sizeof(*new_constraint_pattern_representatives));
      new_physical_pattern_indices =
          (size_t *)malloc(cell_count * sizeof(*new_physical_pattern_indices));
      new_physical_pattern_representatives =
          (size_t *)malloc(cell_count * sizeof(*new_physical_pattern_representatives));
    }
    if (status == SS_OK && (new_constraint_pattern_indices == NULL ||
                            new_constraint_pattern_representatives == NULL ||
                            new_physical_pattern_indices == NULL ||
                            new_physical_pattern_representatives == NULL)) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    for (cell_index = 0U; status == SS_OK && cell_index < cell_count; ++cell_index) {
      const SsCellPlan *cell = &fine_operator->plan->cells[cell_index];
      size_t pattern;
      for (pattern = 0U; pattern < constraint_pattern_count; ++pattern) {
        const SsCellPlan *representative =
            &fine_operator->plan
                 ->cells[new_constraint_pattern_representatives[pattern]];
        if (cell->closure_count == representative->closure_count &&
            memcmp(cell->constrained, representative->constrained,
                   cell->closure_count * sizeof(*cell->constrained)) == 0) {
          break;
        }
      }
      if (pattern == constraint_pattern_count) {
        new_constraint_pattern_representatives[constraint_pattern_count++] = cell_index;
      }
      new_constraint_pattern_indices[cell_index] = pattern;
    }
    cache_projection_physical_patterns = compact_affine_projection;
    for (cell_index = 0U; status == SS_OK && cache_projection_physical_patterns != 0 &&
                          cell_index < cell_count;
         ++cell_index) {
      const SsOperatorCell *metadata = &fine_operator->cells[cell_index];
      size_t pattern;
      for (pattern = 0U; pattern < physical_pattern_count; ++pattern) {
        const size_t representative_index =
            new_physical_pattern_representatives[pattern];
        const SsOperatorCell *representative =
            &fine_operator->cells[representative_index];
        if (new_constraint_pattern_indices[cell_index] ==
                new_constraint_pattern_indices[representative_index] &&
            memcmp(metadata->inverse_transpose, representative->inverse_transpose,
                   fine_descriptor->dimension * fine_descriptor->dimension *
                       sizeof(*metadata->inverse_transpose)) == 0) {
          break;
        }
      }
      if (pattern == physical_pattern_count) {
        if (physical_pattern_count ==
            SS_OPERATOR_MATRIX_FREE_MAX_PROJECTION_PHYSICAL_PATTERNS) {
          cache_projection_physical_patterns = 0;
          physical_pattern_count = 0U;
          break;
        }
        new_physical_pattern_representatives[physical_pattern_count++] = cell_index;
      }
      new_physical_pattern_indices[cell_index] = pattern;
    }
  } else if (status == SS_OK) {
    constraint_pattern_count =
        coarse_operator->matrix_free_projection_constraint_pattern_count;
    physical_pattern_count =
        coarse_operator->matrix_free_projection_physical_pattern_count;
    cache_projection_physical_patterns = physical_pattern_count == 0U ? 0 : 1;
  }
  if (status == SS_OK &&
      (constraint_pattern_count == 0U ||
       constraint_pattern_count > SIZE_MAX / fine_descriptor->dimension ||
       constraint_pattern_count * fine_descriptor->dimension >
           SIZE_MAX / reference_gradient_count)) {
    status = SS_ERROR_OVERFLOW;
  } else if (status == SS_OK) {
    constraint_pattern_gradient_count = constraint_pattern_count *
                                        fine_descriptor->dimension *
                                        reference_gradient_count;
    if (cache_projection_physical_patterns != 0 &&
        (physical_pattern_count > SIZE_MAX / fine_descriptor->dimension ||
         physical_pattern_count * fine_descriptor->dimension >
             SIZE_MAX / reference_gradient_count)) {
      status = SS_ERROR_OVERFLOW;
    } else if (cache_projection_physical_patterns != 0) {
      physical_pattern_gradient_count = physical_pattern_count *
                                        fine_descriptor->dimension *
                                        reference_gradient_count;
    }
  }
  status =
      ss_assembly_collective_status_private(fine_operator->plan->communicator, status);
  if (status != SS_OK) {
    free(new_constraint_pattern_indices);
    free(new_constraint_pattern_representatives);
    free(new_physical_pattern_indices);
    free(new_physical_pattern_representatives);
    ss_assembly_error_private(error, status, "ASM_MATFREE_COARSE_PATTERN", 0U,
                              fine_operator->plan->rank);
    return status;
  }
  if (status == SS_OK &&
      coarse_operator->matrix_free_projection_gradient_offsets == NULL) {
    const uint64_t memory_cap = coarse_operator->spec.memory_cap_bytes == 0U
                                    ? SS_ASSEMBLY_DEFAULT_MEMORY_CAP
                                    : (uint64_t)coarse_operator->spec.memory_cap_bytes;
    if (fine_operator->plan->cell_count >
            SIZE_MAX /
                    sizeof(*coarse_operator->matrix_free_projection_gradient_offsets) -
                1U ||
        gradient_count >
            SIZE_MAX / sizeof(*coarse_operator->matrix_free_projection_gradients) ||
        ss_operator_checked_bytes(
            fine_operator->plan->cell_count + 1U,
            sizeof(*coarse_operator->matrix_free_projection_gradient_offsets),
            &offset_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            gradient_count, sizeof(*coarse_operator->matrix_free_projection_gradients),
            &gradient_bytes) != SS_OK ||
        transfer_rows > SIZE_MAX / transfer_columns ||
        ss_operator_checked_bytes(
            transfer_rows * transfer_columns,
            sizeof(*coarse_operator->matrix_free_projection_transfer),
            &transfer_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            fine_operator->plan->cell_count,
            sizeof(*coarse_operator->matrix_free_projection_source_constrained),
            &constrained_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            fine_operator->plan->cell_count,
            sizeof(*coarse_operator->matrix_free_projection_constraint_pattern_indices),
            &constraint_pattern_index_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            constraint_pattern_count,
            sizeof(*coarse_operator
                        ->matrix_free_projection_constraint_pattern_representatives),
            &constraint_pattern_representative_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            constraint_pattern_gradient_count,
            sizeof(*coarse_operator
                        ->matrix_free_projection_constraint_reference_blas_cache),
            &constraint_pattern_gradient_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            cache_projection_physical_patterns != 0 ? fine_operator->plan->cell_count
                                                    : 0U,
            sizeof(*coarse_operator->matrix_free_projection_physical_pattern_indices),
            &physical_pattern_index_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            physical_pattern_count,
            sizeof(*coarse_operator
                        ->matrix_free_projection_physical_pattern_representatives),
            &physical_pattern_representative_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            physical_pattern_gradient_count,
            sizeof(*coarse_operator->matrix_free_projection_pattern_physical_gradients),
            &physical_pattern_gradient_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            reference_gradient_count,
            sizeof(*coarse_operator->matrix_free_projection_reference_gradients),
            &reference_gradient_bytes) != SS_OK ||
        ss_operator_checked_bytes(
            reference_gradient_count,
            sizeof(
                *coarse_operator->matrix_free_projection_reference_gradient_blas_cache),
            &reference_gradient_blas_bytes) != SS_OK ||
        offset_bytes > UINT64_MAX - gradient_bytes ||
        offset_bytes + gradient_bytes > UINT64_MAX - transfer_bytes ||
        offset_bytes + gradient_bytes + transfer_bytes >
            UINT64_MAX - constrained_bytes ||
        offset_bytes + gradient_bytes + transfer_bytes + constrained_bytes >
            UINT64_MAX - constraint_pattern_index_bytes ||
        offset_bytes + gradient_bytes + transfer_bytes + constrained_bytes +
                constraint_pattern_index_bytes >
            UINT64_MAX - constraint_pattern_representative_bytes ||
        offset_bytes + gradient_bytes + transfer_bytes + constrained_bytes +
                constraint_pattern_index_bytes +
                constraint_pattern_representative_bytes >
            UINT64_MAX - constraint_pattern_gradient_bytes ||
        offset_bytes + gradient_bytes + transfer_bytes + constrained_bytes +
                constraint_pattern_index_bytes +
                constraint_pattern_representative_bytes +
                constraint_pattern_gradient_bytes >
            UINT64_MAX - physical_pattern_index_bytes ||
        offset_bytes + gradient_bytes + transfer_bytes + constrained_bytes +
                constraint_pattern_index_bytes +
                constraint_pattern_representative_bytes +
                constraint_pattern_gradient_bytes + physical_pattern_index_bytes >
            UINT64_MAX - physical_pattern_representative_bytes ||
        offset_bytes + gradient_bytes + transfer_bytes + constrained_bytes +
                constraint_pattern_index_bytes +
                constraint_pattern_representative_bytes +
                constraint_pattern_gradient_bytes + physical_pattern_index_bytes +
                physical_pattern_representative_bytes >
            UINT64_MAX - physical_pattern_gradient_bytes ||
        offset_bytes + gradient_bytes + transfer_bytes + constrained_bytes +
                constraint_pattern_index_bytes +
                constraint_pattern_representative_bytes +
                constraint_pattern_gradient_bytes + physical_pattern_index_bytes +
                physical_pattern_representative_bytes +
                physical_pattern_gradient_bytes >
            UINT64_MAX - reference_gradient_bytes ||
        offset_bytes + gradient_bytes + transfer_bytes + constrained_bytes +
                constraint_pattern_index_bytes +
                constraint_pattern_representative_bytes +
                constraint_pattern_gradient_bytes + physical_pattern_index_bytes +
                physical_pattern_representative_bytes +
                physical_pattern_gradient_bytes + reference_gradient_bytes >
            UINT64_MAX - reference_gradient_blas_bytes) {
      status = SS_ERROR_OVERFLOW;
    } else {
      projection_bytes =
          offset_bytes + gradient_bytes + transfer_bytes + constrained_bytes +
          constraint_pattern_index_bytes + constraint_pattern_representative_bytes +
          constraint_pattern_gradient_bytes + physical_pattern_index_bytes +
          physical_pattern_representative_bytes + physical_pattern_gradient_bytes +
          reference_gradient_bytes + reference_gradient_blas_bytes;
      if (projection_bytes > memory_cap ||
          coarse_operator->diagnostics.admitted_peak_bytes >
              memory_cap - projection_bytes) {
        status = SS_ERROR_RESOURCE;
      }
    }
    if (status == SS_OK) {
      coarse_operator->matrix_free_projection_gradient_offsets = (size_t *)calloc(
          fine_operator->plan->cell_count + 1U,
          sizeof(*coarse_operator->matrix_free_projection_gradient_offsets));
      coarse_operator->matrix_free_projection_gradients =
          (double *)calloc(gradient_count == 0U ? 1U : gradient_count,
                           sizeof(*coarse_operator->matrix_free_projection_gradients));
      coarse_operator->matrix_free_projection_transfer =
          (double *)malloc(transfer_rows * transfer_columns *
                           sizeof(*coarse_operator->matrix_free_projection_transfer));
      coarse_operator->matrix_free_projection_source_constrained =
          (unsigned char *)calloc(
              fine_operator->plan->cell_count == 0U ? 1U
                                                    : fine_operator->plan->cell_count,
              sizeof(*coarse_operator->matrix_free_projection_source_constrained));
      coarse_operator->matrix_free_projection_constraint_pattern_indices =
          (size_t *)malloc(
              fine_operator->plan->cell_count *
              sizeof(
                  *coarse_operator->matrix_free_projection_constraint_pattern_indices));
      coarse_operator->matrix_free_projection_constraint_pattern_representatives =
          (size_t *)malloc(
              constraint_pattern_count *
              sizeof(*coarse_operator
                          ->matrix_free_projection_constraint_pattern_representatives));
      coarse_operator->matrix_free_projection_constraint_reference_blas_cache =
          (PetscScalar *)calloc(
              constraint_pattern_gradient_count == 0U
                  ? 1U
                  : constraint_pattern_gradient_count,
              sizeof(*coarse_operator
                          ->matrix_free_projection_constraint_reference_blas_cache));
      if (cache_projection_physical_patterns != 0) {
        coarse_operator->matrix_free_projection_physical_pattern_indices =
            (size_t *)malloc(
                fine_operator->plan->cell_count *
                sizeof(
                    *coarse_operator->matrix_free_projection_physical_pattern_indices));
        coarse_operator->matrix_free_projection_physical_pattern_representatives =
            (size_t *)malloc(
                physical_pattern_count *
                sizeof(*coarse_operator
                            ->matrix_free_projection_physical_pattern_representatives));
        coarse_operator->matrix_free_projection_pattern_physical_gradients =
            (double *)calloc(
                physical_pattern_gradient_count == 0U ? 1U
                                                      : physical_pattern_gradient_count,
                sizeof(*coarse_operator
                            ->matrix_free_projection_pattern_physical_gradients));
      }
      coarse_operator->matrix_free_projection_reference_gradients = (double *)calloc(
          reference_gradient_count == 0U ? 1U : reference_gradient_count,
          sizeof(*coarse_operator->matrix_free_projection_reference_gradients));
      coarse_operator->matrix_free_projection_reference_gradient_blas_cache =
          (PetscScalar *)calloc(
              reference_gradient_count == 0U ? 1U : reference_gradient_count,
              sizeof(*coarse_operator
                          ->matrix_free_projection_reference_gradient_blas_cache));
      if (coarse_operator->matrix_free_projection_gradient_offsets == NULL ||
          coarse_operator->matrix_free_projection_gradients == NULL ||
          coarse_operator->matrix_free_projection_transfer == NULL ||
          coarse_operator->matrix_free_projection_source_constrained == NULL ||
          coarse_operator->matrix_free_projection_constraint_pattern_indices == NULL ||
          coarse_operator->matrix_free_projection_constraint_pattern_representatives ==
              NULL ||
          coarse_operator->matrix_free_projection_constraint_reference_blas_cache ==
              NULL ||
          (cache_projection_physical_patterns != 0 &&
           (coarse_operator->matrix_free_projection_physical_pattern_indices == NULL ||
            coarse_operator->matrix_free_projection_physical_pattern_representatives ==
                NULL ||
            coarse_operator->matrix_free_projection_pattern_physical_gradients ==
                NULL)) ||
          coarse_operator->matrix_free_projection_reference_gradients == NULL ||
          coarse_operator->matrix_free_projection_reference_gradient_blas_cache ==
              NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      } else {
        memcpy(coarse_operator->matrix_free_projection_transfer, transfer,
               transfer_rows * transfer_columns * sizeof(*transfer));
        coarse_operator->matrix_free_projection_transfer_rows = transfer_rows;
        coarse_operator->matrix_free_projection_transfer_columns = transfer_columns;
        memcpy(coarse_operator->matrix_free_projection_constraint_pattern_indices,
               new_constraint_pattern_indices,
               fine_operator->plan->cell_count *
                   sizeof(*new_constraint_pattern_indices));
        memcpy(
            coarse_operator->matrix_free_projection_constraint_pattern_representatives,
            new_constraint_pattern_representatives,
            constraint_pattern_count * sizeof(*new_constraint_pattern_representatives));
        coarse_operator->matrix_free_projection_constraint_pattern_count =
            constraint_pattern_count;
        if (cache_projection_physical_patterns != 0) {
          memcpy(coarse_operator->matrix_free_projection_physical_pattern_indices,
                 new_physical_pattern_indices,
                 fine_operator->plan->cell_count *
                     sizeof(*new_physical_pattern_indices));
          memcpy(
              coarse_operator->matrix_free_projection_physical_pattern_representatives,
              new_physical_pattern_representatives,
              physical_pattern_count * sizeof(*new_physical_pattern_representatives));
          coarse_operator->matrix_free_projection_physical_pattern_count =
              physical_pattern_count;
        }
        coarse_operator->diagnostics.matrix_free_bytes += projection_bytes;
        coarse_operator->diagnostics.admitted_peak_bytes += projection_bytes;
      }
    }
  } else if (status == SS_OK &&
             (coarse_operator->matrix_free_projection_gradient_count !=
                  gradient_count ||
              coarse_operator->matrix_free_projection_reference_gradient_count !=
                  reference_gradient_count ||
              coarse_operator->matrix_free_projection_transfer_rows != transfer_rows ||
              coarse_operator->matrix_free_projection_transfer_columns !=
                  transfer_columns ||
              coarse_operator->matrix_free_projection_constraint_pattern_count !=
                  constraint_pattern_count ||
              coarse_operator->matrix_free_projection_physical_pattern_count !=
                  physical_pattern_count)) {
    status = SS_ERROR_INTEGRITY;
  }
  free(new_constraint_pattern_indices);
  free(new_constraint_pattern_representatives);
  free(new_physical_pattern_indices);
  free(new_physical_pattern_representatives);
  new_constraint_pattern_indices = NULL;
  new_constraint_pattern_representatives = NULL;
  new_physical_pattern_indices = NULL;
  new_physical_pattern_representatives = NULL;
  status =
      ss_assembly_collective_status_private(fine_operator->plan->communicator, status);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, "ASM_MATFREE_COARSE_CACHE", 0U,
                              fine_operator->plan->rank);
    return status;
  }
  if (coarse_operator->matrix_free_projection_source == NULL) {
    size_t gradient_offset = 0U;
    size_t reference_quadrature_index;
    for (cell_index = 0U; cell_index < fine_operator->plan->cell_count; ++cell_index) {
      const SsCellPlan *fine_cell = &fine_operator->plan->cells[cell_index];
      size_t fine_slot;
      for (fine_slot = 0U; fine_slot < fine_cell->closure_count; ++fine_slot) {
        if (fine_cell->constrained[fine_slot] != 0U) {
          coarse_operator->matrix_free_projection_source_constrained[cell_index] = 1U;
          break;
        }
      }
    }
    for (reference_quadrature_index = 0U;
         reference_quadrature_index < fine_operator->cells[0].quadrature->point_count;
         ++reference_quadrature_index) {
      const double *fine_reference_gradients =
          fine_operator->reference_gradient_cache + reference_quadrature_index *
                                                        fine_descriptor->node_count *
                                                        fine_descriptor->dimension;
      size_t coarse_node;
      for (coarse_node = 0U; coarse_node < coarse_descriptor->node_count;
           ++coarse_node) {
        size_t coordinate;
        for (coordinate = 0U; coordinate < fine_descriptor->dimension; ++coordinate) {
          double value = 0.0;
          size_t fine_node;
          for (fine_node = 0U; fine_node < fine_descriptor->node_count; ++fine_node) {
            value += transfer[fine_node * transfer_columns + coarse_node] *
                     fine_reference_gradients[fine_node * fine_descriptor->dimension +
                                              coordinate];
          }
          coarse_operator->matrix_free_projection_reference_gradients
              [(reference_quadrature_index * coarse_descriptor->node_count +
                coarse_node) *
                   fine_descriptor->dimension +
               coordinate] = value;
          coarse_operator->matrix_free_projection_reference_gradient_blas_cache
              [coarse_node * fine_descriptor->dimension *
                   fine_operator->cells[0].quadrature->point_count +
               coordinate * fine_operator->cells[0].quadrature->point_count +
               reference_quadrature_index] = (PetscScalar)value;
        }
      }
    }
    coarse_operator->matrix_free_projection_reference_gradient_count =
        reference_gradient_count;
    if (compact_affine_projection != 0) {
      PetscScalar masked_transfer[SS_FE_MAX_NODES * SS_FE_MAX_NODES];
      const size_t point_count = fine_operator->cells[0].quadrature->point_count;
      const size_t dimension = fine_descriptor->dimension;
      const size_t stacked_points = dimension * point_count;
      const PetscBLASInt m = (PetscBLASInt)stacked_points;
      const PetscBLASInt n = (PetscBLASInt)coarse_descriptor->node_count;
      const PetscBLASInt k = (PetscBLASInt)fine_descriptor->node_count;
      const PetscScalar one = 1.0;
      const PetscScalar zero = 0.0;
      size_t pattern;
      for (pattern = 0U; pattern < constraint_pattern_count; ++pattern) {
        const size_t representative_index =
            coarse_operator
                ->matrix_free_projection_constraint_pattern_representatives[pattern];
        const SsCellPlan *representative =
            &fine_operator->plan->cells[representative_index];
        size_t component;
        for (component = 0U; component < dimension; ++component) {
          size_t coarse_node;
          for (coarse_node = 0U; coarse_node < coarse_descriptor->node_count;
               ++coarse_node) {
            size_t fine_node;
            for (fine_node = 0U; fine_node < fine_descriptor->node_count; ++fine_node) {
              masked_transfer[fine_node + coarse_node * fine_descriptor->node_count] =
                  representative->constrained[fine_node * dimension + component] != 0U
                      ? 0.0
                      : (PetscScalar)
                            transfer[fine_node * transfer_columns + coarse_node];
            }
          }
          BLASgemm_(
              "N", "N", &m, &n, &k, &one, fine_operator->reference_gradient_blas_cache,
              &m, masked_transfer, &k, &zero,
              coarse_operator->matrix_free_projection_constraint_reference_blas_cache +
                  (pattern * dimension + component) * coarse_descriptor->node_count *
                      stacked_points,
              &m);
        }
      }
    }
    if (cache_projection_physical_patterns != 0) {
      const size_t point_count = fine_operator->cells[0].quadrature->point_count;
      const size_t dimension = fine_descriptor->dimension;
      const size_t stacked_points = dimension * point_count;
      const size_t closure_count = coarse_descriptor->node_count * dimension;
      const size_t entries_per_pattern = point_count * closure_count * dimension;
      size_t physical_pattern;
      for (physical_pattern = 0U; physical_pattern < physical_pattern_count;
           ++physical_pattern) {
        const size_t representative_index =
            coarse_operator->matrix_free_projection_physical_pattern_representatives
                [physical_pattern];
        const SsOperatorCell *representative =
            &fine_operator->cells[representative_index];
        const size_t constraint_pattern =
            coarse_operator->matrix_free_projection_constraint_pattern_indices
                [representative_index];
        double *physical =
            coarse_operator->matrix_free_projection_pattern_physical_gradients +
            physical_pattern * entries_per_pattern;
        size_t quadrature_index;
        for (quadrature_index = 0U; quadrature_index < point_count;
             ++quadrature_index) {
          size_t coarse_slot;
          for (coarse_slot = 0U; coarse_slot < closure_count; ++coarse_slot) {
            const size_t coarse_node = coarse_slot / dimension;
            const size_t component = coarse_slot % dimension;
            const PetscScalar *reference =
                coarse_operator
                    ->matrix_free_projection_constraint_reference_blas_cache +
                (constraint_pattern * dimension + component) *
                    coarse_descriptor->node_count * stacked_points +
                coarse_node * stacked_points;
            size_t coordinate;
            for (coordinate = 0U; coordinate < dimension; ++coordinate) {
              size_t reference_coordinate;
              double value = 0.0;
              for (reference_coordinate = 0U; reference_coordinate < dimension;
                   ++reference_coordinate) {
                value +=
                    representative->inverse_transpose[coordinate * dimension +
                                                      reference_coordinate] *
                    (double)PetscRealPart(reference[reference_coordinate * point_count +
                                                    quadrature_index]);
              }
              physical[(quadrature_index * closure_count + coarse_slot) * dimension +
                       coordinate] = value;
            }
          }
        }
      }
    }
    for (cell_index = 0U;
         compact_affine_projection == 0 && cell_index < fine_operator->plan->cell_count;
         ++cell_index) {
      const SsCellPlan *fine_cell = &fine_operator->plan->cells[cell_index];
      const SsCellPlan *coarse_cell = &coarse_operator->plan->cells[cell_index];
      const SsOperatorCell *fine_metadata = &fine_operator->cells[cell_index];
      const size_t dimension = fine_cell->dimension;
      size_t quadrature_index;
      coarse_operator->matrix_free_projection_gradient_offsets[cell_index] =
          gradient_offset;
      for (quadrature_index = 0U;
           quadrature_index < fine_metadata->quadrature->point_count;
           ++quadrature_index) {
        size_t coarse_slot;
        for (coarse_slot = 0U; coarse_slot < coarse_cell->closure_count;
             ++coarse_slot) {
          const size_t coarse_node = coarse_slot / dimension;
          const size_t component = coarse_slot % dimension;
          double *projected =
              coarse_operator->matrix_free_projection_gradients + gradient_offset;
          size_t fine_node;
          for (fine_node = 0U; fine_node < fine_descriptor->node_count; ++fine_node) {
            const size_t fine_slot = fine_node * dimension + component;
            const double coefficient =
                transfer[fine_node * transfer_columns + coarse_node];
            size_t coordinate;
            double fine_gradient[SS_FE_MAX_DIMENSION];
            if (fine_cell->constrained[fine_slot] != 0U || coefficient == 0.0) {
              continue;
            }
            if (ss_operator_mechanics_physical_gradient(fine_operator, cell_index,
                                                        quadrature_index, fine_node,
                                                        fine_gradient) == 0) {
              status = SS_ERROR_INTEGRITY;
              break;
            }
            for (coordinate = 0U; coordinate < dimension; ++coordinate) {
              projected[coordinate] += coefficient * fine_gradient[coordinate];
            }
          }
          if (status != SS_OK) {
            break;
          }
          gradient_offset += dimension;
        }
        if (status != SS_OK) {
          break;
        }
      }
      if (status != SS_OK) {
        break;
      }
    }
    coarse_operator
        ->matrix_free_projection_gradient_offsets[fine_operator->plan->cell_count] =
        gradient_offset;
    if (gradient_offset != gradient_count) {
      status = SS_ERROR_INTEGRITY;
    } else {
      coarse_operator->matrix_free_projection_gradient_count = gradient_count;
      coarse_operator->matrix_free_projection_source = fine_operator;
    }
  }
  status =
      ss_assembly_collective_status_private(fine_operator->plan->communicator, status);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, "ASM_MATFREE_COARSE_CACHE", 0U,
                              fine_operator->plan->rank);
    return status;
  }
  status = ss_operator_matrix_free_reserve_block_workspace(
      coarse_operator, fine_operator->cells[0].quadrature->point_count,
      fine_descriptor->dimension == 2U ? 3U : 6U);
  status =
      ss_assembly_collective_status_private(fine_operator->plan->communicator, status);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, "ASM_MATFREE_COARSE_WORKSPACE", 0U,
                              fine_operator->plan->rank);
    return status;
  }
  (void)PetscLogEventBegin(ss_operator_matrix_free_block_setup_event, 0, 0, 0, 0);
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel for num_threads(                                                  \
        coarse_operator->spec                                                          \
            .requested_threads) if (ss_operator_use_openmp(coarse_operator) != 0)      \
    schedule(static) reduction(| : block_failure)
#endif
  for (block_index = 0U; block_index < (size_t)coarse_operator->matrix_free_block_count;
       ++block_index) {
    const size_t block_row_start =
        coarse_operator->matrix_free_block_row_offsets[block_index];
    const PetscInt block_size = coarse_operator->matrix_free_block_sizes[block_index];
    PetscScalar *inverse = coarse_operator->matrix_free_inverse_blocks +
                           coarse_operator->matrix_free_block_offsets[block_index];
    PetscInt block_row;
    int valid = 1;
    if (block_size > 3 ||
        (coarse_operator->spec.determinism == SS_ASSEMBLY_PERFORMANCE &&
         coarse_operator->matrix_free_block_workspace_stride != 0U)) {
      const size_t thread = (size_t)ss_operator_thread();
      if (thread >= (size_t)coarse_operator->spec.requested_threads ||
          coarse_operator->matrix_free_block_workspace_stride == 0U) {
        valid = 0;
      } else {
        PetscScalar *workspace =
            coarse_operator->matrix_free_block_workspace +
            thread * coarse_operator->matrix_free_block_workspace_stride;
        valid = ss_operator_matrix_free_entity_block(coarse_operator, block_row_start,
                                                     block_size, inverse, workspace);
      }
    } else {
      for (block_row = 0; valid != 0 && block_row < block_size; ++block_row) {
        const SsOwnedRowPlan *row =
            &coarse_operator->plan->rows[block_row_start + (size_t)block_row];
        PetscInt block_column;
        for (block_column = 0; block_column < block_size; ++block_column) {
          const SsOwnedRowPlan *column =
              &coarse_operator->plan->rows[block_row_start + (size_t)block_column];
          double entry = 0.0;
          size_t contribution;
          for (contribution = 0U; contribution < row->cell_count; ++contribution) {
            const size_t coarse_cell_number = row->cell_slots[contribution].cell_index;
            const SsCellPlan *coarse_cell =
                &coarse_operator->plan->cells[coarse_cell_number];
            const size_t column_slot =
                ss_operator_matrix_free_cell_slot(coarse_cell, column);
            if (column_slot == SIZE_MAX ||
                coarse_cell->constrained[column_slot] != 0U) {
              valid = 0;
              break;
            }
            entry += ss_operator_matrix_free_projected_cell_entry(
                fine_operator, coarse_operator, coarse_cell_number,
                row->cell_slots[contribution].row_slot, column_slot);
          }
          inverse[block_row + block_column * block_size] = (PetscScalar)entry;
        }
      }
    }
    for (block_row = 0; valid != 0 && block_row < block_size; ++block_row) {
      coarse_operator->matrix_free_diagonal[block_row_start + (size_t)block_row] =
          (double)PetscRealPart(inverse[block_row + block_row * block_size]);
    }
    if (valid == 0 || ss_operator_matrix_free_invert_block(block_size, inverse) == 0) {
      block_failure = 1;
    }
  }
  block_failure =
      block_failure || !ss_operator_matrix_free_inverses_finite(coarse_operator);
  (void)PetscLogEventEnd(ss_operator_matrix_free_block_setup_event, 0, 0, 0, 0);
  status = ss_assembly_collective_status_private(
      fine_operator->plan->communicator,
      block_failure == 0 ? SS_OK : SS_ERROR_INTEGRITY);
  if (status == SS_OK) {
    coarse_operator->matrix_free_linearized = 1;
    ++coarse_operator->matrix_free_generation;
    if (MatAssemblyBegin(coarse_operator->matrix_free_operator, MAT_FINAL_ASSEMBLY) !=
            PETSC_SUCCESS ||
        MatAssemblyEnd(coarse_operator->matrix_free_operator, MAT_FINAL_ASSEMBLY) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  if (status != SS_OK) {
    coarse_operator->matrix_free_linearized = 0;
    ss_assembly_error_private(error, status, "ASM_MATFREE_COARSE_PROJECT", 0U,
                              fine_operator->plan->rank);
  }
  return status;
}
