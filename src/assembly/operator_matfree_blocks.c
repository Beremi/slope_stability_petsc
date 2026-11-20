/* Build and apply matrix-free variable-block inverses.
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
#include <petscvec.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <float.h>
#include <math.h>
#include <string.h>

double ss_operator_matrix_free_cell_entry(const SsOperator *operator_handle,
                                          size_t cell_index, size_t row_slot,
                                          size_t column_slot) {
  const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
  const SsOperatorCell *metadata = &operator_handle->cells[cell_index];
  const size_t strain_count = cell->dimension == 2U ? 3U : 6U;
  const uint32_t row_component = (uint32_t)(row_slot % cell->dimension);
  const uint32_t column_component = (uint32_t)(column_slot % cell->dimension);
  double value = 0.0;
  size_t quadrature_index;
  for (quadrature_index = 0U; quadrature_index < metadata->quadrature->point_count;
       ++quadrature_index) {
    double row_gradient[SS_FE_MAX_DIMENSION];
    double column_gradient[SS_FE_MAX_DIMENSION];
    const double *tangent = operator_handle->matrix_free_tangents +
                            operator_handle->matrix_free_tangent_offsets[cell_index] +
                            quadrature_index * strain_count * strain_count;
    double basis_strain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double product[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (ss_operator_mechanics_physical_gradient(
            operator_handle, cell_index, quadrature_index, row_slot / cell->dimension,
            row_gradient) == 0 ||
        ss_operator_mechanics_physical_gradient(
            operator_handle, cell_index, quadrature_index,
            column_slot / cell->dimension, column_gradient) == 0) {
      return NAN;
    }
    ss_mechanics_accumulate_strain(cell->dimension, column_gradient, column_component,
                                   1.0, basis_strain);
    ss_mechanics_tangent_product(tangent, strain_count, basis_strain, 0, product);
    value +=
        ss_mechanics_b_dot(cell->dimension, row_gradient, row_component, product, 0) *
        metadata->prepared.weights[quadrature_index];
  }
  return value;
}

size_t ss_operator_matrix_free_cell_slot(const SsCellPlan *cell,
                                         const SsOwnedRowPlan *row) {
  size_t slot;
  for (slot = 0U; slot < cell->closure_count; ++slot) {
    if (cell->field_entity_ids[slot] == row->field_entity_id &&
        cell->field_entity_slots[slot] == row->field_entity_slot &&
        cell->field_components[slot] == row->component) {
      return slot;
    }
  }
  return SIZE_MAX;
}

/* Form a larger topological-entity diagonal block as a sum of element
 * B^T D B products.  The point-block path below deliberately retains its
 * historic scalar accumulation order; for larger blocks, one BLAS product
 * per incident cell avoids repeating the quadrature/tangent walk for every
 * matrix entry. */
int ss_operator_matrix_free_entity_block(const SsOperator *operator_handle,
                                         size_t block_row_start, PetscInt block_size,
                                         PetscScalar *inverse, PetscScalar *workspace) {
  const SsOperator *tangent_operator;
  const SsOwnedRowPlan *first;
  size_t contribution;
  if (operator_handle == NULL || inverse == NULL || workspace == NULL ||
      block_size <= 0 ||
      block_size > (PetscInt)SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS ||
      block_row_start + (size_t)block_size > operator_handle->plan->row_count) {
    return 0;
  }
  tangent_operator = operator_handle->matrix_free_projection_source == NULL
                         ? operator_handle
                         : operator_handle->matrix_free_projection_source;
  first = &operator_handle->plan->rows[block_row_start];
  memset(inverse, 0, (size_t)block_size * (size_t)block_size * sizeof(*inverse));
  for (contribution = 0U; contribution < first->cell_count; ++contribution) {
    const size_t cell_number = first->cell_slots[contribution].cell_index;
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_number];
    const SsOperatorCell *metadata = &tangent_operator->cells[cell_number];
    const size_t strain_count = cell->dimension == 2U ? 3U : 6U;
    const size_t point_count = metadata->quadrature->point_count;
    const size_t stacked_strains = point_count * strain_count;
    const size_t workspace_entries = stacked_strains * (size_t)block_size;
    PetscScalar *basis = workspace;
    PetscScalar *product = workspace + workspace_entries;
    PetscBLASInt m;
    PetscBLASInt n;
    PetscScalar one = 1.0;
    PetscScalar beta = contribution == 0U ? 0.0 : 1.0;
    PetscInt column;
    if (stacked_strains == 0U ||
        workspace_entries > operator_handle->matrix_free_block_workspace_stride / 2U ||
        stacked_strains > (size_t)PETSC_BLAS_INT_MAX ||
        block_size > (PetscInt)PETSC_BLAS_INT_MAX) {
      return 0;
    }
    m = (PetscBLASInt)stacked_strains;
    n = (PetscBLASInt)block_size;
    for (column = 0; column < block_size; ++column) {
      const SsOwnedRowPlan *column_row =
          &operator_handle->plan->rows[block_row_start + (size_t)column];
      const size_t column_slot = ss_operator_matrix_free_cell_slot(cell, column_row);
      size_t quadrature_index;
      if (column_slot == SIZE_MAX || cell->constrained[column_slot] != 0U) {
        return 0;
      }
      for (quadrature_index = 0U; quadrature_index < point_count; ++quadrature_index) {
        double gradient[SS_FE_MAX_DIMENSION];
        const uint32_t component = (uint32_t)(column_slot % cell->dimension);
        const double *tangent =
            tangent_operator->matrix_free_tangents +
            tangent_operator->matrix_free_tangent_offsets[cell_number] +
            quadrature_index * strain_count * strain_count;
        double basis_strain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double tangent_product[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        size_t strain;
        if ((operator_handle->matrix_free_projection_source == NULL &&
             ss_operator_mechanics_physical_gradient(
                 operator_handle, cell_number, quadrature_index,
                 column_slot / cell->dimension, gradient) == 0) ||
            (operator_handle->matrix_free_projection_source != NULL &&
             ss_operator_matrix_free_projected_physical_gradient(
                 tangent_operator, operator_handle, cell_number, quadrature_index,
                 column_slot, gradient) == 0)) {
          return 0;
        }
        ss_mechanics_accumulate_strain(cell->dimension, gradient, component, 1.0,
                                       basis_strain);
        ss_mechanics_tangent_product(tangent, strain_count, basis_strain, 0,
                                     tangent_product);
        for (strain = 0U; strain < strain_count; ++strain) {
          const size_t entry = (size_t)column * stacked_strains +
                               quadrature_index * strain_count + strain;
          basis[entry] = (PetscScalar)basis_strain[strain];
          product[entry] = (PetscScalar)(tangent_product[strain] *
                                         metadata->prepared.weights[quadrature_index]);
        }
      }
    }
    BLASgemm_("T", "N", &n, &n, &m, &one, basis, &m, product, &m, &beta, inverse, &n);
  }
  return first->cell_count != 0U;
}

int ss_operator_matrix_free_invert_block(PetscInt size, PetscScalar *values) {
  PetscScalar augmented[3][6] = {{0.0}};
  PetscInt row;
  PetscInt column;
  if (values == NULL || size <= 0 ||
      size > (PetscInt)SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS) {
    return 0;
  }
  if (size > 3) {
    PetscBLASInt pivot[SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS];
    PetscScalar work[SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS *
                     SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS];
    const PetscBLASInt n = (PetscBLASInt)size;
    const PetscBLASInt lwork = (PetscBLASInt)(SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS *
                                              SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS);
    PetscBLASInt info = 0;
    /* Mechanics tangents are Hessians and the positive elastic
     * regularization normally makes every entity restriction SPD.  Factor a
     * copy so the general LU path remains an exact fallback for a genuinely
     * indefinite or numerically marginal block. */
    memcpy(work, values, (size_t)size * (size_t)size * sizeof(*work));
    LAPACKpotrf_("L", &n, work, &n, &info);
    if (info == 0) {
      LAPACKpotri_("L", &n, work, &n, &info);
    }
    if (info == 0) {
      for (column = 0; column < size; ++column) {
        for (row = 0; row < size; ++row) {
          values[row + column * size] = row >= column
                                            ? work[row + column * size]
                                            : PetscConj(work[column + row * size]);
        }
      }
      return 1;
    }
    info = 0;
    LAPACKgetrf_(&n, &n, values, &n, pivot, &info);
    if (info != 0) {
      return 0;
    }
    LAPACKgetri_(&n, values, &n, pivot, work, &lwork, &info);
    if (info != 0) {
      return 0;
    }
    return 1;
  }
  for (row = 0; row < size; ++row) {
    for (column = 0; column < size; ++column) {
      augmented[row][column] = values[row + column * size];
    }
    augmented[row][size + row] = 1.0;
  }
  for (column = 0; column < size; ++column) {
    PetscInt pivot = column;
    PetscReal pivot_magnitude = PetscAbsScalar(augmented[pivot][column]);
    PetscReal column_scale = pivot_magnitude;
    for (row = column + 1; row < size; ++row) {
      const PetscReal magnitude = PetscAbsScalar(augmented[row][column]);
      if (magnitude > pivot_magnitude) {
        pivot = row;
        pivot_magnitude = magnitude;
      }
      if (magnitude > column_scale) {
        column_scale = magnitude;
      }
    }
    if (!isfinite((double)pivot_magnitude) ||
        pivot_magnitude <= DBL_EPSILON * fmax(1.0, (double)column_scale)) {
      return 0;
    }
    if (pivot != column) {
      PetscInt entry;
      for (entry = 0; entry < 2 * size; ++entry) {
        const PetscScalar temporary = augmented[column][entry];
        augmented[column][entry] = augmented[pivot][entry];
        augmented[pivot][entry] = temporary;
      }
    }
    {
      const PetscScalar diagonal = augmented[column][column];
      PetscInt entry;
      for (entry = 0; entry < 2 * size; ++entry) {
        augmented[column][entry] /= diagonal;
      }
    }
    for (row = 0; row < size; ++row) {
      PetscInt entry;
      const PetscScalar multiplier = augmented[row][column];
      if (row == column) {
        continue;
      }
      for (entry = 0; entry < 2 * size; ++entry) {
        augmented[row][entry] -= multiplier * augmented[column][entry];
      }
    }
  }
  for (row = 0; row < size; ++row) {
    for (column = 0; column < size; ++column) {
      values[row + column * size] = augmented[row][size + column];
    }
  }
  return 1;
}

/* Inversion reports singularity in the worker loop. Check the packed inverse
 * once on the calling thread before publishing the new tangent generation. */
int ss_operator_matrix_free_inverses_finite(const SsOperator *operator_handle) {
  const PetscScalar *inverse = operator_handle->matrix_free_inverse_blocks;
  const size_t entry_count =
      operator_handle
          ->matrix_free_block_offsets[operator_handle->matrix_free_block_count];
  for (size_t entry = 0U; entry < entry_count; ++entry) {
    if (PetscIsInfOrNanScalar(inverse[entry])) {
      return 0;
    }
  }
  return 1;
}

PetscErrorCode
ss_operator_matrix_free_invert_variable_block_diagonal(Mat matrix, PetscInt block_count,
                                                       const PetscInt *block_sizes,
                                                       PetscScalar *values) {
  SsOperator *operator_handle = NULL;
  PetscInt block;
  PetscErrorCode ierr = MatShellGetContext(matrix, &operator_handle);
  if (ierr != PETSC_SUCCESS) {
    return ierr;
  }
  if (operator_handle == NULL || operator_handle->matrix_free_linearized == 0 ||
      block_count != operator_handle->matrix_free_block_count ||
      (block_count != 0 && (block_sizes == NULL || values == NULL))) {
    return PETSC_ERR_ARG_WRONGSTATE;
  }
  for (block = 0; block < block_count; ++block) {
    if (block_sizes[block] != operator_handle->matrix_free_block_sizes[block]) {
      return PETSC_ERR_ARG_INCOMP;
    }
  }
  memcpy(values, operator_handle->matrix_free_inverse_blocks,
         operator_handle->matrix_free_block_offsets[block_count] * sizeof(*values));
  return PETSC_SUCCESS;
}

/*
 * PCVPBJACOBI correctly consumes MATOP_INVERT_VBLOCK_DIAGONAL, but its local
 * dense-block application is serial inside an MPI rank.  Entity blocks are
 * deliberately much larger than the historical nodal blocks at P8, so that
 * serial apply becomes the dominant part of a hybrid V-cycle.  Publish this
 * matrix-specific operation for PMG's PC shell: PETSc still owns the KSP and
 * V-cycle, while the independent local blocks are spread across the OpenMP
 * team without making PETSc calls from worker threads.
 */
PetscErrorCode ss_operator_matrix_free_apply_inverse_variable_blocks(Mat matrix,
                                                                     Vec input,
                                                                     Vec output) {
  SsOperator *operator_handle = NULL;
  const PetscScalar *input_values = NULL;
  PetscScalar *output_values = NULL;
  PetscInt input_size = 0;
  PetscInt output_size = 0;
  int use_openmp = 0;
  PetscErrorCode ierr;
#if !SLOPE_ENABLE_OPENMP
  (void)use_openmp;
#endif
  ierr = MatShellGetContext(matrix, &operator_handle);
  if (ierr != PETSC_SUCCESS) {
    return ierr;
  }
  if (operator_handle == NULL || operator_handle->matrix_free_linearized == 0 ||
      operator_handle->matrix_free_threaded_block_apply == 0 ||
      operator_handle->matrix_free_block_count <= 0 ||
      operator_handle->matrix_free_block_sizes == NULL ||
      operator_handle->matrix_free_block_row_offsets == NULL ||
      operator_handle->matrix_free_block_offsets == NULL ||
      operator_handle->matrix_free_inverse_blocks == NULL || input == output ||
      VecGetLocalSize(input, &input_size) != PETSC_SUCCESS ||
      VecGetLocalSize(output, &output_size) != PETSC_SUCCESS ||
      input_size != output_size || input_size < 0 ||
      (size_t)input_size != operator_handle->plan->row_count ||
      operator_handle->matrix_free_block_row_offsets[operator_handle
                                                         ->matrix_free_block_count] !=
          (size_t)input_size) {
    return PETSC_ERR_ARG_WRONGSTATE;
  }
  use_openmp = ss_operator_use_openmp(operator_handle);
  if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK ||
      VecGetArrayRead(input, &input_values) != PETSC_SUCCESS ||
      VecGetArrayWrite(output, &output_values) != PETSC_SUCCESS) {
    if (input_values != NULL) {
      (void)VecRestoreArrayRead(input, &input_values);
    }
    return PETSC_ERR_LIB;
  }
  if (PetscLogEventBegin(ss_operator_matrix_free_block_apply_event, matrix, input,
                         output, 0) != PETSC_SUCCESS) {
    (void)VecRestoreArrayWrite(output, &output_values);
    (void)VecRestoreArrayRead(input, &input_values);
    return PETSC_ERR_LIB;
  }
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel for num_threads(                                                  \
        operator_handle->spec.requested_threads) if (use_openmp != 0) schedule(static)
#endif
  for (PetscInt block = 0; block < operator_handle->matrix_free_block_count; ++block) {
    const PetscInt block_size = operator_handle->matrix_free_block_sizes[block];
    const size_t row_offset = operator_handle->matrix_free_block_row_offsets[block];
    const PetscScalar *inverse = operator_handle->matrix_free_inverse_blocks +
                                 operator_handle->matrix_free_block_offsets[block];
    PetscInt row;
    for (row = 0; row < block_size; ++row) {
      output_values[row_offset + (size_t)row] = 0.0;
    }
    /* Column-major traversal matches LAPACK/MatInvertVariableBlockDiagonal
     * storage and gives each inner loop contiguous reads and writes. */
    for (PetscInt column = 0; column < block_size; ++column) {
      const PetscScalar input_value = input_values[row_offset + (size_t)column];
      const PetscScalar *inverse_column = inverse + (size_t)column * (size_t)block_size;
#if SLOPE_ENABLE_OPENMP
#pragma omp simd
#endif
      for (row = 0; row < block_size; ++row) {
        output_values[row_offset + (size_t)row] += inverse_column[row] * input_value;
      }
    }
  }
  ierr = VecRestoreArrayWrite(output, &output_values);
  {
    const PetscErrorCode input_status = VecRestoreArrayRead(input, &input_values);
    const PetscErrorCode event_status = PetscLogEventEnd(
        ss_operator_matrix_free_block_apply_event, matrix, input, output, 0);
    if (ierr != PETSC_SUCCESS || input_status != PETSC_SUCCESS ||
        event_status != PETSC_SUCCESS) {
      return PETSC_ERR_LIB;
    }
  }
  return PETSC_SUCCESS;
}
