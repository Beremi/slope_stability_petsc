/* Reference-table tetrahedral residual and matrix-free action kernels for affine
 * and curved cells (J^{-T} per cell or per quadrature point).
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "operator_internal.h"
#include "voigt.h"

#include "slope_internal/build/config.h"

#include <petscblaslapack.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <string.h>

SsStatus ss_operator_matrix_free_reference_residual_cell_3d(
    const SsOperator *operator_handle, size_t cell_index, const SsOperatorState *state,
    uint32_t request, SsElementOutput *output, SsErrorRecord *error) {
  const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
  const SsOperatorCell *metadata = &operator_handle->cells[cell_index];
  const size_t point_count = metadata->quadrature->point_count;
  const size_t node_count = metadata->field_descriptor->node_count;
  const size_t closure_count = 3U * node_count;
  const double *field_values =
      operator_handle->packed_values + cell_index * operator_handle->maximum_closure;
  PetscScalar element_values[SS_FE_MAX_NODES * 3U];
  PetscScalar displacement[SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE * 3U];
  PetscScalar
      reference_derivatives[3U * SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE * 3U];
  PetscScalar reference_fluxes[3U * SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE * 3U];
  PetscScalar load_fluxes[SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE * 3U];
  PetscScalar element_residual[SS_FE_MAX_NODES * 3U];
  const PetscScalar one = 1.0;
  const PetscScalar zero = 0.0;
  PetscBLASInt q;
  PetscBLASInt stacked_q;
  PetscBLASInt p;
  const PetscBLASInt components = 3;
  double force[3];
  size_t node;
  size_t coordinate;
  size_t quadrature_index;
  if (operator_handle == NULL || state == NULL || output == NULL || error == NULL ||
      cell->dimension != 3U || metadata->prepared.inverse_transposes == NULL ||
      operator_handle->basis_blas_cache == NULL ||
      operator_handle->reference_gradient_blas_cache == NULL ||
      point_count > SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE ||
      node_count > SS_FE_MAX_NODES || cell->closure_count != closure_count ||
      output->closure_capacity < closure_count ||
      (request & SS_ASSEMBLY_JACOBIAN) != 0U ||
      (request & (SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_POTENTIAL)) == 0U ||
      operator_handle->mechanics.coupling_point_kernel != NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  q = (PetscBLASInt)point_count;
  stacked_q = 3 * q;
  p = (PetscBLASInt)node_count;
  memset(output->residual, 0, closure_count * sizeof(*output->residual));
  output->potential = 0.0;
  for (coordinate = 0U; coordinate < 3U; ++coordinate) {
    force[coordinate] =
        (metadata->body_force == NULL
             ? 0.0
             : state->external_load_factor * metadata->body_force[coordinate]) +
        operator_handle->mechanics.fixed_coupling_force[coordinate];
  }
  for (node = 0U; node < node_count; ++node) {
    element_values[node] = (PetscScalar)field_values[3U * node];
    element_values[node_count + node] = (PetscScalar)field_values[3U * node + 1U];
    element_values[2U * node_count + node] = (PetscScalar)field_values[3U * node + 2U];
  }
  BLASgemm_("N", "N", &q, &components, &p, &one, operator_handle->basis_blas_cache, &q,
            element_values, &p, &zero, displacement, &q);
  BLASgemm_("N", "N", &stacked_q, &components, &p, &one,
            operator_handle->reference_gradient_blas_cache, &stacked_q, element_values,
            &p, &zero, reference_derivatives, &stacked_q);
  for (quadrature_index = 0U; quadrature_index < point_count; ++quadrature_index) {
    const double *inverse_transpose =
        ss_operator_cell_inverse_transpose(metadata, quadrature_index);
    const double weight = metadata->prepared.weights[quadrature_index];
    double physical_gradient[9];
    double strain[6];
    double physical_flux[9];
    SsMcResponse response = {0};
    SsMcError material_error = {0};
    SsStatus status;
    for (coordinate = 0U; coordinate < 3U; ++coordinate) {
      size_t component;
      for (component = 0U; component < 3U; ++component) {
        physical_gradient[component * 3U + coordinate] = (double)PetscRealPart(
            reference_derivatives[component * (size_t)stacked_q +
                                  coordinate * point_count + quadrature_index]);
      }
    }
    {
      double transformed[9];
      size_t component;
      for (component = 0U; component < 3U; ++component) {
        const double *reference = physical_gradient + component * 3U;
        transformed[component * 3U] = inverse_transpose[0] * reference[0] +
                                      inverse_transpose[1] * reference[1] +
                                      inverse_transpose[2] * reference[2];
        transformed[component * 3U + 1U] = inverse_transpose[3] * reference[0] +
                                           inverse_transpose[4] * reference[1] +
                                           inverse_transpose[5] * reference[2];
        transformed[component * 3U + 2U] = inverse_transpose[6] * reference[0] +
                                           inverse_transpose[7] * reference[1] +
                                           inverse_transpose[8] * reference[2];
      }
      memcpy(physical_gradient, transformed, sizeof(physical_gradient));
    }
    strain[0] = physical_gradient[0];
    strain[1] = physical_gradient[4];
    strain[2] = physical_gradient[8];
    strain[3] = physical_gradient[1] + physical_gradient[3];
    strain[4] = physical_gradient[5] + physical_gradient[7];
    strain[5] = physical_gradient[2] + physical_gradient[6];
    status = ss_mc_evaluate_prepared(metadata->material, SS_MC_DIMENSION_3,
                                     metadata->prepared.reduced_material, 0U, strain,
                                     &response, &material_error);
    if (status != SS_OK) {
      ss_assembly_error_private(error, status,
                                material_error.code[0] == '\0' ? "ASM_ELEMENT_MATERIAL"
                                                               : material_error.code,
                                cell->stable_cell_id, operator_handle->plan->rank);
      error->key.quadrature = (uint64_t)quadrature_index;
      return status;
    }
    if ((request & SS_ASSEMBLY_POTENTIAL) != 0U) {
      const double external =
          (double)PetscRealPart(displacement[quadrature_index]) * force[0] +
          (double)PetscRealPart(displacement[point_count + quadrature_index]) *
              force[1] +
          (double)PetscRealPart(displacement[2U * point_count + quadrature_index]) *
              force[2];
      output->potential += (response.potential - external) * weight;
    }
    if ((request & SS_ASSEMBLY_RESIDUAL) == 0U) {
      continue;
    }
    physical_flux[0] = response.stress[0];
    physical_flux[1] = response.stress[3];
    physical_flux[2] = response.stress[5];
    physical_flux[3] = response.stress[3];
    physical_flux[4] = response.stress[1];
    physical_flux[5] = response.stress[4];
    physical_flux[6] = response.stress[5];
    physical_flux[7] = response.stress[4];
    physical_flux[8] = response.stress[2];
    for (coordinate = 0U; coordinate < 3U; ++coordinate) {
      size_t component;
      for (component = 0U; component < 3U; ++component) {
        const double *physical = physical_flux + component * 3U;
        reference_fluxes[component * (size_t)stacked_q + coordinate * point_count +
                         quadrature_index] =
            (PetscScalar)((inverse_transpose[coordinate] * physical[0] +
                           inverse_transpose[3U + coordinate] * physical[1] +
                           inverse_transpose[6U + coordinate] * physical[2]) *
                          weight);
      }
    }
    load_fluxes[quadrature_index] = (PetscScalar)(-force[0] * weight);
    load_fluxes[point_count + quadrature_index] = (PetscScalar)(-force[1] * weight);
    load_fluxes[2U * point_count + quadrature_index] =
        (PetscScalar)(-force[2] * weight);
  }
  if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
    BLASgemm_("T", "N", &p, &components, &stacked_q, &one,
              operator_handle->reference_gradient_blas_cache, &stacked_q,
              reference_fluxes, &stacked_q, &zero, element_residual, &p);
    BLASgemm_("T", "N", &p, &components, &q, &one, operator_handle->basis_blas_cache,
              &q, load_fluxes, &q, &one, element_residual, &p);
    for (node = 0U; node < node_count; ++node) {
      const size_t slot = 3U * node;
      output->residual[slot] = (double)PetscRealPart(element_residual[node]);
      output->residual[slot + 1U] =
          (double)PetscRealPart(element_residual[node_count + node]);
      output->residual[slot + 2U] =
          (double)PetscRealPart(element_residual[2U * node_count + node]);
    }
  }
  if (!isfinite(output->potential)) {
    ss_assembly_error_private(error, SS_ERROR_INTEGRITY, "ASM_ELEMENT_NONFINITE_OUTPUT",
                              cell->stable_cell_id, operator_handle->plan->rank);
    return SS_ERROR_INTEGRITY;
  }
  return SS_OK;
}

/*
 * Affine tetrahedra share one reference-space basis table across every cell.
 * Applying the physical-gradient form cached per cell streams
 * Q * P * dimension doubles for each MatMult.  At P8 that table is much larger
 * than the useful tangent data and makes the action memory-bandwidth bound.
 *
 * Accumulate the displacement gradient in reference space, transform the small
 * 3-by-3 gradient once per quadrature point, and transform the stress flux back
 * before B^T.  This is algebraically identical to using
 * grad_x(N) = J^{-T} grad_X(N), while only reading the shared reference table.
 */
void ss_operator_matrix_free_reference_cell_action_3d(
    const SsOperator *action_operator, const SsOperator *tangent_operator,
    size_t cell_index, const double *reference_gradients, size_t node_count,
    const PetscScalar *gathered, int transpose) {
  const SsCellPlan *cell = &action_operator->plan->cells[cell_index];
  const SsOperatorCell *source_metadata = &tangent_operator->cells[cell_index];
  const size_t closure_count = cell->closure_count;
  const size_t packed_offset = cell_index * action_operator->maximum_closure;
  const size_t *packed_slots = action_operator->packed_slots + packed_offset;
  double *action = action_operator->matrix_free_element_action + packed_offset;
  double element_values[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  size_t quadrature_index;
  size_t node;
  if (cell->dimension != 3U || closure_count != 3U * node_count ||
      node_count > SS_FE_MAX_NODES || reference_gradients == NULL ||
      source_metadata->prepared.inverse_transposes == NULL) {
    return;
  }
  memset(action, 0, closure_count * sizeof(*action));
  for (node = 0U; node < node_count; ++node) {
    const size_t slot = 3U * node;
    const size_t x_slot = packed_slots[slot];
    const size_t y_slot = packed_slots[slot + 1U];
    const size_t z_slot = packed_slots[slot + 2U];
    element_values[slot] =
        x_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[x_slot]);
    element_values[slot + 1U] =
        y_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[y_slot]);
    element_values[slot + 2U] =
        z_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[z_slot]);
  }
  for (quadrature_index = 0U;
       quadrature_index < source_metadata->quadrature->point_count;
       ++quadrature_index) {
    const double *inverse_transpose =
        ss_operator_cell_inverse_transpose(source_metadata, quadrature_index);
    const double *gradients = reference_gradients + quadrature_index * node_count * 3U;
    const double *tangent = tangent_operator->matrix_free_tangents +
                            tangent_operator->matrix_free_tangent_offsets[cell_index] +
                            quadrature_index * 36U;
    const double weight = source_metadata->prepared.weights[quadrature_index];
    double reference_displacement_gradient[9] = {0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0};
    double strain[6];
    double product[6];
    double reference_flux[9];
#if SLOPE_ENABLE_OPENMP
#pragma omp simd reduction(+ : reference_displacement_gradient[0 : 9])
#endif
    for (node = 0U; node < node_count; ++node) {
      const size_t slot = 3U * node;
      const double *gradient = gradients + 3U * node;
      const double x = element_values[slot];
      const double y = element_values[slot + 1U];
      const double z = element_values[slot + 2U];
      reference_displacement_gradient[0] += gradient[0] * x;
      reference_displacement_gradient[1] += gradient[1] * x;
      reference_displacement_gradient[2] += gradient[2] * x;
      reference_displacement_gradient[3] += gradient[0] * y;
      reference_displacement_gradient[4] += gradient[1] * y;
      reference_displacement_gradient[5] += gradient[2] * y;
      reference_displacement_gradient[6] += gradient[0] * z;
      reference_displacement_gradient[7] += gradient[1] * z;
      reference_displacement_gradient[8] += gradient[2] * z;
    }
    {
      const double ux_x = inverse_transpose[0] * reference_displacement_gradient[0] +
                          inverse_transpose[1] * reference_displacement_gradient[1] +
                          inverse_transpose[2] * reference_displacement_gradient[2];
      const double ux_y = inverse_transpose[3] * reference_displacement_gradient[0] +
                          inverse_transpose[4] * reference_displacement_gradient[1] +
                          inverse_transpose[5] * reference_displacement_gradient[2];
      const double ux_z = inverse_transpose[6] * reference_displacement_gradient[0] +
                          inverse_transpose[7] * reference_displacement_gradient[1] +
                          inverse_transpose[8] * reference_displacement_gradient[2];
      const double uy_x = inverse_transpose[0] * reference_displacement_gradient[3] +
                          inverse_transpose[1] * reference_displacement_gradient[4] +
                          inverse_transpose[2] * reference_displacement_gradient[5];
      const double uy_y = inverse_transpose[3] * reference_displacement_gradient[3] +
                          inverse_transpose[4] * reference_displacement_gradient[4] +
                          inverse_transpose[5] * reference_displacement_gradient[5];
      const double uy_z = inverse_transpose[6] * reference_displacement_gradient[3] +
                          inverse_transpose[7] * reference_displacement_gradient[4] +
                          inverse_transpose[8] * reference_displacement_gradient[5];
      const double uz_x = inverse_transpose[0] * reference_displacement_gradient[6] +
                          inverse_transpose[1] * reference_displacement_gradient[7] +
                          inverse_transpose[2] * reference_displacement_gradient[8];
      const double uz_y = inverse_transpose[3] * reference_displacement_gradient[6] +
                          inverse_transpose[4] * reference_displacement_gradient[7] +
                          inverse_transpose[5] * reference_displacement_gradient[8];
      const double uz_z = inverse_transpose[6] * reference_displacement_gradient[6] +
                          inverse_transpose[7] * reference_displacement_gradient[7] +
                          inverse_transpose[8] * reference_displacement_gradient[8];
      strain[0] = ux_x;
      strain[1] = uy_y;
      strain[2] = uz_z;
      strain[3] = ux_y + uy_x;
      strain[4] = uy_z + uz_y;
      strain[5] = ux_z + uz_x;
    }
    ss_mechanics_tangent_product(tangent, 6U, strain, transpose, product);
    reference_flux[0] = inverse_transpose[0] * product[0] +
                        inverse_transpose[3] * product[3] +
                        inverse_transpose[6] * product[5];
    reference_flux[1] = inverse_transpose[1] * product[0] +
                        inverse_transpose[4] * product[3] +
                        inverse_transpose[7] * product[5];
    reference_flux[2] = inverse_transpose[2] * product[0] +
                        inverse_transpose[5] * product[3] +
                        inverse_transpose[8] * product[5];
    reference_flux[3] = inverse_transpose[0] * product[3] +
                        inverse_transpose[3] * product[1] +
                        inverse_transpose[6] * product[4];
    reference_flux[4] = inverse_transpose[1] * product[3] +
                        inverse_transpose[4] * product[1] +
                        inverse_transpose[7] * product[4];
    reference_flux[5] = inverse_transpose[2] * product[3] +
                        inverse_transpose[5] * product[1] +
                        inverse_transpose[8] * product[4];
    reference_flux[6] = inverse_transpose[0] * product[5] +
                        inverse_transpose[3] * product[4] +
                        inverse_transpose[6] * product[2];
    reference_flux[7] = inverse_transpose[1] * product[5] +
                        inverse_transpose[4] * product[4] +
                        inverse_transpose[7] * product[2];
    reference_flux[8] = inverse_transpose[2] * product[5] +
                        inverse_transpose[5] * product[4] +
                        inverse_transpose[8] * product[2];
#if SLOPE_ENABLE_OPENMP
#pragma omp simd
#endif
    for (node = 0U; node < node_count; ++node) {
      const size_t slot = 3U * node;
      const double *gradient = gradients + 3U * node;
      action[slot] +=
          (gradient[0] * reference_flux[0] + gradient[1] * reference_flux[1] +
           gradient[2] * reference_flux[2]) *
          weight;
      action[slot + 1U] +=
          (gradient[0] * reference_flux[3] + gradient[1] * reference_flux[4] +
           gradient[2] * reference_flux[5]) *
          weight;
      action[slot + 2U] +=
          (gradient[0] * reference_flux[6] + gradient[1] * reference_flux[7] +
           gradient[2] * reference_flux[8]) *
          weight;
    }
  }
}

/* Each displacement component occupies one column of stacked reference
 * derivatives. Transform into physical strain, apply the material tangent,
 * then transform the weighted flux back into the same column layout. */
static void ss_operator_matrix_free_reference_flux_3d(
    const SsOperatorCell *metadata, const double *tangents, size_t component_stride,
    const PetscScalar *restrict derivatives, PetscScalar *restrict fluxes,
    int transpose) {
  const size_t point_count = metadata->quadrature->point_count;
#if SLOPE_ENABLE_OPENMP
#pragma omp simd
#endif
  for (size_t point = 0U; point < point_count; ++point) {
    const double *map = ss_operator_cell_inverse_transpose(metadata, point);
    const double weight = metadata->prepared.weights[point];
    double gradient[9];
    double strain[6];
    double product[6];
    for (size_t component = 0U; component < 3U; ++component) {
      const PetscScalar *reference = derivatives + component * component_stride + point;
      const double x = (double)PetscRealPart(reference[0]);
      const double y = (double)PetscRealPart(reference[point_count]);
      const double z = (double)PetscRealPart(reference[2U * point_count]);
      gradient[3U * component] = map[0] * x + map[1] * y + map[2] * z;
      gradient[3U * component + 1U] = map[3] * x + map[4] * y + map[5] * z;
      gradient[3U * component + 2U] = map[6] * x + map[7] * y + map[8] * z;
    }
    strain[0] = gradient[0];
    strain[1] = gradient[4];
    strain[2] = gradient[8];
    strain[3] = gradient[1] + gradient[3];
    strain[4] = gradient[5] + gradient[7];
    strain[5] = gradient[2] + gradient[6];
    ss_mechanics_tangent_product(tangents + point * 36U, 6U, strain, transpose,
                                 product);
    const double physical[9] = {product[0], product[3], product[5],
                                product[3], product[1], product[4],
                                product[5], product[4], product[2]};
    for (size_t component = 0U; component < 3U; ++component) {
      PetscScalar *reference = fluxes + component * component_stride + point;
      const double *stress = physical + 3U * component;
      for (size_t axis = 0U; axis < 3U; ++axis) {
        reference[axis * point_count] =
            (PetscScalar)((map[axis] * stress[0] + map[3U + axis] * stress[1] +
                           map[6U + axis] * stress[2]) *
                          weight);
      }
    }
  }
}

/* Dense simplex tabulation has no tensor-product factorization.  Stack the
 * three reference-gradient directions into one tall matrix so the B and B^T
 * stages each use one GEMM, allowing the PETSc-selected BLAS to spend less
 * time entering small micro-kernels. */
void ss_operator_matrix_free_blas_cell_action_3d(
    const SsOperator *action_operator, const SsOperator *tangent_operator,
    size_t cell_index, const PetscScalar *reference_gradient_matrices,
    size_t node_count, const PetscScalar *gathered, int transpose) {
  const SsCellPlan *cell = &action_operator->plan->cells[cell_index];
  const SsOperatorCell *source_metadata = &tangent_operator->cells[cell_index];
  const size_t point_count = source_metadata->quadrature->point_count;
  const size_t closure_count = cell->closure_count;
  const size_t packed_offset = cell_index * action_operator->maximum_closure;
  const size_t *packed_slots = action_operator->packed_slots + packed_offset;
  double *action = action_operator->matrix_free_element_action + packed_offset;
  PetscScalar element_values[SS_FE_MAX_NODES * 3U];
  PetscScalar
      reference_derivatives[3U * SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE * 3U];
  PetscScalar reference_fluxes[3U * SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE * 3U];
  PetscScalar element_action[SS_FE_MAX_NODES * 3U];
  const PetscScalar one = 1.0;
  const PetscScalar zero = 0.0;
  PetscBLASInt q;
  PetscBLASInt stacked_q;
  PetscBLASInt p;
  const PetscBLASInt components = 3;
  size_t node;
  if (cell->dimension != 3U || closure_count != 3U * node_count ||
      node_count > SS_FE_MAX_NODES ||
      point_count > SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE ||
      reference_gradient_matrices == NULL ||
      source_metadata->prepared.inverse_transposes == NULL) {
    return;
  }
  q = (PetscBLASInt)point_count;
  stacked_q = 3 * q;
  p = (PetscBLASInt)node_count;
  for (node = 0U; node < node_count; ++node) {
    const size_t slot = 3U * node;
    const size_t x_slot = packed_slots[slot];
    const size_t y_slot = packed_slots[slot + 1U];
    const size_t z_slot = packed_slots[slot + 2U];
    element_values[node] =
        x_slot == SIZE_MAX ? 0.0 : (PetscScalar)PetscRealPart(gathered[x_slot]);
    element_values[node_count + node] =
        y_slot == SIZE_MAX ? 0.0 : (PetscScalar)PetscRealPart(gathered[y_slot]);
    element_values[2U * node_count + node] =
        z_slot == SIZE_MAX ? 0.0 : (PetscScalar)PetscRealPart(gathered[z_slot]);
  }
  BLASgemm_("N", "N", &stacked_q, &components, &p, &one, reference_gradient_matrices,
            &stacked_q, element_values, &p, &zero, reference_derivatives, &stacked_q);
  ss_operator_matrix_free_reference_flux_3d(
      source_metadata,
      tangent_operator->matrix_free_tangents +
          tangent_operator->matrix_free_tangent_offsets[cell_index],
      (size_t)stacked_q, reference_derivatives, reference_fluxes, transpose);
  BLASgemm_("T", "N", &p, &components, &stacked_q, &one, reference_gradient_matrices,
            &stacked_q, reference_fluxes, &stacked_q, &zero, element_action, &p);
  for (node = 0U; node < node_count; ++node) {
    const size_t slot = 3U * node;
    action[slot] = (double)PetscRealPart(element_action[node]);
    action[slot + 1U] = (double)PetscRealPart(element_action[node_count + node]);
    action[slot + 2U] = (double)PetscRealPart(element_action[2U * node_count + node]);
  }
}

/* Apply the shared dense simplex basis to several affine cells at once.  The
 * columns remain independent, so this is the same B, D, B^T calculation as
 * the single-cell path above; only the BLAS call shape changes from N-by-3 to
 * N-by-(3*batch). */
int ss_operator_matrix_free_blas_cell_batch_3d(const SsOperator *operator_handle,
                                               const size_t *cell_indices,
                                               size_t batch_count,
                                               const PetscScalar *gathered,
                                               int transpose, PetscScalar *workspace,
                                               size_t workspace_stride) {
  const SsOperator *tangent_operator = NULL;
  const SsOperatorCell *first_action_metadata = NULL;
  const SsOperatorCell *first_tangent_metadata = NULL;
  const PetscScalar *reference_gradient_matrices = NULL;
  const PetscScalar *component_reference_gradient_matrices = NULL;
  size_t node_count;
  size_t point_count;
  size_t stacked_points;
  size_t column_count;
  size_t element_entries;
  size_t quadrature_entries;
  size_t required_entries;
  PetscScalar *element_values;
  PetscScalar *reference_derivatives;
  PetscScalar *reference_fluxes;
  PetscScalar *element_action;
  const PetscScalar one = 1.0;
  const PetscScalar zero = 0.0;
  PetscBLASInt q;
  PetscBLASInt stacked_q;
  PetscBLASInt p;
  PetscBLASInt columns;
  size_t batch_slot;
  size_t first_cell;
  size_t projection_pattern = SIZE_MAX;
  int component_specific_projection = 0;
  if (operator_handle == NULL || cell_indices == NULL || gathered == NULL ||
      workspace == NULL || batch_count < 1U ||
      batch_count > SS_OPERATOR_MATRIX_FREE_MAX_ACTION_BATCH ||
      batch_count > operator_handle->plan->cell_count) {
    return 0;
  }
  first_cell = cell_indices[0];
  if (first_cell >= operator_handle->plan->cell_count) {
    return 0;
  }
  tangent_operator = operator_handle->matrix_free_projection_source == NULL
                         ? operator_handle
                         : operator_handle->matrix_free_projection_source;
  first_action_metadata = &operator_handle->cells[first_cell];
  first_tangent_metadata = &tangent_operator->cells[first_cell];
  reference_gradient_matrices =
      operator_handle->matrix_free_projection_source == NULL
          ? operator_handle->reference_gradient_blas_cache
          : operator_handle->matrix_free_projection_reference_gradient_blas_cache;
  node_count = first_action_metadata->field_descriptor->node_count;
  point_count = first_tangent_metadata->quadrature->point_count;
  if (node_count == 0U || node_count > SS_FE_MAX_NODES || point_count == 0U ||
      point_count > SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE ||
      reference_gradient_matrices == NULL) {
    return 0;
  }
  if (operator_handle->matrix_free_projection_source != NULL &&
      operator_handle->matrix_free_projection_source_constrained != NULL &&
      operator_handle->matrix_free_projection_source_constrained[first_cell] != 0U) {
    if (operator_handle->matrix_free_projection_constraint_pattern_indices == NULL ||
        operator_handle->matrix_free_projection_constraint_reference_blas_cache ==
            NULL) {
      return 0;
    }
    projection_pattern =
        operator_handle->matrix_free_projection_constraint_pattern_indices[first_cell];
    if (projection_pattern >=
        operator_handle->matrix_free_projection_constraint_pattern_count) {
      return 0;
    }
    component_specific_projection = 1;
  }
  stacked_points = 3U * point_count;
  column_count = 3U * batch_count;
  element_entries = node_count * column_count;
  quadrature_entries = stacked_points * column_count;
  required_entries = 2U * element_entries + 2U * quadrature_entries;
  if (required_entries > workspace_stride) {
    return 0;
  }
  element_values = workspace;
  reference_derivatives = element_values + element_entries;
  reference_fluxes = reference_derivatives + quadrature_entries;
  element_action = reference_fluxes + quadrature_entries;
  for (batch_slot = 0U; batch_slot < batch_count; ++batch_slot) {
    const size_t cell_index = cell_indices[batch_slot];
    if (cell_index >= operator_handle->plan->cell_count) {
      return 0;
    }
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    const SsOperatorCell *action_metadata = &operator_handle->cells[cell_index];
    const SsOperatorCell *tangent_metadata = &tangent_operator->cells[cell_index];
    const size_t packed_offset = cell_index * operator_handle->maximum_closure;
    const size_t *packed_slots = operator_handle->packed_slots + packed_offset;
    size_t node;
    if (cell->dimension != 3U || cell->closure_count != 3U * node_count ||
        action_metadata->field_descriptor != first_action_metadata->field_descriptor ||
        tangent_metadata->quadrature != first_tangent_metadata->quadrature ||
        tangent_metadata->prepared.inverse_transposes == NULL) {
      return 0;
    }
    if (operator_handle->matrix_free_projection_source != NULL) {
      const int source_has_constraints =
          operator_handle->matrix_free_projection_source_constrained == NULL
              ? 1
              : operator_handle
                        ->matrix_free_projection_source_constrained[cell_index] != 0U;
      if (component_specific_projection != 0) {
        if (source_has_constraints == 0 ||
            operator_handle->matrix_free_projection_constraint_pattern_indices ==
                NULL ||
            operator_handle
                    ->matrix_free_projection_constraint_pattern_indices[cell_index] !=
                projection_pattern) {
          return 0;
        }
      } else if (source_has_constraints != 0) {
        return 0;
      }
    }
    for (node = 0U; node < node_count; ++node) {
      const size_t slot = 3U * node;
      const size_t x_slot = packed_slots[slot];
      const size_t y_slot = packed_slots[slot + 1U];
      const size_t z_slot = packed_slots[slot + 2U];
      const size_t x_column =
          component_specific_projection != 0 ? batch_slot : 3U * batch_slot;
      const size_t y_column = component_specific_projection != 0
                                  ? batch_count + batch_slot
                                  : 3U * batch_slot + 1U;
      const size_t z_column = component_specific_projection != 0
                                  ? 2U * batch_count + batch_slot
                                  : 3U * batch_slot + 2U;
      element_values[x_column * node_count + node] =
          x_slot == SIZE_MAX ? 0.0 : (PetscScalar)PetscRealPart(gathered[x_slot]);
      element_values[y_column * node_count + node] =
          y_slot == SIZE_MAX ? 0.0 : (PetscScalar)PetscRealPart(gathered[y_slot]);
      element_values[z_column * node_count + node] =
          z_slot == SIZE_MAX ? 0.0 : (PetscScalar)PetscRealPart(gathered[z_slot]);
    }
  }
  q = (PetscBLASInt)point_count;
  stacked_q = 3 * q;
  p = (PetscBLASInt)node_count;
  columns = (PetscBLASInt)column_count;
  if (component_specific_projection == 0) {
    BLASgemm_("N", "N", &stacked_q, &columns, &p, &one, reference_gradient_matrices,
              &stacked_q, element_values, &p, &zero, reference_derivatives, &stacked_q);
  } else {
    const PetscBLASInt batch_columns = (PetscBLASInt)batch_count;
    size_t component;
    component_reference_gradient_matrices =
        operator_handle->matrix_free_projection_constraint_reference_blas_cache +
        projection_pattern * 3U * node_count * stacked_points;
    for (component = 0U; component < 3U; ++component) {
      BLASgemm_("N", "N", &stacked_q, &batch_columns, &p, &one,
                component_reference_gradient_matrices +
                    component * node_count * stacked_points,
                &stacked_q, element_values + component * batch_count * node_count, &p,
                &zero, reference_derivatives + component * batch_count * stacked_points,
                &stacked_q);
    }
  }
  for (batch_slot = 0U; batch_slot < batch_count; ++batch_slot) {
    const size_t cell_index = cell_indices[batch_slot];
    const size_t first_column =
        component_specific_projection != 0 ? batch_slot : 3U * batch_slot;
    const size_t component_stride = component_specific_projection != 0
                                        ? batch_count * stacked_points
                                        : stacked_points;
    ss_operator_matrix_free_reference_flux_3d(
        &tangent_operator->cells[cell_index],
        tangent_operator->matrix_free_tangents +
            tangent_operator->matrix_free_tangent_offsets[cell_index],
        component_stride, reference_derivatives + first_column * stacked_points,
        reference_fluxes + first_column * stacked_points, transpose);
  }
  if (component_specific_projection == 0) {
    BLASgemm_("T", "N", &p, &columns, &stacked_q, &one, reference_gradient_matrices,
              &stacked_q, reference_fluxes, &stacked_q, &zero, element_action, &p);
  } else {
    const PetscBLASInt batch_columns = (PetscBLASInt)batch_count;
    size_t component;
    for (component = 0U; component < 3U; ++component) {
      BLASgemm_("T", "N", &p, &batch_columns, &stacked_q, &one,
                component_reference_gradient_matrices +
                    component * node_count * stacked_points,
                &stacked_q, reference_fluxes + component * batch_count * stacked_points,
                &stacked_q, &zero,
                element_action + component * batch_count * node_count, &p);
    }
  }
  for (batch_slot = 0U; batch_slot < batch_count; ++batch_slot) {
    const size_t cell_index = cell_indices[batch_slot];
    const size_t packed_offset = cell_index * operator_handle->maximum_closure;
    double *action = operator_handle->matrix_free_element_action + packed_offset;
    const size_t x_column =
        component_specific_projection != 0 ? batch_slot : 3U * batch_slot;
    const size_t y_column = component_specific_projection != 0
                                ? batch_count + batch_slot
                                : 3U * batch_slot + 1U;
    const size_t z_column = component_specific_projection != 0
                                ? 2U * batch_count + batch_slot
                                : 3U * batch_slot + 2U;
    size_t node;
    for (node = 0U; node < node_count; ++node) {
      const size_t slot = 3U * node;
      action[slot] =
          (double)PetscRealPart(element_action[x_column * node_count + node]);
      action[slot + 1U] =
          (double)PetscRealPart(element_action[y_column * node_count + node]);
      action[slot + 2U] =
          (double)PetscRealPart(element_action[z_column * node_count + node]);
    }
  }
  return 1;
}
