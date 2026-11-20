/* Apply matrix-free tangents and their transposes.
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

#include <string.h>

int ss_operator_matrix_free_projected_physical_gradient(
    const SsOperator *fine_operator, const SsOperator *coarse_operator,
    size_t cell_index, size_t quadrature_index, size_t coarse_slot,
    double physical[SS_FE_MAX_DIMENSION]) {
  const SsCellPlan *fine_cell;
  const SsCellPlan *coarse_cell;
  const SsOperatorCell *fine_metadata;
  const SsFeDescriptor *coarse_descriptor;
  size_t dimension;
  size_t coarse_node;
  size_t component;
  double reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  size_t coordinate;
  if (fine_operator == NULL || coarse_operator == NULL || physical == NULL ||
      cell_index >= fine_operator->plan->cell_count ||
      cell_index >= coarse_operator->plan->cell_count) {
    return 0;
  }
  fine_cell = &fine_operator->plan->cells[cell_index];
  coarse_cell = &coarse_operator->plan->cells[cell_index];
  fine_metadata = &fine_operator->cells[cell_index];
  coarse_descriptor = coarse_operator->cells[cell_index].field_descriptor;
  dimension = fine_cell->dimension;
  if (dimension == 0U || dimension > SS_FE_MAX_DIMENSION ||
      coarse_cell->dimension != dimension ||
      quadrature_index >= fine_metadata->quadrature->point_count ||
      coarse_slot >= coarse_cell->closure_count || coarse_descriptor == NULL) {
    return 0;
  }
  if (coarse_operator->matrix_free_projection_gradient_count != 0U) {
    const size_t offset =
        coarse_operator->matrix_free_projection_gradient_offsets[cell_index] +
        quadrature_index * coarse_cell->closure_count * dimension +
        coarse_slot * dimension;
    if (coarse_operator->matrix_free_projection_gradients == NULL ||
        coarse_operator->matrix_free_projection_gradient_count < dimension ||
        offset > coarse_operator->matrix_free_projection_gradient_count - dimension) {
      return 0;
    }
    memcpy(physical, coarse_operator->matrix_free_projection_gradients + offset,
           dimension * sizeof(*physical));
    return 1;
  }
  if (coarse_operator->matrix_free_projection_physical_pattern_count != 0U &&
      coarse_operator->matrix_free_projection_physical_pattern_indices != NULL &&
      coarse_operator->matrix_free_projection_pattern_physical_gradients != NULL) {
    const size_t pattern =
        coarse_operator->matrix_free_projection_physical_pattern_indices[cell_index];
    const size_t entries_per_pattern =
        fine_metadata->quadrature->point_count * coarse_cell->closure_count * dimension;
    size_t offset;
    if (pattern >= coarse_operator->matrix_free_projection_physical_pattern_count) {
      return 0;
    }
    offset = pattern * entries_per_pattern +
             quadrature_index * coarse_cell->closure_count * dimension +
             coarse_slot * dimension;
    memcpy(physical,
           coarse_operator->matrix_free_projection_pattern_physical_gradients + offset,
           dimension * sizeof(*physical));
    return 1;
  }
  if (dimension != 3U || fine_metadata->affine_geometry == 0 ||
      coarse_operator->matrix_free_projection_constraint_pattern_indices == NULL ||
      coarse_operator->matrix_free_projection_constraint_reference_blas_cache == NULL ||
      coarse_operator->matrix_free_projection_constraint_pattern_count == 0U) {
    return 0;
  }
  coarse_node = coarse_slot / dimension;
  component = coarse_slot % dimension;
  {
    const size_t pattern =
        coarse_operator->matrix_free_projection_constraint_pattern_indices[cell_index];
    const size_t point_count = fine_metadata->quadrature->point_count;
    const size_t stacked_points = dimension * point_count;
    const PetscScalar *pattern_gradients;
    if (pattern >= coarse_operator->matrix_free_projection_constraint_pattern_count) {
      return 0;
    }
    pattern_gradients =
        coarse_operator->matrix_free_projection_constraint_reference_blas_cache +
        (pattern * dimension + component) * coarse_descriptor->node_count *
            stacked_points;
    for (coordinate = 0U; coordinate < dimension; ++coordinate) {
      reference[coordinate] = (double)PetscRealPart(
          pattern_gradients[coarse_node * stacked_points + coordinate * point_count +
                            quadrature_index]);
    }
  }
  for (coordinate = 0U; coordinate < dimension; ++coordinate) {
    physical[coordinate] =
        fine_metadata->inverse_transpose[coordinate * dimension] * reference[0] +
        fine_metadata->inverse_transpose[coordinate * dimension + 1U] * reference[1] +
        fine_metadata->inverse_transpose[coordinate * dimension + 2U] * reference[2];
  }
  return 1;
}

static void ss_operator_matrix_free_projected_cell_action(
    const SsOperator *operator_handle, size_t cell_index, const PetscScalar *gathered,
    int transpose) {
  const SsOperator *source = operator_handle->matrix_free_projection_source;
  const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
  const SsOperatorCell *source_metadata = &source->cells[cell_index];
  const size_t closure_count = cell->closure_count;
  const size_t node_count = closure_count / cell->dimension;
  const size_t packed_offset = cell_index * operator_handle->maximum_closure;
  const size_t *packed_slots = operator_handle->packed_slots + packed_offset;
  double *action = operator_handle->matrix_free_element_action + packed_offset;
  size_t quadrature_index;
  if (cell->dimension == 3U && source_metadata->affine_geometry != 0 &&
      operator_handle->matrix_free_projection_reference_gradient_blas_cache != NULL) {
    const int source_has_constraints =
        operator_handle->matrix_free_projection_source_constrained == NULL
            ? 1
            : operator_handle->matrix_free_projection_source_constrained[cell_index] !=
                  0U;
    if (source_has_constraints == 0) {
      if (source_metadata->quadrature->point_count <=
          SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE) {
        ss_operator_matrix_free_blas_cell_action_3d(
            operator_handle, source, cell_index,
            operator_handle->matrix_free_projection_reference_gradient_blas_cache,
            node_count, gathered, transpose);
      } else {
        ss_operator_matrix_free_reference_cell_action_3d(
            operator_handle, source, cell_index,
            operator_handle->matrix_free_projection_reference_gradients, node_count,
            gathered, transpose);
      }
      return;
    }
  }
  memset(action, 0, closure_count * sizeof(*action));
  if (cell->dimension == 3U) {
    if (closure_count != 3U * node_count) {
      return;
    }
    for (quadrature_index = 0U;
         quadrature_index < source_metadata->quadrature->point_count;
         ++quadrature_index) {
      double reconstructed_gradients[SS_FE_MAX_NODES * 3U * 3U];
      const double *gradients = reconstructed_gradients;
      const double *tangent = source->matrix_free_tangents +
                              source->matrix_free_tangent_offsets[cell_index] +
                              quadrature_index * 36U;
      const double weight = source_metadata->prepared.weights[quadrature_index];
      double strain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      double product[6];
      size_t node;
      if (operator_handle->matrix_free_projection_physical_pattern_count != 0U &&
          operator_handle->matrix_free_projection_physical_pattern_indices != NULL &&
          operator_handle->matrix_free_projection_pattern_physical_gradients != NULL) {
        const size_t pattern =
            operator_handle
                ->matrix_free_projection_physical_pattern_indices[cell_index];
        const size_t entries_per_pattern =
            source_metadata->quadrature->point_count * closure_count * 3U;
        if (pattern >= operator_handle->matrix_free_projection_physical_pattern_count) {
          return;
        }
        gradients = operator_handle->matrix_free_projection_pattern_physical_gradients +
                    pattern * entries_per_pattern +
                    quadrature_index * closure_count * 3U;
      } else {
        for (node = 0U; node < node_count; ++node) {
          size_t component;
          for (component = 0U; component < 3U; ++component) {
            if (ss_operator_matrix_free_projected_physical_gradient(
                    source, operator_handle, cell_index, quadrature_index,
                    3U * node + component,
                    reconstructed_gradients + (3U * node + component) * 3U) == 0) {
              return;
            }
          }
        }
      }
      for (node = 0U; node < node_count; ++node) {
        const size_t slot = 3U * node;
        const double *x_gradient = gradients + slot * 3U;
        const double *y_gradient = gradients + (slot + 1U) * 3U;
        const double *z_gradient = gradients + (slot + 2U) * 3U;
        const size_t x_slot = packed_slots[slot];
        const size_t y_slot = packed_slots[slot + 1U];
        const size_t z_slot = packed_slots[slot + 2U];
        const double x =
            x_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[x_slot]);
        const double y =
            y_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[y_slot]);
        const double z =
            z_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[z_slot]);
        strain[0] += x_gradient[0] * x;
        strain[3] += x_gradient[1] * x;
        strain[5] += x_gradient[2] * x;
        strain[1] += y_gradient[1] * y;
        strain[3] += y_gradient[0] * y;
        strain[4] += y_gradient[2] * y;
        strain[2] += z_gradient[2] * z;
        strain[4] += z_gradient[1] * z;
        strain[5] += z_gradient[0] * z;
      }
      if (transpose == 0) {
        product[0] = tangent[0] * strain[0] + tangent[6] * strain[1] +
                     tangent[12] * strain[2] + tangent[18] * strain[3] +
                     tangent[24] * strain[4] + tangent[30] * strain[5];
        product[1] = tangent[1] * strain[0] + tangent[7] * strain[1] +
                     tangent[13] * strain[2] + tangent[19] * strain[3] +
                     tangent[25] * strain[4] + tangent[31] * strain[5];
        product[2] = tangent[2] * strain[0] + tangent[8] * strain[1] +
                     tangent[14] * strain[2] + tangent[20] * strain[3] +
                     tangent[26] * strain[4] + tangent[32] * strain[5];
        product[3] = tangent[3] * strain[0] + tangent[9] * strain[1] +
                     tangent[15] * strain[2] + tangent[21] * strain[3] +
                     tangent[27] * strain[4] + tangent[33] * strain[5];
        product[4] = tangent[4] * strain[0] + tangent[10] * strain[1] +
                     tangent[16] * strain[2] + tangent[22] * strain[3] +
                     tangent[28] * strain[4] + tangent[34] * strain[5];
        product[5] = tangent[5] * strain[0] + tangent[11] * strain[1] +
                     tangent[17] * strain[2] + tangent[23] * strain[3] +
                     tangent[29] * strain[4] + tangent[35] * strain[5];
      } else {
        product[0] = tangent[0] * strain[0] + tangent[1] * strain[1] +
                     tangent[2] * strain[2] + tangent[3] * strain[3] +
                     tangent[4] * strain[4] + tangent[5] * strain[5];
        product[1] = tangent[6] * strain[0] + tangent[7] * strain[1] +
                     tangent[8] * strain[2] + tangent[9] * strain[3] +
                     tangent[10] * strain[4] + tangent[11] * strain[5];
        product[2] = tangent[12] * strain[0] + tangent[13] * strain[1] +
                     tangent[14] * strain[2] + tangent[15] * strain[3] +
                     tangent[16] * strain[4] + tangent[17] * strain[5];
        product[3] = tangent[18] * strain[0] + tangent[19] * strain[1] +
                     tangent[20] * strain[2] + tangent[21] * strain[3] +
                     tangent[22] * strain[4] + tangent[23] * strain[5];
        product[4] = tangent[24] * strain[0] + tangent[25] * strain[1] +
                     tangent[26] * strain[2] + tangent[27] * strain[3] +
                     tangent[28] * strain[4] + tangent[29] * strain[5];
        product[5] = tangent[30] * strain[0] + tangent[31] * strain[1] +
                     tangent[32] * strain[2] + tangent[33] * strain[3] +
                     tangent[34] * strain[4] + tangent[35] * strain[5];
      }
      for (node = 0U; node < node_count; ++node) {
        const size_t slot = 3U * node;
        const double *x_gradient = gradients + slot * 3U;
        const double *y_gradient = gradients + (slot + 1U) * 3U;
        const double *z_gradient = gradients + (slot + 2U) * 3U;
        action[slot] += (x_gradient[0] * product[0] + x_gradient[1] * product[3] +
                         x_gradient[2] * product[5]) *
                        weight;
        action[slot + 1U] += (y_gradient[1] * product[1] + y_gradient[0] * product[3] +
                              y_gradient[2] * product[4]) *
                             weight;
        action[slot + 2U] += (z_gradient[2] * product[2] + z_gradient[1] * product[4] +
                              z_gradient[0] * product[5]) *
                             weight;
      }
    }
    return;
  }
  if (cell->dimension == 2U) {
    if (closure_count != 2U * node_count) {
      return;
    }
    for (quadrature_index = 0U;
         quadrature_index < source_metadata->quadrature->point_count;
         ++quadrature_index) {
      const double *gradients =
          operator_handle->matrix_free_projection_gradients +
          operator_handle->matrix_free_projection_gradient_offsets[cell_index] +
          quadrature_index * closure_count * 2U;
      const double *tangent = source->matrix_free_tangents +
                              source->matrix_free_tangent_offsets[cell_index] +
                              quadrature_index * 9U;
      const double weight = source_metadata->prepared.weights[quadrature_index];
      double strain[3] = {0.0, 0.0, 0.0};
      double product[3];
      size_t node;
      for (node = 0U; node < node_count; ++node) {
        const size_t slot = 2U * node;
        const double *x_gradient = gradients + slot * 2U;
        const double *y_gradient = gradients + (slot + 1U) * 2U;
        const size_t x_slot = packed_slots[slot];
        const size_t y_slot = packed_slots[slot + 1U];
        const double x =
            x_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[x_slot]);
        const double y =
            y_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[y_slot]);
        strain[0] += x_gradient[0] * x;
        strain[2] += x_gradient[1] * x;
        strain[1] += y_gradient[1] * y;
        strain[2] += y_gradient[0] * y;
      }
      if (transpose == 0) {
        product[0] =
            tangent[0] * strain[0] + tangent[3] * strain[1] + tangent[6] * strain[2];
        product[1] =
            tangent[1] * strain[0] + tangent[4] * strain[1] + tangent[7] * strain[2];
        product[2] =
            tangent[2] * strain[0] + tangent[5] * strain[1] + tangent[8] * strain[2];
      } else {
        product[0] =
            tangent[0] * strain[0] + tangent[1] * strain[1] + tangent[2] * strain[2];
        product[1] =
            tangent[3] * strain[0] + tangent[4] * strain[1] + tangent[5] * strain[2];
        product[2] =
            tangent[6] * strain[0] + tangent[7] * strain[1] + tangent[8] * strain[2];
      }
      for (node = 0U; node < node_count; ++node) {
        const size_t slot = 2U * node;
        const double *x_gradient = gradients + slot * 2U;
        const double *y_gradient = gradients + (slot + 1U) * 2U;
        action[slot] +=
            (x_gradient[0] * product[0] + x_gradient[1] * product[2]) * weight;
        action[slot + 1U] +=
            (y_gradient[1] * product[1] + y_gradient[0] * product[2]) * weight;
      }
    }
  }
}

static void ss_operator_matrix_free_cell_action(const SsOperator *operator_handle,
                                                size_t cell_index,
                                                const PetscScalar *gathered,
                                                int transpose) {
  const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
  const SsOperatorCell *metadata = &operator_handle->cells[cell_index];
  const size_t closure_count = cell->closure_count;
  const size_t node_count = metadata->field_descriptor->node_count;
  const size_t packed_offset = cell_index * operator_handle->maximum_closure;
  const size_t *packed_slots = operator_handle->packed_slots + packed_offset;
  double *action = operator_handle->matrix_free_element_action + packed_offset;
  size_t quadrature_index;
  if (operator_handle->matrix_free_projection_source != NULL) {
    ss_operator_matrix_free_projected_cell_action(operator_handle, cell_index, gathered,
                                                  transpose);
    return;
  }
  if (cell->dimension == 3U && operator_handle->reference_gradient_blas_cache != NULL &&
      operator_handle->reference_gradient_count ==
          metadata->quadrature->point_count * node_count * 3U) {
    if (metadata->quadrature->point_count <=
        SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE) {
      ss_operator_matrix_free_blas_cell_action_3d(
          operator_handle, operator_handle, cell_index,
          operator_handle->reference_gradient_blas_cache, node_count, gathered,
          transpose);
    } else {
      ss_operator_matrix_free_reference_cell_action_3d(
          operator_handle, operator_handle, cell_index,
          operator_handle->reference_gradient_cache, node_count, gathered, transpose);
    }
    return;
  }
  memset(action, 0, closure_count * sizeof(*action));
  /*
   * Keep the dimension dispatch outside the quadrature and closure loops.  A
   * generic slot/dimension formulation makes the compiler emit integer div
   * and remainder instructions for every component at every quadrature
   * point.  P4 tetrahedra execute this path often enough for those divides to
   * dominate the otherwise compact B^T D B action.  The specialized loops
   * preserve the original component and floating-point accumulation order.
   */
  if (cell->dimension == 3U) {
    if (closure_count != 3U * node_count) {
      return;
    }
    for (quadrature_index = 0U; quadrature_index < metadata->quadrature->point_count;
         ++quadrature_index) {
      double gradients[SS_FE_MAX_NODES * 3U];
      const double *tangent = operator_handle->matrix_free_tangents +
                              operator_handle->matrix_free_tangent_offsets[cell_index] +
                              quadrature_index * 36U;
      const double weight = metadata->prepared.weights[quadrature_index];
      double strain[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      double product[6];
      size_t node;
      ss_operator_mechanics_point_gradients(operator_handle, metadata, quadrature_index,
                                            gradients);
      for (node = 0U; node < node_count; ++node) {
        const size_t slot = 3U * node;
        const double *gradient = gradients + slot;
        const size_t x_slot = packed_slots[slot];
        const size_t y_slot = packed_slots[slot + 1U];
        const size_t z_slot = packed_slots[slot + 2U];
        const double x =
            x_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[x_slot]);
        const double y =
            y_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[y_slot]);
        const double z =
            z_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[z_slot]);
        strain[0] += gradient[0] * x;
        strain[3] += gradient[1] * x;
        strain[5] += gradient[2] * x;
        strain[1] += gradient[1] * y;
        strain[3] += gradient[0] * y;
        strain[4] += gradient[2] * y;
        strain[2] += gradient[2] * z;
        strain[4] += gradient[1] * z;
        strain[5] += gradient[0] * z;
      }
      if (transpose == 0) {
        product[0] = tangent[0] * strain[0] + tangent[6] * strain[1] +
                     tangent[12] * strain[2] + tangent[18] * strain[3] +
                     tangent[24] * strain[4] + tangent[30] * strain[5];
        product[1] = tangent[1] * strain[0] + tangent[7] * strain[1] +
                     tangent[13] * strain[2] + tangent[19] * strain[3] +
                     tangent[25] * strain[4] + tangent[31] * strain[5];
        product[2] = tangent[2] * strain[0] + tangent[8] * strain[1] +
                     tangent[14] * strain[2] + tangent[20] * strain[3] +
                     tangent[26] * strain[4] + tangent[32] * strain[5];
        product[3] = tangent[3] * strain[0] + tangent[9] * strain[1] +
                     tangent[15] * strain[2] + tangent[21] * strain[3] +
                     tangent[27] * strain[4] + tangent[33] * strain[5];
        product[4] = tangent[4] * strain[0] + tangent[10] * strain[1] +
                     tangent[16] * strain[2] + tangent[22] * strain[3] +
                     tangent[28] * strain[4] + tangent[34] * strain[5];
        product[5] = tangent[5] * strain[0] + tangent[11] * strain[1] +
                     tangent[17] * strain[2] + tangent[23] * strain[3] +
                     tangent[29] * strain[4] + tangent[35] * strain[5];
      } else {
        product[0] = tangent[0] * strain[0] + tangent[1] * strain[1] +
                     tangent[2] * strain[2] + tangent[3] * strain[3] +
                     tangent[4] * strain[4] + tangent[5] * strain[5];
        product[1] = tangent[6] * strain[0] + tangent[7] * strain[1] +
                     tangent[8] * strain[2] + tangent[9] * strain[3] +
                     tangent[10] * strain[4] + tangent[11] * strain[5];
        product[2] = tangent[12] * strain[0] + tangent[13] * strain[1] +
                     tangent[14] * strain[2] + tangent[15] * strain[3] +
                     tangent[16] * strain[4] + tangent[17] * strain[5];
        product[3] = tangent[18] * strain[0] + tangent[19] * strain[1] +
                     tangent[20] * strain[2] + tangent[21] * strain[3] +
                     tangent[22] * strain[4] + tangent[23] * strain[5];
        product[4] = tangent[24] * strain[0] + tangent[25] * strain[1] +
                     tangent[26] * strain[2] + tangent[27] * strain[3] +
                     tangent[28] * strain[4] + tangent[29] * strain[5];
        product[5] = tangent[30] * strain[0] + tangent[31] * strain[1] +
                     tangent[32] * strain[2] + tangent[33] * strain[3] +
                     tangent[34] * strain[4] + tangent[35] * strain[5];
      }
      for (node = 0U; node < node_count; ++node) {
        const size_t slot = 3U * node;
        const double *gradient = gradients + slot;
        action[slot] += (gradient[0] * product[0] + gradient[1] * product[3] +
                         gradient[2] * product[5]) *
                        weight;
        action[slot + 1U] += (gradient[1] * product[1] + gradient[0] * product[3] +
                              gradient[2] * product[4]) *
                             weight;
        action[slot + 2U] += (gradient[2] * product[2] + gradient[1] * product[4] +
                              gradient[0] * product[5]) *
                             weight;
      }
    }
    return;
  }
  if (cell->dimension == 2U) {
    if (closure_count != 2U * node_count) {
      return;
    }
    for (quadrature_index = 0U; quadrature_index < metadata->quadrature->point_count;
         ++quadrature_index) {
      double gradients[SS_FE_MAX_NODES * 2U];
      const double *tangent = operator_handle->matrix_free_tangents +
                              operator_handle->matrix_free_tangent_offsets[cell_index] +
                              quadrature_index * 9U;
      const double weight = metadata->prepared.weights[quadrature_index];
      double strain[3] = {0.0, 0.0, 0.0};
      double product[3];
      size_t node;
      ss_operator_mechanics_point_gradients(operator_handle, metadata, quadrature_index,
                                            gradients);
      for (node = 0U; node < node_count; ++node) {
        const size_t slot = 2U * node;
        const double *gradient = gradients + slot;
        const size_t x_slot = packed_slots[slot];
        const size_t y_slot = packed_slots[slot + 1U];
        const double x =
            x_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[x_slot]);
        const double y =
            y_slot == SIZE_MAX ? 0.0 : (double)PetscRealPart(gathered[y_slot]);
        strain[0] += gradient[0] * x;
        strain[2] += gradient[1] * x;
        strain[1] += gradient[1] * y;
        strain[2] += gradient[0] * y;
      }
      if (transpose == 0) {
        product[0] =
            tangent[0] * strain[0] + tangent[3] * strain[1] + tangent[6] * strain[2];
        product[1] =
            tangent[1] * strain[0] + tangent[4] * strain[1] + tangent[7] * strain[2];
        product[2] =
            tangent[2] * strain[0] + tangent[5] * strain[1] + tangent[8] * strain[2];
      } else {
        product[0] =
            tangent[0] * strain[0] + tangent[1] * strain[1] + tangent[2] * strain[2];
        product[1] =
            tangent[3] * strain[0] + tangent[4] * strain[1] + tangent[5] * strain[2];
        product[2] =
            tangent[6] * strain[0] + tangent[7] * strain[1] + tangent[8] * strain[2];
      }
      for (node = 0U; node < node_count; ++node) {
        const size_t slot = 2U * node;
        const double *gradient = gradients + slot;
        action[slot] += (gradient[0] * product[0] + gradient[1] * product[2]) * weight;
        action[slot + 1U] +=
            (gradient[1] * product[1] + gradient[0] * product[2]) * weight;
      }
    }
    return;
  }
}

static int
ss_operator_matrix_free_cell_batch_eligible(const SsOperator *operator_handle,
                                            size_t cell_index) {
  const SsOperator *tangent_operator;
  const SsCellPlan *cell;
  const SsOperatorCell *metadata;
  if (operator_handle == NULL || cell_index >= operator_handle->plan->cell_count) {
    return 0;
  }
  tangent_operator = operator_handle->matrix_free_projection_source == NULL
                         ? operator_handle
                         : operator_handle->matrix_free_projection_source;
  cell = &operator_handle->plan->cells[cell_index];
  metadata = &tangent_operator->cells[cell_index];
  if (cell->dimension != 3U ||
      cell->closure_count !=
          3U * operator_handle->cells[cell_index].field_descriptor->node_count ||
      metadata->prepared.inverse_transposes == NULL ||
      metadata->quadrature->point_count > SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE ||
      (operator_handle->matrix_free_projection_source == NULL
           ? operator_handle->reference_gradient_blas_cache
           : operator_handle->matrix_free_projection_reference_gradient_blas_cache) ==
          NULL) {
    return 0;
  }
  if (operator_handle->matrix_free_projection_source == NULL) {
    return 1;
  }
  if (operator_handle->matrix_free_projection_source_constrained == NULL) {
    return 0;
  }
  if (operator_handle->matrix_free_projection_source_constrained[cell_index] == 0U) {
    return 1;
  }
  return operator_handle->matrix_free_projection_constraint_pattern_indices != NULL &&
                 operator_handle
                         ->matrix_free_projection_constraint_reference_blas_cache !=
                     NULL
             ? 1
             : 0;
}

static size_t
ss_operator_matrix_free_cell_batch_pattern(const SsOperator *operator_handle,
                                           size_t cell_index) {
  if (operator_handle == NULL ||
      operator_handle->matrix_free_projection_source == NULL) {
    return 0U;
  }
  if (operator_handle->matrix_free_projection_constraint_pattern_indices == NULL ||
      cell_index >= operator_handle->plan->cell_count) {
    return SIZE_MAX;
  }
  return operator_handle->matrix_free_projection_constraint_pattern_indices[cell_index];
}

static void ss_operator_matrix_free_flush_cell_batch(
    const SsOperator *operator_handle, const size_t *cell_indices, size_t batch_count,
    const PetscScalar *gathered, int transpose, PetscScalar *workspace,
    size_t workspace_stride) {
  size_t batch_slot;
  if (batch_count == 0U) {
    return;
  }
  if (ss_operator_matrix_free_blas_cell_batch_3d(operator_handle, cell_indices,
                                                 batch_count, gathered, transpose,
                                                 workspace, workspace_stride) != 0) {
    return;
  }
  for (batch_slot = 0U; batch_slot < batch_count; ++batch_slot) {
    ss_operator_matrix_free_cell_action(operator_handle, cell_indices[batch_slot],
                                        gathered, transpose);
  }
}

static void ss_operator_matrix_free_cell_action_range(
    const SsOperator *operator_handle, const size_t *cells, size_t first, size_t end,
    size_t batch_cells, const PetscScalar *gathered, int transpose,
    PetscScalar *workspace, size_t workspace_stride) {
  size_t batch_indices[SS_OPERATOR_MATRIX_FREE_MAX_ACTION_BATCH];
  size_t batch_count = 0U;
  size_t batch_pattern = SIZE_MAX;
  if (batch_cells <= 1U || workspace == NULL || workspace_stride == 0U) {
    if (cells != NULL) {
      for (size_t index = first; index < end; ++index) {
        ss_operator_matrix_free_cell_action(operator_handle, cells[index], gathered,
                                            transpose);
      }
    } else {
      for (size_t index = first; index < end; ++index) {
        if (operator_handle->cells[index].action_needed) {
          ss_operator_matrix_free_cell_action(operator_handle, index, gathered,
                                              transpose);
        }
      }
    }
    return;
  }
  for (size_t index = first; index < end; ++index) {
    const size_t cell_index = cells == NULL ? index : cells[index];
    if (cells == NULL && !operator_handle->cells[cell_index].action_needed) {
      continue;
    }
    if (ss_operator_matrix_free_cell_batch_eligible(operator_handle, cell_index) != 0) {
      const size_t pattern =
          ss_operator_matrix_free_cell_batch_pattern(operator_handle, cell_index);
      if (batch_count != 0U && pattern != batch_pattern) {
        ss_operator_matrix_free_flush_cell_batch(operator_handle, batch_indices,
                                                 batch_count, gathered, transpose,
                                                 workspace, workspace_stride);
        batch_count = 0U;
      }
      if (batch_count == 0U) {
        batch_pattern = pattern;
      }
      batch_indices[batch_count++] = cell_index;
      if (batch_count == batch_cells) {
        ss_operator_matrix_free_flush_cell_batch(operator_handle, batch_indices,
                                                 batch_count, gathered, transpose,
                                                 workspace, workspace_stride);
        batch_count = 0U;
        batch_pattern = SIZE_MAX;
      }
      continue;
    }
    if (batch_count != 0U) {
      ss_operator_matrix_free_flush_cell_batch(operator_handle, batch_indices,
                                               batch_count, gathered, transpose,
                                               workspace, workspace_stride);
      batch_count = 0U;
      batch_pattern = SIZE_MAX;
    }
    ss_operator_matrix_free_cell_action(operator_handle, cell_index, gathered,
                                        transpose);
  }
  ss_operator_matrix_free_flush_cell_batch(operator_handle, batch_indices, batch_count,
                                           gathered, transpose, workspace,
                                           workspace_stride);
}

static inline void ss_operator_matrix_free_sum_row(const SsOperator *operator_handle,
                                                   size_t local_row,
                                                   PetscScalar *output) {
  const SsOwnedRowPlan *row = &operator_handle->plan->rows[local_row];
  const double *element_action = operator_handle->matrix_free_element_action;
  const size_t stride = operator_handle->maximum_closure;
  double value = 0.0;
  for (size_t contribution = 0U; contribution < row->cell_count; ++contribution) {
    value += element_action[row->cell_slots[contribution].cell_index * stride +
                            row->cell_slots[contribution].row_slot];
  }
  output[local_row] = (PetscScalar)value;
}

static void ss_operator_matrix_free_apply_local(const SsOperator *operator_handle,
                                                const SsOperator *workspace_owner,
                                                size_t action_batch_cells,
                                                const PetscScalar *gathered,
                                                int transpose,
                                                PetscScalar *output_values) {
  const size_t cell_count = operator_handle->plan->cell_count;
  const size_t row_count = operator_handle->plan->row_count;
  const size_t workspace_stride = workspace_owner->matrix_free_action_workspace_stride;
  PetscScalar *const workspace = workspace_owner->matrix_free_action_workspace;
#if SLOPE_ENABLE_OPENMP
  if (ss_operator_use_openmp(operator_handle)) {
#pragma omp parallel num_threads(operator_handle->spec.requested_threads)
    {
      const size_t thread = (size_t)omp_get_thread_num();
      const size_t thread_count = (size_t)omp_get_num_threads();
      PetscScalar *thread_workspace =
          workspace_stride == 0U ||
                  thread >= (size_t)workspace_owner->spec.requested_threads
              ? NULL
              : workspace + thread * workspace_stride;
      ss_operator_matrix_free_cell_action_range(
          operator_handle, NULL, cell_count * thread / thread_count,
          cell_count * (thread + 1U) / thread_count, action_batch_cells, gathered,
          transpose, thread_workspace, workspace_stride);
      /* All cell contributions must be ready before the row owners read them.
       * The team join already waits for every row writer. */
#pragma omp barrier
#pragma omp for schedule(static) nowait
      for (size_t row = 0U; row < row_count; ++row) {
        ss_operator_matrix_free_sum_row(operator_handle, row, output_values);
      }
    }
    return;
  }
#endif
  ss_operator_matrix_free_cell_action_range(operator_handle, NULL, 0U, cell_count,
                                            action_batch_cells, gathered, transpose,
                                            workspace, workspace_stride);
  for (size_t row = 0U; row < row_count; ++row) {
    ss_operator_matrix_free_sum_row(operator_handle, row, output_values);
  }
}

static void ss_matrix_free_exchange_cells(const SsOperator *op,
                                          const SsOperator *workspace_owner,
                                          size_t first, size_t end,
                                          const PetscScalar *gathered, int transpose,
                                          int use_openmp) {
  const size_t stride = workspace_owner->matrix_free_action_workspace_stride;
  const size_t batch = workspace_owner->matrix_free_action_batch_cells;
  PetscScalar *const workspace = workspace_owner->matrix_free_action_workspace;
  if (first == end) {
    return;
  }
#if SLOPE_ENABLE_OPENMP
  if (use_openmp) {
#pragma omp parallel num_threads(op->spec.requested_threads)
    {
      const size_t thread = (size_t)omp_get_thread_num();
      const size_t threads = (size_t)omp_get_num_threads();
      PetscScalar *scratch =
          stride == 0U || thread >= (size_t)workspace_owner->spec.requested_threads
              ? NULL
              : workspace + thread * stride;
      ss_operator_matrix_free_cell_action_range(
          op, op->matrix_free_exchange->cells, first + (end - first) * thread / threads,
          first + (end - first) * (thread + 1U) / threads, batch, gathered, transpose,
          scratch, stride);
    }
    return;
  }
#else
  (void)use_openmp;
#endif
  ss_operator_matrix_free_cell_action_range(op, op->matrix_free_exchange->cells, first,
                                            end, batch, gathered, transpose, workspace,
                                            stride);
}

static void ss_matrix_free_exchange_rows(const SsOperator *op, size_t first, size_t end,
                                         PetscScalar *output, int use_openmp) {
  const size_t *rows = op->matrix_free_exchange->rows;
  if (first == end) {
    return;
  }
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel for num_threads(op->spec.requested_threads) if (use_openmp)       \
    schedule(static)
#else
  (void)use_openmp;
#endif
  for (size_t index = first; index < end; ++index) {
    ss_operator_matrix_free_sum_row(op, rows[index], output);
  }
}

static int ss_matrix_free_apply_exchange(SsOperator *op,
                                         const SsOperator *workspace_owner,
                                         const PetscScalar *input, PetscScalar *output,
                                         int transpose, int ready) {
  const SsMatrixFreeExchange *exchange = op->matrix_free_exchange;
  const size_t rows = op->plan->row_count;
  const int use_openmp = ss_operator_use_openmp(op);
  /* The coefficient pack is idle during MatMult. Its gathered slots let the
   * input SF write only ghost values while workers read the owned values. */
  PetscScalar *const gathered = op->packed_values;
  PetscScalar *const owned = gathered + exchange->owned_offset;
  double *const action = op->matrix_free_element_action;
  if (input == NULL) {
    memset(owned, 0, rows * sizeof(*owned));
    input = owned;
  }
  PetscErrorCode status =
      PetscSFBcastBegin(exchange->input, MPIU_SCALAR, input, gathered, MPI_REPLACE);
  if (status == PETSC_SUCCESS) {
    if (ready) {
      if (input != owned) {
        memcpy(owned, input, rows * sizeof(*owned));
      }
      ss_matrix_free_exchange_cells(op, workspace_owner, 0U, exchange->interior_cells,
                                    gathered, transpose, use_openmp);
    }
    status =
        PetscSFBcastEnd(exchange->input, MPIU_SCALAR, input, gathered, MPI_REPLACE);
  }
  ready = ready && status == PETSC_SUCCESS;
  if (ready) {
    ss_matrix_free_exchange_cells(op, workspace_owner, exchange->interior_cells,
                                  exchange->active_cells, gathered, transpose,
                                  use_openmp);
  } else {
    /* Even a failed rank sends the action. The caller flags its output so
     * the next Krylov norm propagates the failure to every rank. */
    memset(action, 0, op->plan->cell_count * op->maximum_closure * sizeof(*action));
  }
  /* Send owned-cell entries and receive into disjoint ghost-cell slots.
   * Each slot has one writer. The row owner performs the addition in plan order. */
  status =
      PetscSFReduceBegin(exchange->action, MPI_DOUBLE, action, action, MPI_REPLACE);
  if (ready) {
    ss_matrix_free_exchange_rows(op, 0U, exchange->local_rows, output, use_openmp);
  }
  if (status == PETSC_SUCCESS) {
    status =
        PetscSFReduceEnd(exchange->action, MPI_DOUBLE, action, action, MPI_REPLACE);
  }
  if (ready && status == PETSC_SUCCESS) {
    ss_matrix_free_exchange_rows(op, exchange->local_rows, rows, output, use_openmp);
    return 1;
  }
  return 0;
}

static PetscErrorCode ss_operator_matrix_free_apply(Mat matrix, Vec input, Vec output,
                                                    int transpose) {
  SsOperator *operator_handle = NULL;
  const SsOperator *workspace_owner = NULL;
  const PetscScalar *gathered = NULL;
  PetscScalar *output_values = NULL;
  Vec input_storage = input;
  int input_open = 0;
  int output_open = 0;
  int event_open = 0;
  PetscErrorCode ierr;
  ierr = MatShellGetContext(matrix, &operator_handle);
  if (ierr != PETSC_SUCCESS) {
    return ierr;
  }
  if (operator_handle == NULL ||
      ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK) {
    return PETSC_ERR_ARG_WRONGSTATE;
  }
  int failed =
      operator_handle->matrix_free_linearized == 0 ||
      (operator_handle->matrix_free_projection_source != NULL &&
       (operator_handle->matrix_free_projection_source->matrix_free_linearized == 0 ||
        operator_handle->matrix_free_projection_gradients == NULL ||
        operator_handle->matrix_free_projection_gradient_offsets == NULL));
  if (operator_handle->failure_point == SS_ASSEMBLY_FAIL_ACTION) {
    failed |= operator_handle->failure_rank < 0 ||
              operator_handle->failure_rank == operator_handle->plan->rank;
    operator_handle->failure_point = SS_ASSEMBLY_FAIL_NONE;
  }
  workspace_owner = operator_handle->matrix_free_projection_source == NULL
                        ? operator_handle
                        : operator_handle->matrix_free_projection_source;
  const size_t action_batch_cells = workspace_owner->matrix_free_action_batch_cells;
  failed |= action_batch_cells == 0U;
  const SsMatrixFreeExchange *exchange = operator_handle->matrix_free_exchange;
  /* Retain the original row-owned apply for distributed operators whose
   * exchange plans do not fit the workspace budget. A single rank already
   * has a dense, ordered coefficient array and can use it directly. */
  if (exchange == NULL && operator_handle->plan->size > 1) {
    input_storage = operator_handle->gathered_values;
    ierr = VecScatterBegin(operator_handle->scatter, input, input_storage,
                           INSERT_VALUES, SCATTER_FORWARD);
    if (ierr == PETSC_SUCCESS) {
      ierr = VecScatterEnd(operator_handle->scatter, input, input_storage,
                           INSERT_VALUES, SCATTER_FORWARD);
    }
    failed |= ierr != PETSC_SUCCESS;
  }
  if (exchange != NULL || !failed) {
    input_open = VecGetArrayRead(input_storage, &gathered) == PETSC_SUCCESS;
    failed |= !input_open;
  }
  const PetscLogEvent action_event =
      operator_handle->matrix_free_projection_source == NULL
          ? ss_operator_matrix_free_fine_event
          : ss_operator_matrix_free_projected_event;
  if (!failed) {
    output_open = VecGetArrayWrite(output, &output_values) == PETSC_SUCCESS;
    event_open =
        PetscLogEventBegin(action_event, matrix, input, output, 0) == PETSC_SUCCESS;
    failed = !output_open || !event_open;
  }
  if (exchange != NULL) {
    failed = !ss_matrix_free_apply_exchange(operator_handle, workspace_owner,
                                            input_open ? gathered : NULL, output_values,
                                            transpose, !failed);
  } else if (!failed) {
    ss_operator_matrix_free_apply_local(operator_handle, workspace_owner,
                                        action_batch_cells, gathered, transpose,
                                        output_values);
  }
  if (output_open) {
    failed |= VecRestoreArrayWrite(output, &output_values) != PETSC_SUCCESS;
  }
  if (input_open) {
    failed |= VecRestoreArrayRead(input_storage, &gathered) != PETSC_SUCCESS;
  }
  if (event_open) {
    failed |= PetscLogEventEnd(action_event, matrix, input, output, 0) != PETSC_SUCCESS;
  }
  if (failed) {
    operator_handle->matrix_free_action_failed = 1;
    /* The next Krylov dot or norm detects the poisoned local rows. Never
     * return a rank-local numerical error to PETSc's iteration driver. */
    (void)VecFlag(output, 1);
  }
  return PETSC_SUCCESS;
}

PetscErrorCode ss_operator_matrix_free_mult(Mat matrix, Vec input, Vec output) {
  return ss_operator_matrix_free_apply(matrix, input, output, 0);
}

PetscErrorCode ss_operator_matrix_free_mult_transpose(Mat matrix, Vec input,
                                                      Vec output) {
  return ss_operator_matrix_free_apply(matrix, input, output, 1);
}

int ss_operator_matrix_free_take_action_failure(Mat matrix) {
  PetscBool is_shell = PETSC_FALSE;
  void (*multiply)(void) = NULL;
  SsOperator *operator_handle = NULL;
  if (PetscObjectTypeCompare((PetscObject)matrix, MATSHELL, &is_shell) !=
          PETSC_SUCCESS ||
      !is_shell ||
      MatShellGetOperation(matrix, MATOP_MULT, &multiply) != PETSC_SUCCESS ||
      multiply != (void (*)(void))ss_operator_matrix_free_mult ||
      MatShellGetContext(matrix, &operator_handle) != PETSC_SUCCESS ||
      operator_handle == NULL) {
    return 0;
  }
  const int failed = operator_handle->matrix_free_action_failed;
  operator_handle->matrix_free_action_failed = 0;
  return failed;
}

PetscErrorCode ss_operator_matrix_free_get_diagonal(Mat matrix, Vec diagonal) {
  SsOperator *operator_handle = NULL;
  PetscScalar *values = NULL;
  size_t local_row;
  PetscErrorCode ierr = MatShellGetContext(matrix, &operator_handle);
  if (ierr != PETSC_SUCCESS) {
    return ierr;
  }
  if (operator_handle == NULL || operator_handle->matrix_free_linearized == 0) {
    return PETSC_ERR_ARG_WRONGSTATE;
  }
  if (VecGetArrayWrite(diagonal, &values) != PETSC_SUCCESS) {
    return PETSC_ERR_LIB;
  }
  for (local_row = 0U; local_row < operator_handle->plan->row_count; ++local_row) {
    const PetscInt slot = operator_handle->plan->rows[local_row].global_row -
                          operator_handle->plan->row_start;
    values[slot] = (PetscScalar)operator_handle->matrix_free_diagonal[local_row];
  }
  return VecRestoreArrayWrite(diagonal, &values);
}

PetscErrorCode ss_operator_matrix_free_create_vecs(Mat matrix, Vec *right, Vec *left) {
  SsOperator *operator_handle = NULL;
  PetscErrorCode ierr = MatShellGetContext(matrix, &operator_handle);
  if (ierr != PETSC_SUCCESS) {
    return ierr;
  }
  if (operator_handle == NULL || operator_handle->staging_residual == NULL) {
    return PETSC_ERR_ARG_WRONGSTATE;
  }
  if (right != NULL) {
    ierr = VecDuplicate(operator_handle->staging_residual, right);
    if (ierr != PETSC_SUCCESS) {
      return ierr;
    }
  }
  if (left != NULL) {
    ierr = VecDuplicate(operator_handle->staging_residual, left);
    if (ierr != PETSC_SUCCESS) {
      if (right != NULL) {
        (void)VecDestroy(right);
      }
      return ierr;
    }
  }
  return PETSC_SUCCESS;
}
