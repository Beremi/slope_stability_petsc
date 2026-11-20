/* Cell metadata, material lookup, natural facets, and input gather setup.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "operator_internal.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"
#include "slope_internal/mesh/cell_index.h"

#include "slope_internal/build/config.h"

#include <petscblaslapack.h>
#include <petscis.h>
#include <petscvec.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ss_operator_material_compare(const void *left, const void *right) {
  const SsMechanicsMaterialRegion *lhs = (const SsMechanicsMaterialRegion *)left;
  const SsMechanicsMaterialRegion *rhs = (const SsMechanicsMaterialRegion *)right;
  return lhs->material_value < rhs->material_value   ? -1
         : lhs->material_value > rhs->material_value ? 1
                                                     : 0;
}

static const SsMechanicsMaterialRegion *
ss_operator_material_find(const SsOperator *operator_handle, uint32_t material_value) {
  size_t left = 0U;
  size_t right;
  if (operator_handle == NULL || operator_handle->material_regions == NULL) {
    return NULL;
  }
  right = operator_handle->mechanics.region_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (operator_handle->material_regions[middle].material_value < material_value) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < operator_handle->mechanics.region_count &&
      operator_handle->material_regions[left].material_value == material_value) {
    return &operator_handle->material_regions[left];
  }
  return NULL;
}

static size_t ss_operator_find_gather_slot(const SsOperator *operator_handle,
                                           PetscInt global_index) {
  size_t left = 0U;
  size_t right = operator_handle->gather_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (operator_handle->gather_indices[middle] < global_index) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  return left < operator_handle->gather_count &&
                 operator_handle->gather_indices[left] == global_index
             ? left
             : SIZE_MAX;
}

static SsStatus ss_operator_build_p1_reference_cache(SsOperator *operator_handle);

SsStatus ss_operator_build_cell_metadata(SsOperator *operator_handle) {
  const SsFeGeometryPolicy geometry_policy = {
      64.0 * DBL_EPSILON,
      1.0 / (64.0 * DBL_EPSILON),
  };
  size_t basis_count = 0U;
  size_t inverse_count = 0U;
  size_t physical_count = 0U;
  size_t weight_count = 0U;
  size_t inverse_offset = 0U;
  size_t physical_offset = 0U;
  size_t weight_offset = 0U;
  double *reference_gradient_cache = NULL;
  uint64_t basis_blas_bytes = 0U;
  uint64_t reference_blas_bytes = 0U;
  size_t cell_index;
  if (operator_handle->plan->cell_count == 0U) {
    return SS_OK;
  }
  if (operator_handle->plan->cell_count > SIZE_MAX / sizeof(*operator_handle->cells)) {
    return SS_ERROR_OVERFLOW;
  }
  operator_handle->cells = (SsOperatorCell *)calloc(operator_handle->plan->cell_count,
                                                    sizeof(*operator_handle->cells));
  if (operator_handle->cells == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    SsOperatorCell *metadata = &operator_handle->cells[cell_index];
    SsFeQuadratureRequest request = {0};
    SsFeQuadraturePlan quadrature = {0};
    if (cell->dimension != operator_handle->properties.dimension ||
        cell->components != operator_handle->properties.components ||
        ss_fe_descriptor_find(cell->dimension, cell->order,
                              &metadata->field_descriptor) != SS_OK ||
        ss_fe_descriptor_find(cell->dimension, cell->coordinate_order,
                              &metadata->coordinate_descriptor) != SS_OK) {
      return SS_ERROR_INTEGRITY;
    }
    request.parent_dimension = cell->dimension;
    request.domain = SS_FE_QUADRATURE_VOLUME;
    request.operation = operator_handle->spec.physics == SS_ASSEMBLY_MECHANICS
                            ? SS_FE_OPERATION_MECHANICS_STIFFNESS
                            : SS_FE_OPERATION_HYDRAULIC_DIFFUSION;
    request.field_order = cell->order;
    request.coordinate_order = cell->coordinate_order;
    request.coefficient_order = 0U;
    request.mode = (int)operator_handle->spec.volume_quadrature_mode == 0
                       ? SS_FE_QUADRATURE_MODE_DEFAULT
                       : operator_handle->spec.volume_quadrature_mode;
    if (ss_fe_quadrature_plan(&request, &quadrature) != SS_OK ||
        quadrature.selected_rule == NULL) {
      return SS_ERROR_INTEGRITY;
    }
    metadata->quadrature = quadrature.selected_rule;
    if (operator_handle->spec.matrix_free != 0) {
      for (size_t slot = 0U; slot < cell->closure_count; ++slot) {
        const PetscInt row = cell->global_indices[slot];
        if (cell->constrained[slot] == 0U && row >= operator_handle->plan->row_start &&
            row < operator_handle->plan->row_end) {
          metadata->action_needed = 1;
          break;
        }
      }
    }
    if (cell_index == 0U) {
      (void)snprintf(operator_handle->diagnostics.volume_quadrature_id,
                     sizeof(operator_handle->diagnostics.volume_quadrature_id), "%s",
                     quadrature.selected_rule->id);
    }
    metadata->affine_geometry = metadata->coordinate_descriptor->order == 1U ? 1 : 0;
    if (operator_handle->spec.physics == SS_ASSEMBLY_MECHANICS) {
      const SsMechanicsMaterialRegion *region =
          ss_operator_material_find(operator_handle, cell->material_value);
      const size_t point_count = metadata->quadrature->point_count;
      const size_t node_count = metadata->field_descriptor->node_count;
      size_t region_index;
      if (region == NULL) {
        return SS_ERROR_NOT_FOUND;
      }
      region_index = (size_t)(region - operator_handle->material_regions);
      metadata->material = region->parameters;
      metadata->body_force = region->has_body_force != 0
                                 ? region->body_force
                                 : operator_handle->mechanics.body_force;
      metadata->material_region_index = region_index;
      if (point_count > SIZE_MAX / node_count ||
          point_count > SIZE_MAX / (cell->dimension * cell->dimension) ||
          inverse_count > SIZE_MAX - point_count * cell->dimension * cell->dimension ||
          physical_count > SIZE_MAX - point_count * cell->dimension ||
          weight_count > SIZE_MAX - point_count) {
        return SS_ERROR_OVERFLOW;
      }
      if (basis_count == 0U) {
        basis_count = point_count * node_count;
      } else if (basis_count != point_count * node_count ||
                 metadata->field_descriptor !=
                     operator_handle->cells[0].field_descriptor ||
                 metadata->quadrature != operator_handle->cells[0].quadrature) {
        /* A field space is homogeneous.  Sharing reference basis data across
         * cells is valid only while that invariant remains explicit. */
        return SS_ERROR_INTEGRITY;
      }
      /*
       * No cell materializes its physical Q-by-P-by-d gradient table (35 x 3
       * doubles per P4 tetrahedron point, 840 B).  Every kernel rebuilds the
       * gradients from the shared reference tabulation and J^{-T}: one map per
       * affine cell, or one 3 x 3 map per point of a curved cell (72 B).
       */
      if (metadata->affine_geometry == 0) {
        inverse_count += point_count * cell->dimension * cell->dimension;
      }
      physical_count += point_count * cell->dimension;
      weight_count += point_count;
    }
  }
  if (operator_handle->spec.physics != SS_ASSEMBLY_MECHANICS) {
    return SS_OK;
  }
  if (basis_count > SIZE_MAX / sizeof(*operator_handle->cell_basis_cache) ||
      inverse_count >
          SIZE_MAX / sizeof(*operator_handle->cell_inverse_transpose_cache) ||
      physical_count > SIZE_MAX / sizeof(*operator_handle->cell_physical_cache) ||
      weight_count > SIZE_MAX / sizeof(*operator_handle->cell_weight_cache) ||
      operator_handle->mechanics.region_count >
          SIZE_MAX / sizeof(*operator_handle->reduced_materials) ||
      basis_count > SIZE_MAX / operator_handle->properties.dimension ||
      basis_count * operator_handle->properties.dimension >
          SIZE_MAX / sizeof(*reference_gradient_cache)) {
    return SS_ERROR_OVERFLOW;
  }
  operator_handle->cell_basis_cache =
      (double *)malloc(basis_count * sizeof(*operator_handle->cell_basis_cache));
  operator_handle->basis_blas_cache =
      (PetscScalar *)malloc(basis_count * sizeof(*operator_handle->basis_blas_cache));
  operator_handle->cell_inverse_transpose_cache =
      (double *)malloc((inverse_count == 0U ? 1U : inverse_count) *
                       sizeof(*operator_handle->cell_inverse_transpose_cache));
  operator_handle->cell_physical_cache =
      (double *)malloc(physical_count * sizeof(*operator_handle->cell_physical_cache));
  operator_handle->cell_weight_cache =
      (double *)malloc(weight_count * sizeof(*operator_handle->cell_weight_cache));
  operator_handle->reduced_materials =
      (SsMcReduced *)calloc(operator_handle->mechanics.region_count,
                            sizeof(*operator_handle->reduced_materials));
  reference_gradient_cache =
      (double *)malloc(basis_count * operator_handle->properties.dimension *
                       sizeof(*reference_gradient_cache));
  operator_handle->reference_gradient_blas_cache =
      (PetscScalar *)malloc(basis_count * operator_handle->properties.dimension *
                            sizeof(*operator_handle->reference_gradient_blas_cache));
  if (operator_handle->cell_basis_cache == NULL ||
      operator_handle->basis_blas_cache == NULL ||
      operator_handle->cell_inverse_transpose_cache == NULL ||
      operator_handle->cell_physical_cache == NULL ||
      operator_handle->cell_weight_cache == NULL ||
      operator_handle->reduced_materials == NULL || reference_gradient_cache == NULL) {
    free(reference_gradient_cache);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (operator_handle->reference_gradient_blas_cache == NULL) {
    free(reference_gradient_cache);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if ((uint64_t)basis_count > UINT64_MAX / sizeof(double) ||
      (uint64_t)inverse_count > UINT64_MAX / sizeof(double) ||
      (uint64_t)physical_count > UINT64_MAX / sizeof(double) ||
      (uint64_t)weight_count > UINT64_MAX / sizeof(double) ||
      (uint64_t)operator_handle->mechanics.region_count >
          UINT64_MAX / sizeof(*operator_handle->reduced_materials)) {
    free(reference_gradient_cache);
    return SS_ERROR_OVERFLOW;
  }
  {
    const uint64_t reduced_bytes = (uint64_t)operator_handle->mechanics.region_count *
                                   sizeof(*operator_handle->reduced_materials);
    uint64_t double_count = (uint64_t)basis_count;
    if (double_count > UINT64_MAX - (uint64_t)inverse_count) {
      free(reference_gradient_cache);
      return SS_ERROR_OVERFLOW;
    }
    double_count += (uint64_t)inverse_count;
    if (double_count > UINT64_MAX - (uint64_t)physical_count) {
      free(reference_gradient_cache);
      return SS_ERROR_OVERFLOW;
    }
    double_count += (uint64_t)physical_count;
    if (double_count > UINT64_MAX - (uint64_t)weight_count) {
      free(reference_gradient_cache);
      return SS_ERROR_OVERFLOW;
    }
    double_count += (uint64_t)weight_count;
    if (double_count >
        UINT64_MAX - (uint64_t)basis_count * operator_handle->properties.dimension) {
      free(reference_gradient_cache);
      return SS_ERROR_OVERFLOW;
    }
    double_count += (uint64_t)basis_count * operator_handle->properties.dimension;
    if (basis_count * operator_handle->properties.dimension >
        UINT64_MAX / sizeof(*operator_handle->reference_gradient_blas_cache)) {
      free(reference_gradient_cache);
      return SS_ERROR_OVERFLOW;
    }
    reference_blas_bytes =
        (uint64_t)(basis_count * operator_handle->properties.dimension) *
        sizeof(*operator_handle->reference_gradient_blas_cache);
    if (basis_count > UINT64_MAX / sizeof(*operator_handle->basis_blas_cache)) {
      free(reference_gradient_cache);
      return SS_ERROR_OVERFLOW;
    }
    basis_blas_bytes =
        (uint64_t)basis_count * sizeof(*operator_handle->basis_blas_cache);
    if (double_count > UINT64_MAX / sizeof(double) ||
        double_count * sizeof(double) > UINT64_MAX - reduced_bytes ||
        double_count * sizeof(double) + reduced_bytes >
            UINT64_MAX - reference_blas_bytes ||
        double_count * sizeof(double) + reduced_bytes + reference_blas_bytes >
            UINT64_MAX - basis_blas_bytes) {
      free(reference_gradient_cache);
      return SS_ERROR_OVERFLOW;
    }
    operator_handle->cell_cache_bytes = double_count * sizeof(double) + reduced_bytes +
                                        reference_blas_bytes + basis_blas_bytes;
    operator_handle->diagnostics.geometry_cache_bytes =
        operator_handle->cell_cache_bytes;
    operator_handle->diagnostics.quadrature_points = (uint64_t)weight_count;
  }
  operator_handle->cached_reduction_factor = NAN;
  {
    const SsOperatorCell *reference_metadata = &operator_handle->cells[0];
    const size_t point_count = reference_metadata->quadrature->point_count;
    const size_t node_count = reference_metadata->field_descriptor->node_count;
    size_t quadrature_index;
    for (quadrature_index = 0U; quadrature_index < point_count; ++quadrature_index) {
      const double *reference =
          reference_metadata->quadrature->points +
          quadrature_index * operator_handle->properties.dimension;
      if (ss_fe_basis_eval_scalar(
              reference_metadata->field_descriptor, reference,
              operator_handle->cell_basis_cache + quadrature_index * node_count,
              reference_gradient_cache + quadrature_index * node_count *
                                             operator_handle->properties.dimension) !=
          SS_OK) {
        free(reference_gradient_cache);
        return SS_ERROR_INTEGRITY;
      }
    }
  }
  /* Assign table slices serially (a prefix over curved cells), then evaluate
   * the geometry in parallel with the static cell partition of the kernel
   * loops, so each worker first-touches the pages it later reads. */
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    SsOperatorCell *metadata = &operator_handle->cells[cell_index];
    const size_t point_count = metadata->quadrature->point_count;
    metadata->prepared.basis = operator_handle->cell_basis_cache;
    metadata->prepared.physical_gradients = NULL;
    metadata->prepared.reference_gradients = reference_gradient_cache;
    metadata->prepared.inverse_transposes =
        metadata->affine_geometry != 0
            ? metadata->inverse_transpose
            : operator_handle->cell_inverse_transpose_cache + inverse_offset;
    metadata->prepared.inverse_transpose_stride =
        metadata->affine_geometry != 0 ? 0U : (size_t)cell->dimension * cell->dimension;
    metadata->prepared.physical_points =
        operator_handle->cell_physical_cache + physical_offset;
    metadata->prepared.weights = operator_handle->cell_weight_cache + weight_offset;
    metadata->prepared.reduced_material =
        operator_handle->reduced_materials + metadata->material_region_index;
    if (metadata->affine_geometry == 0) {
      inverse_offset += point_count * cell->dimension * cell->dimension;
    }
    physical_offset += point_count * cell->dimension;
    weight_offset += point_count;
  }
  {
    int failed = 0;
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel for num_threads(                                                  \
        operator_handle->spec                                                          \
            .requested_threads) if (ss_operator_use_openmp(operator_handle) != 0)      \
    schedule(static) reduction(max : failed)
#endif
    for (cell_index = 0U; cell_index < operator_handle->plan->cell_count;
         ++cell_index) {
      const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
      SsOperatorCell *metadata = &operator_handle->cells[cell_index];
      const size_t point_count = metadata->quadrature->point_count;
      const size_t map_size = (size_t)cell->dimension * cell->dimension;
      double *point_maps = (double *)metadata->prepared.inverse_transposes;
      double *physical_points = (double *)metadata->prepared.physical_points;
      double *weights = (double *)metadata->prepared.weights;
      size_t quadrature_index;
      for (quadrature_index = 0U; quadrature_index < point_count; ++quadrature_index) {
        const double *reference =
            metadata->quadrature->points + quadrature_index * cell->dimension;
        SsFeGeometry geometry = {0};
        const double weight = metadata->quadrature->weights[quadrature_index];
        if (ss_fe_geometry_eval(metadata->coordinate_descriptor,
                                cell->coordinate_coordinates, reference,
                                &geometry_policy, &geometry) != SS_OK ||
            !isfinite(weight * geometry.determinant) ||
            weight * geometry.determinant <= 0.0) {
          failed = 1;
          break;
        }
        if (quadrature_index == 0U) {
          memcpy(metadata->inverse_transpose, geometry.inverse_transpose,
                 map_size * sizeof(*metadata->inverse_transpose));
        }
        if (metadata->affine_geometry == 0) {
          memcpy(point_maps + quadrature_index * map_size, geometry.inverse_transpose,
                 map_size * sizeof(double));
        }
        memcpy(physical_points + quadrature_index * cell->dimension, geometry.physical,
               cell->dimension * sizeof(double));
        weights[quadrature_index] = weight * geometry.determinant;
      }
    }
    if (failed != 0) {
      free(reference_gradient_cache);
      return SS_ERROR_INTEGRITY;
    }
  }
  if (inverse_offset != inverse_count || physical_offset != physical_count ||
      weight_offset != weight_count) {
    free(reference_gradient_cache);
    return SS_ERROR_INTEGRITY;
  }
  operator_handle->reference_gradient_cache = reference_gradient_cache;
  operator_handle->reference_gradient_count =
      basis_count * operator_handle->properties.dimension;
  {
    const size_t point_count = operator_handle->cells[0].quadrature->point_count;
    const size_t node_count = operator_handle->cells[0].field_descriptor->node_count;
    size_t coordinate;
    size_t node;
    for (node = 0U; node < node_count; ++node) {
      size_t point;
      for (point = 0U; point < point_count; ++point) {
        operator_handle->basis_blas_cache[node * point_count + point] =
            (PetscScalar)operator_handle->cell_basis_cache[point * node_count + node];
      }
    }
    for (node = 0U; node < node_count; ++node) {
      for (coordinate = 0U; coordinate < operator_handle->properties.dimension;
           ++coordinate) {
        size_t point;
        for (point = 0U; point < point_count; ++point) {
          operator_handle->reference_gradient_blas_cache
              [node * operator_handle->properties.dimension * point_count +
               coordinate * point_count + point] = (PetscScalar)
              reference_gradient_cache[(point * node_count + node) *
                                           operator_handle->properties.dimension +
                                       coordinate];
        }
      }
    }
  }
  return ss_operator_build_p1_reference_cache(operator_handle);
}

static SsStatus ss_operator_build_p1_reference_cache(SsOperator *operator_handle) {
  double transfer[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
  PetscScalar masked_transfer[SS_FE_MAX_NODES * SS_FE_MAX_NODES];
  const SsFeDescriptor *fine_descriptor;
  const SsFeDescriptor *p1_descriptor = NULL;
  size_t transfer_rows = 0U;
  size_t transfer_columns = 0U;
  size_t *pattern_indices = NULL;
  size_t *representatives = NULL;
  PetscScalar *reference_cache = NULL;
  size_t pattern_count = 0U;
  size_t cache_count;
  size_t cell_index;
  uint64_t index_bytes;
  uint64_t cache_bytes;
  if (operator_handle == NULL || operator_handle->spec.matrix_free == 0 ||
      operator_handle->spec.physics != SS_ASSEMBLY_MECHANICS ||
      operator_handle->properties.dimension != 3U ||
      operator_handle->plan->cell_count == 0U ||
      operator_handle->cells[0].field_descriptor == NULL ||
      operator_handle->cells[0].field_descriptor->order <= 1U) {
    return SS_OK;
  }
  fine_descriptor = operator_handle->cells[0].field_descriptor;
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    if (operator_handle->cells[cell_index].affine_geometry == 0 ||
        operator_handle->cells[cell_index].field_descriptor != fine_descriptor ||
        operator_handle->cells[cell_index].quadrature !=
            operator_handle->cells[0].quadrature) {
      return SS_OK;
    }
  }
  if (ss_fe_descriptor_find(3U, 1U, &p1_descriptor) != SS_OK || p1_descriptor == NULL ||
      ss_fe_transfer_build(p1_descriptor, fine_descriptor, transfer,
                           SS_FE_MAX_NODES * SS_FE_MAX_NODES, &transfer_rows,
                           &transfer_columns) != SS_OK ||
      transfer_rows != fine_descriptor->node_count ||
      transfer_columns != p1_descriptor->node_count ||
      operator_handle->plan->cell_count > SIZE_MAX / sizeof(*pattern_indices)) {
    return SS_ERROR_INTEGRITY;
  }
  pattern_indices =
      (size_t *)malloc(operator_handle->plan->cell_count * sizeof(*pattern_indices));
  representatives =
      (size_t *)malloc(operator_handle->plan->cell_count * sizeof(*representatives));
  if (pattern_indices == NULL || representatives == NULL) {
    free(pattern_indices);
    free(representatives);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    size_t pattern;
    for (pattern = 0U; pattern < pattern_count; ++pattern) {
      const SsCellPlan *representative =
          &operator_handle->plan->cells[representatives[pattern]];
      if (cell->closure_count == representative->closure_count &&
          memcmp(cell->constrained, representative->constrained,
                 cell->closure_count * sizeof(*cell->constrained)) == 0) {
        break;
      }
    }
    if (pattern == pattern_count) {
      if (pattern_count == SS_OPERATOR_MATRIX_FREE_MAX_PROJECTION_PHYSICAL_PATTERNS) {
        free(pattern_indices);
        free(representatives);
        return SS_OK;
      }
      representatives[pattern_count++] = cell_index;
    }
    pattern_indices[cell_index] = pattern;
  }
  {
    const size_t point_count = operator_handle->cells[0].quadrature->point_count;
    const size_t dimension = operator_handle->properties.dimension;
    const size_t stacked_points = dimension * point_count;
    if (pattern_count > SIZE_MAX / dimension ||
        pattern_count * dimension > SIZE_MAX / p1_descriptor->node_count ||
        pattern_count * dimension * p1_descriptor->node_count >
            SIZE_MAX / stacked_points) {
      free(pattern_indices);
      free(representatives);
      return SS_ERROR_OVERFLOW;
    }
    cache_count =
        pattern_count * dimension * p1_descriptor->node_count * stacked_points;
    reference_cache = (PetscScalar *)calloc(cache_count == 0U ? 1U : cache_count,
                                            sizeof(*reference_cache));
    if (reference_cache == NULL) {
      free(pattern_indices);
      free(representatives);
      return SS_ERROR_OUT_OF_MEMORY;
    }
    {
      const PetscBLASInt m = (PetscBLASInt)stacked_points;
      const PetscBLASInt n = (PetscBLASInt)p1_descriptor->node_count;
      const PetscBLASInt k = (PetscBLASInt)fine_descriptor->node_count;
      const PetscScalar one = 1.0;
      const PetscScalar zero = 0.0;
      size_t pattern;
      for (pattern = 0U; pattern < pattern_count; ++pattern) {
        const SsCellPlan *representative =
            &operator_handle->plan->cells[representatives[pattern]];
        size_t component;
        for (component = 0U; component < dimension; ++component) {
          size_t coarse_node;
          for (coarse_node = 0U; coarse_node < p1_descriptor->node_count;
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
          BLASgemm_("N", "N", &m, &n, &k, &one,
                    operator_handle->reference_gradient_blas_cache, &m, masked_transfer,
                    &k, &zero,
                    reference_cache + (pattern * dimension + component) *
                                          p1_descriptor->node_count * stacked_points,
                    &m);
        }
      }
    }
  }
  free(representatives);
  if ((uint64_t)operator_handle->plan->cell_count >
          UINT64_MAX / sizeof(*pattern_indices) ||
      (uint64_t)cache_count > UINT64_MAX / sizeof(*reference_cache)) {
    free(pattern_indices);
    free(reference_cache);
    return SS_ERROR_OVERFLOW;
  }
  index_bytes = (uint64_t)operator_handle->plan->cell_count * sizeof(*pattern_indices);
  cache_bytes = (uint64_t)cache_count * sizeof(*reference_cache);
  if (index_bytes > UINT64_MAX - cache_bytes ||
      operator_handle->cell_cache_bytes > UINT64_MAX - index_bytes - cache_bytes) {
    free(pattern_indices);
    free(reference_cache);
    return SS_ERROR_OVERFLOW;
  }
  operator_handle->matrix_free_p1_constraint_pattern_indices = pattern_indices;
  operator_handle->matrix_free_p1_constraint_pattern_count = pattern_count;
  operator_handle->matrix_free_p1_reference_gradient_blas_cache = reference_cache;
  operator_handle->cell_cache_bytes += index_bytes + cache_bytes;
  return SS_OK;
}

SsStatus ss_operator_prepare_reduced_materials(SsOperator *operator_handle,
                                               double reduction_factor) {
  size_t region;
  if (operator_handle->spec.physics != SS_ASSEMBLY_MECHANICS) {
    return SS_OK;
  }
  if (!isfinite(reduction_factor) || reduction_factor <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (operator_handle->plan->cell_count == 0U) {
    return SS_OK;
  }
  if (operator_handle->reduced_materials == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (operator_handle->reduced_materials_valid != 0 &&
      operator_handle->cached_reduction_factor == reduction_factor) {
    return SS_OK;
  }
  for (region = 0U; region < operator_handle->mechanics.region_count; ++region) {
    SsMcError material_error = {0};
    SsStatus status = ss_mc_reduce(
        operator_handle->material_regions[region].parameters, reduction_factor,
        &operator_handle->reduced_materials[region], &material_error);
    if (status != SS_OK) {
      operator_handle->reduced_materials_valid = 0;
      return status;
    }
  }
  operator_handle->cached_reduction_factor = reduction_factor;
  operator_handle->reduced_materials_valid = 1;
  return SS_OK;
}

static size_t ss_operator_find_cell(const SsOperator *operator_handle,
                                    SsEntityId stable_cell_id) {
  const size_t count = operator_handle->plan->cell_count;
  const size_t index = ss_mesh_cell_index_find(
      operator_handle->plan->cells, count, sizeof(*operator_handle->plan->cells),
      offsetof(SsCellPlan, stable_cell_id), stable_cell_id);
  return index < count ? index : SIZE_MAX;
}

SsStatus ss_operator_build_natural_facets(SsOperator *operator_handle) {
  size_t upper_bound = 0U;
  size_t operation;
  size_t packed = 0U;
  SsStatus status = SS_OK;
  for (operation = 0U; operation < operator_handle->spec.natural_operation_count;
       ++operation) {
    size_t count = 0U;
    status = ss_field_space_pack_incident_facets(
        operator_handle->plan->field_space,
        operator_handle->natural_operations[operation].support_value, NULL, 0U, &count);
    if (status == SS_OK && upper_bound > SIZE_MAX - count) {
      status = SS_ERROR_OVERFLOW;
    }
    status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                   status);
    if (status != SS_OK) {
      return status;
    }
    upper_bound += count;
  }
  if (upper_bound > SIZE_MAX / sizeof(*operator_handle->natural_facets)) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK && upper_bound != 0U) {
    operator_handle->natural_facets = (SsOperatorNaturalFacet *)calloc(
        upper_bound, sizeof(*operator_handle->natural_facets));
    if (operator_handle->natural_facets == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    return status;
  }
  for (operation = 0U; operation < operator_handle->spec.natural_operation_count;
       ++operation) {
    SsFieldFacetPack *facets = NULL;
    size_t count = 0U;
    size_t capacity = 0U;
    size_t facet;
    status = ss_field_space_pack_incident_facets(
        operator_handle->plan->field_space,
        operator_handle->natural_operations[operation].support_value, NULL, 0U, &count);
    if (status == SS_OK) {
      capacity = count;
      facets = (SsFieldFacetPack *)calloc(count == 0U ? 1U : count, sizeof(*facets));
      if (facets == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
    status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                   status);
    if (status == SS_OK) {
      status = ss_field_space_pack_incident_facets(
          operator_handle->plan->field_space,
          operator_handle->natural_operations[operation].support_value, facets,
          capacity, &count);
    }
    if (status == SS_OK && count > capacity) {
      status = SS_ERROR_INTEGRITY;
    }
    status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                   status);
    if (status != SS_OK) {
      free(facets);
      return status;
    }
    for (facet = 0U; facet < count; ++facet) {
      const SsFieldFacetPack *source = &facets[facet];
      SsOperatorNaturalFacet *destination;
      const SsFeQuadrature *quadrature = NULL;
      size_t slot;
      int owns_row = 0;
      size_t parent_cell;
      for (slot = 0U; slot < source->closure_count; ++slot) {
        if (source->constrained[slot] == 0U && source->owned[slot] != 0U) {
          owns_row = 1;
          break;
        }
      }
      if (owns_row == 0 && source->geometry.facet_owned == 0U) {
        continue;
      }
      parent_cell = ss_operator_find_cell(operator_handle,
                                          source->geometry.stable_parent_cell_id);
      if (packed >= upper_bound || parent_cell == SIZE_MAX ||
          source->order != operator_handle->plan->field_summary.order ||
          source->components != operator_handle->plan->field_summary.components ||
          ss_fe_descriptor_find(
              source->geometry.dimension, source->order,
              &operator_handle->cells[parent_cell].field_descriptor) != SS_OK ||
          ss_fe_facet_quadrature_default(source->geometry.dimension, source->order,
                                         &quadrature) != SS_OK ||
          quadrature == NULL) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      destination = &operator_handle->natural_facets[packed++];
      destination->pack = *source;
      destination->field_descriptor =
          operator_handle->cells[parent_cell].field_descriptor;
      destination->quadrature = quadrature;
      destination->parent_cell_index = parent_cell;
      destination->operation_index = operation;
    }
    free(facets);
    status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                   status);
    if (status != SS_OK) {
      return status;
    }
  }
  operator_handle->natural_facet_count = packed;
  return SS_OK;
}

SsStatus ss_operator_build_gather(SsOperator *operator_handle) {
  size_t upper_bound = 0U;
  size_t count = 0U;
  size_t cell_index;
  operator_handle->pack_required = (unsigned char *)calloc(
      operator_handle->plan->cell_count == 0U ? 1U : operator_handle->plan->cell_count,
      sizeof(*operator_handle->pack_required));
  if (operator_handle->pack_required == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE &&
      operator_handle->spec.matrix_free == 0) {
    size_t facet;
    for (cell_index = 0U; cell_index < operator_handle->plan->cell_count;
         ++cell_index) {
      operator_handle->pack_required[cell_index] =
          operator_handle->plan->cells[cell_index].cell_owned != 0U ? 1U : 0U;
    }
    /*
     * An owned natural facet is evaluated even when its parent cell is a
     * read-only overlap copy.  Its closure values are addressed through the
     * parent-cell pack, so retain precisely those additional parent packs.
     */
    for (facet = 0U; facet < operator_handle->natural_facet_count; ++facet) {
      if (operator_handle->natural_facets[facet].pack.geometry.facet_owned != 0U) {
        operator_handle
            ->pack_required[operator_handle->natural_facets[facet].parent_cell_index] =
            1U;
      }
    }
  } else {
    memset(operator_handle->pack_required, 1,
           operator_handle->plan->cell_count * sizeof(*operator_handle->pack_required));
  }
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    if (operator_handle->pack_required[cell_index] == 0U) {
      continue;
    }
    if (upper_bound >
        SIZE_MAX - operator_handle->plan->cells[cell_index].closure_count) {
      return SS_ERROR_OVERFLOW;
    }
    upper_bound += operator_handle->plan->cells[cell_index].closure_count;
  }
  if (upper_bound != 0U) {
    operator_handle->gather_indices =
        (PetscInt *)malloc(upper_bound * sizeof(*operator_handle->gather_indices));
    if (operator_handle->gather_indices == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    size_t slot;
    if (operator_handle->pack_required[cell_index] == 0U) {
      continue;
    }
    for (slot = 0U; slot < cell->closure_count; ++slot) {
      if (cell->constrained[slot] == 0U) {
        if (operator_handle->gather_indices == NULL || count >= upper_bound) {
          return SS_ERROR_INTEGRITY;
        }
        operator_handle->gather_indices[count++] = cell->global_indices[slot];
      }
    }
  }
  ss_sort(operator_handle->gather_indices, count,
          sizeof(*operator_handle->gather_indices), ss_petsc_int_compare);
  {
    size_t source;
    size_t unique = 0U;
    for (source = 0U; source < count; ++source) {
      if (operator_handle->gather_indices[source] < 0) {
        return SS_ERROR_INTEGRITY;
      }
      if (unique == 0U || operator_handle->gather_indices[source] !=
                              operator_handle->gather_indices[unique - 1U]) {
        operator_handle->gather_indices[unique++] =
            operator_handle->gather_indices[source];
      }
    }
    count = unique;
  }
  operator_handle->gather_count = count;
  if (count > (size_t)PETSC_MAX_INT) {
    return SS_ERROR_OVERFLOW;
  }
  if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK ||
      ISCreateGeneral(operator_handle->plan->communicator, (PetscInt)count,
                      operator_handle->gather_indices, PETSC_COPY_VALUES,
                      &operator_handle->gather_is) != PETSC_SUCCESS ||
      VecCreateSeq(PETSC_COMM_SELF, (PetscInt)count,
                   &operator_handle->gathered_values) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (ss_checked_mul_size(operator_handle->plan->cell_count,
                          operator_handle->maximum_closure, &count) != SS_OK ||
      count > SIZE_MAX / sizeof(*operator_handle->packed_slots)) {
    return SS_ERROR_OVERFLOW;
  }
  operator_handle->packed_slots = (size_t *)malloc(
      (count == 0U ? 1U : count) * sizeof(*operator_handle->packed_slots));
  operator_handle->packed_values = (double *)calloc(
      count == 0U ? 1U : count, sizeof(*operator_handle->packed_values));
  if (operator_handle->packed_slots == NULL || operator_handle->packed_values == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (cell_index = 0U; cell_index < count; ++cell_index) {
    operator_handle->packed_slots[cell_index] = SIZE_MAX;
  }
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    size_t slot;
    if (operator_handle->pack_required[cell_index] == 0U) {
      continue;
    }
    for (slot = 0U; slot < operator_handle->maximum_closure; ++slot) {
      size_t packed_slot = SIZE_MAX;
      if (slot < cell->closure_count && cell->constrained[slot] == 0U) {
        packed_slot =
            ss_operator_find_gather_slot(operator_handle, cell->global_indices[slot]);
        if (packed_slot == SIZE_MAX) {
          return SS_ERROR_INTEGRITY;
        }
      }
      operator_handle
          ->packed_slots[cell_index * operator_handle->maximum_closure + slot] =
          packed_slot;
    }
  }
  return SS_OK;
}
