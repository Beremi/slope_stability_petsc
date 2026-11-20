/* Element kernels: mechanics residual and tangent, scalar (seepage)
 * elements and natural boundary terms.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "assembly_impl.h"
#include "voigt.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void ss_element_error_clear(SsErrorRecord *error) {
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
}

static SsStatus ss_element_error_set(SsErrorRecord *error, SsStatus status,
                                     const char *code, SsEntityId entity,
                                     size_t quadrature) {
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
    error->valid = 1;
    error->status = status;
    error->key.component = 1U;
    error->key.entity = entity;
    error->key.quadrature = (uint64_t)quadrature;
    if (code != NULL) {
      (void)snprintf(error->code, sizeof(error->code), "%s", code);
      (void)snprintf(error->message, sizeof(error->message), "%s", code);
    }
  }
  return status;
}

static int ss_element_request_valid(uint32_t request) {
  const uint32_t supported =
      SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN | SS_ASSEMBLY_POTENTIAL;
  return request != 0U && (request & ~supported) == 0U;
}

static SsStatus ss_element_output_prepare(size_t closure_count, uint32_t request,
                                          SsElementOutput *output) {
  size_t matrix_count;
  if (output == NULL || !ss_element_request_valid(request) ||
      ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
       (output->residual == NULL || output->closure_capacity < closure_count)) ||
      (closure_count != 0U && closure_count > SIZE_MAX / closure_count)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  matrix_count = closure_count * closure_count;
  if ((request & SS_ASSEMBLY_JACOBIAN) != 0U &&
      (output->jacobian == NULL || output->matrix_capacity < matrix_count)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
    memset(output->residual, 0, closure_count * sizeof(*output->residual));
  }
  if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
    memset(output->jacobian, 0, matrix_count * sizeof(*output->jacobian));
  }
  output->potential = 0.0;
  return SS_OK;
}

static int ss_all_finite(const double *values, size_t count) {
  size_t index;
  for (index = 0U; index < count; ++index) {
    if (!isfinite(values[index])) {
      return 0;
    }
  }
  return 1;
}

SsStatus ss_assembly_mechanics_element_prepared(
    const SsMechanicsElementInput *input, const SsMechanicsPreparedElement *prepared,
    uint32_t request, SsElementOutput *output, SsErrorRecord *error) {
  const SsFeGeometryPolicy geometry_policy = {
      64.0 * DBL_EPSILON,
      1.0 / (64.0 * DBL_EPSILON),
  };
  size_t node_count;
  size_t closure_count;
  size_t quadrature_index;
  SsMcReduced reduced_value;
  const SsMcReduced *reduced = NULL;
  double constant_force[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  SsMcError material_error = {0};
  SsStatus status;
  ss_element_error_clear(error);
  if (input == NULL || input->field_descriptor == NULL ||
      input->coordinate_descriptor == NULL || input->quadrature == NULL ||
      input->coordinate_values == NULL || input->field_values == NULL ||
      input->material == NULL ||
      (input->point_force_mode != SS_MECHANICS_POINT_FORCE_FIXED &&
       input->point_force_mode != SS_MECHANICS_POINT_FORCE_LOAD) ||
      (input->dimension != 2U && input->dimension != 3U) ||
      input->field_descriptor->dimension != input->dimension ||
      input->coordinate_descriptor->dimension != input->dimension ||
      input->quadrature->domain != SS_FE_QUADRATURE_VOLUME ||
      input->quadrature->dimension != input->dimension ||
      input->quadrature->parent_dimension != input->dimension ||
      !isfinite(input->reduction_factor) || input->reduction_factor <= 0.0 ||
      !isfinite(input->external_load_factor) ||
      (prepared != NULL &&
       (prepared->basis == NULL ||
        (prepared->physical_gradients == NULL &&
         (prepared->reference_gradients == NULL ||
          prepared->inverse_transposes == NULL)) ||
        prepared->physical_points == NULL || prepared->weights == NULL ||
        prepared->reduced_material == NULL)) ||
      (input->body_force != NULL &&
       !ss_all_finite(input->body_force, input->dimension)) ||
      (input->coupling_body_force != NULL &&
       !ss_all_finite(input->coupling_body_force, input->dimension))) {
    return ss_element_error_set(error, SS_ERROR_INVALID_ARGUMENT,
                                "ASM_ELEMENT_MECHANICS_ARGUMENT",
                                input == NULL ? 0U : input->stable_cell_id, 0U);
  }
  node_count = input->field_descriptor->node_count;
  if (node_count > SIZE_MAX / input->dimension) {
    return ss_element_error_set(error, SS_ERROR_OVERFLOW,
                                "ASM_ELEMENT_CLOSURE_OVERFLOW", input->stable_cell_id,
                                0U);
  }
  closure_count = node_count * input->dimension;
  if (!ss_all_finite(input->field_values, closure_count)) {
    return ss_element_error_set(error, SS_ERROR_INVALID_ARGUMENT,
                                "ASM_ELEMENT_NONFINITE_FIELD", input->stable_cell_id,
                                0U);
  }
  status = ss_element_output_prepare(closure_count, request, output);
  if (status != SS_OK) {
    return ss_element_error_set(error, status, "ASM_ELEMENT_OUTPUT",
                                input->stable_cell_id, 0U);
  }
  if (prepared == NULL) {
    status = ss_mc_reduce(input->material, input->reduction_factor, &reduced_value,
                          &material_error);
    if (status != SS_OK) {
      return ss_element_error_set(
          error, status,
          material_error.code[0] == '\0' ? "ASM_ELEMENT_MATERIAL" : material_error.code,
          input->stable_cell_id, 0U);
    }
    reduced = &reduced_value;
  } else {
    reduced = prepared->reduced_material;
  }
  {
    uint32_t component;
    for (component = 0U; component < input->dimension; ++component) {
      constant_force[component] =
          (input->body_force == NULL
               ? 0.0
               : input->external_load_factor * input->body_force[component]) +
          (input->coupling_body_force == NULL ? 0.0
                                              : input->coupling_body_force[component]);
    }
  }
  for (quadrature_index = 0U; quadrature_index < input->quadrature->point_count;
       ++quadrature_index) {
    const double *reference =
        input->quadrature->points + quadrature_index * input->dimension;
    double basis_storage[SS_FE_MAX_NODES];
    double reference_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
    double physical_gradient_storage[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
    double tangent_times_b[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION * 6U];
    double strain[6] = {0.0};
    double displacement[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
    double force[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
    double point_coupling[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
    SsFeGeometry geometry = {0};
    SsMcResponse response;
    const double *basis = prepared == NULL
                              ? basis_storage
                              : prepared->basis + quadrature_index * node_count;
    const double *physical_gradients =
        prepared == NULL || prepared->physical_gradients == NULL
            ? physical_gradient_storage
            : prepared->physical_gradients +
                  quadrature_index * node_count * input->dimension;
    const double *physical = prepared == NULL ? geometry.physical
                                              : prepared->physical_points +
                                                    quadrature_index * input->dimension;
    double weight;
    size_t trial_node;
    if (prepared == NULL) {
      if (ss_fe_basis_eval_scalar(input->field_descriptor, reference, basis_storage,
                                  reference_gradients) != SS_OK ||
          ss_fe_geometry_eval(input->coordinate_descriptor, input->coordinate_values,
                              reference, &geometry_policy, &geometry) != SS_OK ||
          ss_fe_physical_gradients(input->field_descriptor, &geometry,
                                   reference_gradients,
                                   physical_gradient_storage) != SS_OK) {
        return ss_element_error_set(error, SS_ERROR_INTEGRITY, "ASM_ELEMENT_GEOMETRY",
                                    input->stable_cell_id, quadrature_index);
      }
      weight = input->quadrature->weights[quadrature_index] * geometry.determinant;
    } else {
      if (prepared->physical_gradients == NULL) {
        ss_fe_physical_gradients_apply(
            input->dimension, node_count,
            prepared->inverse_transposes +
                quadrature_index * prepared->inverse_transpose_stride,
            prepared->reference_gradients +
                quadrature_index * node_count * input->dimension,
            physical_gradient_storage);
      }
      weight = prepared->weights[quadrature_index];
    }
    if (!isfinite(weight) || weight <= 0.0) {
      return ss_element_error_set(error, SS_ERROR_INTEGRITY, "ASM_ELEMENT_WEIGHT",
                                  input->stable_cell_id, quadrature_index);
    }
    for (trial_node = 0U; trial_node < node_count; ++trial_node) {
      uint32_t component;
      const double *gradient = physical_gradients + trial_node * input->dimension;
      for (component = 0U; component < input->dimension; ++component) {
        const size_t trial_dof = trial_node * input->dimension + component;
        const double value = input->field_values[trial_dof];
        if ((request & SS_ASSEMBLY_POTENTIAL) != 0U) {
          displacement[component] += basis[trial_node] * value;
        }
        ss_mechanics_accumulate_strain(input->dimension, gradient, component, value,
                                       strain);
      }
    }
    status = ss_mc_evaluate_prepared(
        input->material, input->dimension == 2U ? SS_MC_DIMENSION_2 : SS_MC_DIMENSION_3,
        reduced, (request & SS_ASSEMBLY_JACOBIAN) != 0U ? SS_MC_EVALUATE_TANGENT : 0U,
        strain, &response, &material_error);
    if (status != SS_OK) {
      return ss_element_error_set(
          error, status,
          material_error.code[0] == '\0' ? "ASM_ELEMENT_MATERIAL" : material_error.code,
          input->stable_cell_id, quadrature_index);
    }
    if (input->coupling_point_kernel != NULL) {
      status = input->coupling_point_kernel(
          input->coupling_point_context, input->dimension, input->stable_cell_id,
          (uint32_t)quadrature_index, physical, point_coupling);
      if (status != SS_OK || !ss_all_finite(point_coupling, input->dimension)) {
        return ss_element_error_set(
            error, status == SS_OK ? SS_ERROR_INTEGRITY : status,
            status == SS_OK ? "ASM_COUPLING_NONFINITE" : "ASM_COUPLING_POINT",
            input->stable_cell_id, quadrature_index);
      }
    }
    {
      uint32_t component;
      for (component = 0U; component < input->dimension; ++component) {
        const double point_scale =
            input->point_force_mode == SS_MECHANICS_POINT_FORCE_LOAD
                ? input->external_load_factor
                : 1.0;
        force[component] =
            constant_force[component] + point_scale * point_coupling[component];
      }
    }
    if ((request & SS_ASSEMBLY_POTENTIAL) != 0U) {
      double external_density = 0.0;
      uint32_t component;
      for (component = 0U; component < input->dimension; ++component) {
        external_density += displacement[component] * force[component];
      }
      output->potential += (response.potential - external_density) * weight;
    }
    if ((request & (SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN)) != 0U) {
      if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
        size_t column_dof;
        for (column_dof = 0U; column_dof < closure_count; ++column_dof) {
          const double *column_gradient =
              physical_gradients + (column_dof / input->dimension) * input->dimension;
          const uint32_t column_component = (uint32_t)(column_dof % input->dimension);
          double *product = tangent_times_b + column_dof * 6U;
          ss_mechanics_tangent_times_b(input->dimension, response.tangent,
                                       column_gradient, column_component, product);
        }
      }
      size_t test_node;
      for (test_node = 0U; test_node < node_count; ++test_node) {
        const double *test_gradient = physical_gradients + test_node * input->dimension;
        uint32_t test_component;
        for (test_component = 0U; test_component < input->dimension; ++test_component) {
          size_t test_dof = test_node * input->dimension + test_component;
          if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
            const double internal = ss_mechanics_b_dot(
                input->dimension, test_gradient, test_component, response.stress, 1);
            output->residual[test_dof] +=
                (internal - basis[test_node] * force[test_component]) * weight;
          }
          if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
            size_t column_dof;
            for (column_dof = 0U; column_dof < closure_count; ++column_dof) {
              const double *product = tangent_times_b + column_dof * 6U;
              const double tangent_value = ss_mechanics_b_dot(
                  input->dimension, test_gradient, test_component, product, 1);
              output->jacobian[test_dof * closure_count + column_dof] +=
                  tangent_value * weight;
            }
          }
        }
      }
    }
  }
  if (!isfinite(output->potential) ||
      ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
       !ss_all_finite(output->residual, closure_count)) ||
      ((request & SS_ASSEMBLY_JACOBIAN) != 0U &&
       !ss_all_finite(output->jacobian, closure_count * closure_count))) {
    return ss_element_error_set(error, SS_ERROR_INTEGRITY,
                                "ASM_ELEMENT_NONFINITE_OUTPUT", input->stable_cell_id,
                                0U);
  }
  return SS_OK;
}

SsStatus ss_assembly_mechanics_element(const SsMechanicsElementInput *input,
                                       uint32_t request, SsElementOutput *output,
                                       SsErrorRecord *error) {
  return ss_assembly_mechanics_element_prepared(input, NULL, request, output, error);
}

SsStatus ss_assembly_scalar_element(const SsScalarElementInput *input, uint32_t request,
                                    SsElementOutput *output, SsErrorRecord *error) {
  const SsFeGeometryPolicy geometry_policy = {
      64.0 * DBL_EPSILON,
      1.0 / (64.0 * DBL_EPSILON),
  };
  size_t node_count;
  size_t quadrature_index;
  SsStatus status;
  ss_element_error_clear(error);
  if (input == NULL || input->field_descriptor == NULL ||
      input->coordinate_descriptor == NULL || input->quadrature == NULL ||
      input->coordinate_values == NULL || input->field_values == NULL ||
      input->point_kernel == NULL ||
      (input->dimension != 2U && input->dimension != 3U) ||
      input->field_descriptor->dimension != input->dimension ||
      input->coordinate_descriptor->dimension != input->dimension ||
      input->quadrature->domain != SS_FE_QUADRATURE_VOLUME ||
      input->quadrature->dimension != input->dimension ||
      input->quadrature->parent_dimension != input->dimension) {
    return ss_element_error_set(error, SS_ERROR_INVALID_ARGUMENT,
                                "ASM_ELEMENT_SCALAR_ARGUMENT",
                                input == NULL ? 0U : input->stable_cell_id, 0U);
  }
  node_count = input->field_descriptor->node_count;
  status = ss_element_output_prepare(node_count, request, output);
  if (status != SS_OK) {
    return ss_element_error_set(error, status, "ASM_ELEMENT_OUTPUT",
                                input->stable_cell_id, 0U);
  }
  for (quadrature_index = 0U; quadrature_index < input->quadrature->point_count;
       ++quadrature_index) {
    const double *reference =
        input->quadrature->points + quadrature_index * input->dimension;
    double basis[SS_FE_MAX_NODES] = {0.0};
    double reference_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
    double physical_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
    SsFeGeometry geometry = {0};
    SsScalarPointInput point_input = {0};
    SsScalarPointOutput point_output = {0};
    double weight;
    size_t node;
    if (ss_fe_basis_eval_scalar(input->field_descriptor, reference, basis,
                                reference_gradients) != SS_OK ||
        ss_fe_geometry_eval(input->coordinate_descriptor, input->coordinate_values,
                            reference, &geometry_policy, &geometry) != SS_OK ||
        ss_fe_physical_gradients(input->field_descriptor, &geometry,
                                 reference_gradients, physical_gradients) != SS_OK) {
      return ss_element_error_set(error, SS_ERROR_INTEGRITY, "ASM_ELEMENT_GEOMETRY",
                                  input->stable_cell_id, quadrature_index);
    }
    weight = input->quadrature->weights[quadrature_index] * geometry.determinant;
    point_input.dimension = input->dimension;
    point_input.stable_cell_id = input->stable_cell_id;
    point_input.quadrature_index = (uint32_t)quadrature_index;
    memcpy(point_input.physical, geometry.physical,
           input->dimension * sizeof(point_input.physical[0]));
    for (node = 0U; node < node_count; ++node) {
      uint32_t coordinate;
      const double field_value = input->field_values[node];
      if (!isfinite(field_value)) {
        return ss_element_error_set(error, SS_ERROR_INVALID_ARGUMENT,
                                    "ASM_ELEMENT_NONFINITE_FIELD",
                                    input->stable_cell_id, quadrature_index);
      }
      point_input.value += basis[node] * field_value;
      for (coordinate = 0U; coordinate < input->dimension; ++coordinate) {
        point_input.gradient[coordinate] +=
            physical_gradients[node * input->dimension + coordinate] * field_value;
      }
    }
    status = input->point_kernel(input->point_context, &point_input, &point_output);
    if (status != SS_OK || !isfinite(weight) || weight <= 0.0 ||
        !ss_all_finite(point_output.flux, input->dimension) ||
        !ss_all_finite(point_output.flux_derivative_value, input->dimension) ||
        !ss_all_finite(point_output.flux_derivative_gradient,
                       input->dimension * input->dimension) ||
        !isfinite(point_output.source) ||
        !isfinite(point_output.source_derivative_value) ||
        !isfinite(point_output.potential_density)) {
      return ss_element_error_set(error, status == SS_OK ? SS_ERROR_INTEGRITY : status,
                                  status == SS_OK ? "ASM_ELEMENT_SCALAR_NONFINITE"
                                                  : "ASM_ELEMENT_SCALAR_KERNEL",
                                  input->stable_cell_id, quadrature_index);
    }
    if ((request & SS_ASSEMBLY_POTENTIAL) != 0U) {
      output->potential += point_output.potential_density * weight;
    }
    for (node = 0U; node < node_count; ++node) {
      const double *test_gradient = physical_gradients + node * input->dimension;
      if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
        double residual = -basis[node] * point_output.source;
        uint32_t coordinate;
        for (coordinate = 0U; coordinate < input->dimension; ++coordinate) {
          residual += test_gradient[coordinate] * point_output.flux[coordinate];
        }
        output->residual[node] += residual * weight;
      }
      if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
        size_t column_node;
        for (column_node = 0U; column_node < node_count; ++column_node) {
          const double *column_gradient =
              physical_gradients + column_node * input->dimension;
          double tangent =
              -basis[node] * point_output.source_derivative_value * basis[column_node];
          uint32_t flux_coordinate;
          for (flux_coordinate = 0U; flux_coordinate < input->dimension;
               ++flux_coordinate) {
            double flux_derivative =
                point_output.flux_derivative_value[flux_coordinate] *
                basis[column_node];
            uint32_t gradient_coordinate;
            for (gradient_coordinate = 0U; gradient_coordinate < input->dimension;
                 ++gradient_coordinate) {
              flux_derivative +=
                  point_output
                      .flux_derivative_gradient[flux_coordinate * input->dimension +
                                                gradient_coordinate] *
                  column_gradient[gradient_coordinate];
            }
            tangent += test_gradient[flux_coordinate] * flux_derivative;
          }
          output->jacobian[node * node_count + column_node] += tangent * weight;
        }
      }
    }
  }
  if (!isfinite(output->potential) ||
      ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
       !ss_all_finite(output->residual, node_count)) ||
      ((request & SS_ASSEMBLY_JACOBIAN) != 0U &&
       !ss_all_finite(output->jacobian, node_count * node_count))) {
    return ss_element_error_set(error, SS_ERROR_INTEGRITY,
                                "ASM_ELEMENT_NONFINITE_OUTPUT", input->stable_cell_id,
                                0U);
  }
  return SS_OK;
}

SsStatus ss_assembly_natural_element(const SsNaturalElementInput *input,
                                     uint32_t request, SsElementOutput *output,
                                     double *signed_flux, SsErrorRecord *error) {
  const SsFeDescriptor *coordinate_descriptor = NULL;
  size_t closure_count;
  size_t quadrature_index;
  uint32_t dimension;
  uint32_t components;
  SsStatus status;
  ss_element_error_clear(error);
  if (input == NULL || input->facet == NULL || input->field_descriptor == NULL ||
      input->quadrature == NULL || input->field_values == NULL ||
      input->point_kernel == NULL || signed_flux == NULL) {
    return ss_element_error_set(error, SS_ERROR_INVALID_ARGUMENT,
                                "ASM_NATURAL_ARGUMENT", 0U, 0U);
  }
  dimension = input->facet->geometry.dimension;
  components = input->facet->components;
  closure_count = input->facet->closure_count;
  if ((dimension != 2U && dimension != 3U) || components == 0U ||
      components > dimension || input->field_descriptor->dimension != dimension ||
      input->quadrature->domain != SS_FE_QUADRATURE_FACET ||
      input->quadrature->parent_dimension != dimension ||
      ss_fe_descriptor_find(dimension, input->facet->geometry.coordinate_order,
                            &coordinate_descriptor) != SS_OK) {
    return ss_element_error_set(error, SS_ERROR_INVALID_ARGUMENT, "ASM_NATURAL_LAYOUT",
                                input->facet->geometry.stable_facet_id, 0U);
  }
  status = ss_element_output_prepare(closure_count, request, output);
  if (status != SS_OK) {
    return ss_element_error_set(error, status, "ASM_NATURAL_OUTPUT",
                                input->facet->geometry.stable_facet_id, 0U);
  }
  *signed_flux = 0.0;
  for (quadrature_index = 0U; quadrature_index < input->quadrature->point_count;
       ++quadrature_index) {
    const double *facet_reference =
        input->quadrature->points + quadrature_index * (size_t)(dimension - 1U);
    double parent_reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
    double parent_basis[SS_FE_MAX_NODES] = {0.0};
    double parent_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
    double facet_basis[SS_FIELD_MAX_FACET_NODES] = {0.0};
    SsFacetSample geometry = {0};
    SsNaturalPointInput point_input = {0};
    SsNaturalPointOutput point_output = {0};
    size_t parent_node;
    size_t facet_node = 0U;
    double weight;
    if (ss_fe_facet_parent_reference(dimension, input->facet->geometry.facet_index,
                                     facet_reference, parent_reference) != SS_OK ||
        ss_fe_basis_eval_scalar(input->field_descriptor, parent_reference, parent_basis,
                                parent_gradients) != SS_OK ||
        ss_mesh_facet_geometry_at(&input->facet->geometry, facet_reference,
                                  &geometry) != SS_OK) {
      return ss_element_error_set(error, SS_ERROR_INTEGRITY, "ASM_NATURAL_GEOMETRY",
                                  input->facet->geometry.stable_facet_id,
                                  quadrature_index);
    }
    for (parent_node = 0U; parent_node < input->field_descriptor->node_count;
         ++parent_node) {
      size_t component;
      size_t first_slot;
      int on_facet = 0;
      for (first_slot = 0U; first_slot < input->facet->closure_count;
           first_slot += components) {
        if (input->facet->parent_closure_slots[first_slot] ==
            parent_node * components) {
          on_facet = 1;
          break;
        }
      }
      if (on_facet == 0) {
        continue;
      }
      if (facet_node >= input->facet->field_node_count) {
        return ss_element_error_set(error, SS_ERROR_INTEGRITY, "ASM_NATURAL_NODE_MAP",
                                    input->facet->geometry.stable_facet_id,
                                    quadrature_index);
      }
      facet_basis[facet_node] = parent_basis[parent_node];
      for (component = 0U; component < components; ++component) {
        point_input.value[component] +=
            facet_basis[facet_node] *
            input->field_values[facet_node * components + component];
      }
      ++facet_node;
    }
    if (facet_node != input->facet->field_node_count) {
      return ss_element_error_set(error, SS_ERROR_INTEGRITY, "ASM_NATURAL_NODE_MAP",
                                  input->facet->geometry.stable_facet_id,
                                  quadrature_index);
    }
    point_input.dimension = dimension;
    point_input.components = components;
    point_input.stable_facet_id = input->facet->geometry.stable_facet_id;
    point_input.quadrature_index = (uint32_t)quadrature_index;
    memcpy(point_input.physical, geometry.physical,
           dimension * sizeof(point_input.physical[0]));
    memcpy(point_input.outward_normal, geometry.outward_normal,
           dimension * sizeof(point_input.outward_normal[0]));
    status = input->point_kernel(input->point_context, &point_input, &point_output);
    weight = input->quadrature->weights[quadrature_index] * geometry.measure;
    if (status != SS_OK || !isfinite(weight) || weight <= 0.0 ||
        !ss_all_finite(point_output.load, components) ||
        !ss_all_finite(point_output.load_derivative, (size_t)components * components) ||
        !isfinite(point_output.signed_flux)) {
      return ss_element_error_set(
          error, status == SS_OK ? SS_ERROR_INTEGRITY : status,
          status == SS_OK ? "ASM_NATURAL_NONFINITE" : "ASM_NATURAL_KERNEL",
          input->facet->geometry.stable_facet_id, quadrature_index);
    }
    *signed_flux += point_output.signed_flux * weight;
    if ((request & SS_ASSEMBLY_POTENTIAL) != 0U) {
      /* A load t(u) = t0 + K u with symmetric K derives from the potential
       * -(t0.u + u.K u / 2) = -(t.u - u.K u / 2), exactly for constant
       * tractions, pressures and Robin/spring terms.  A nonsymmetric
       * derivative admits no potential consistent with the residual. */
      double work = 0.0;
      uint32_t row;
      for (row = 0U; row < components; ++row) {
        uint32_t column;
        double stiffness_action = 0.0;
        for (column = 0U; column < components; ++column) {
          const double derivative =
              point_output.load_derivative[row * components + column];
          if (derivative != point_output.load_derivative[column * components + row]) {
            return ss_element_error_set(
                error, SS_ERROR_INVALID_STATE, "ASM_NATURAL_NONCONSERVATIVE",
                input->facet->geometry.stable_facet_id, quadrature_index);
          }
          stiffness_action += derivative * point_input.value[column];
        }
        work +=
            (point_output.load[row] - 0.5 * stiffness_action) * point_input.value[row];
      }
      output->potential -= work * weight;
    }
    for (facet_node = 0U; facet_node < input->facet->field_node_count; ++facet_node) {
      uint32_t component;
      for (component = 0U; component < components; ++component) {
        const size_t row = facet_node * components + component;
        if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
          output->residual[row] -=
              facet_basis[facet_node] * point_output.load[component] * weight;
        }
        if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
          size_t column_node;
          for (column_node = 0U; column_node < input->facet->field_node_count;
               ++column_node) {
            uint32_t column_component;
            for (column_component = 0U; column_component < components;
                 ++column_component) {
              const size_t column = column_node * components + column_component;
              output->jacobian[row * closure_count + column] -=
                  facet_basis[facet_node] *
                  point_output
                      .load_derivative[component * components + column_component] *
                  facet_basis[column_node] * weight;
            }
          }
        }
      }
    }
  }
  if (!isfinite(output->potential)) {
    return ss_element_error_set(error, SS_ERROR_INTEGRITY, "ASM_NATURAL_NONFINITE",
                                input->facet->geometry.stable_facet_id, 0U);
  }
  return SS_OK;
}
