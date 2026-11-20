/* Lagrange simplex bases, the quadrature registry, isoparametric geometry,
 * facet maps and order-to-order transfer operators.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/fe/fe.h"

#include "ss_fe_generated.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static double ss_fe_integer_power(double value, uint32_t exponent) {
  double result = 1.0;
  uint32_t index;
  for (index = 0U; index < exponent; ++index) {
    result *= value;
  }
  return result;
}

static double ss_fe_factorial(uint32_t value) {
  double result = 1.0;
  uint32_t index;
  for (index = 2U; index <= value; ++index) {
    result *= (double)index;
  }
  return result;
}

static double ss_fe_monomial_integral(uint32_t dimension, uint32_t a, uint32_t b,
                                      uint32_t c) {
  const uint32_t degree = a + (dimension >= 2U ? b : 0U) + (dimension == 3U ? c : 0U);
  double numerator = ss_fe_factorial(a);
  if (dimension >= 2U) {
    numerator *= ss_fe_factorial(b);
  }
  if (dimension == 3U) {
    numerator *= ss_fe_factorial(c);
  }
  return numerator / ss_fe_factorial(dimension + degree);
}

SsStatus ss_fe_descriptor_find(uint32_t dimension, uint32_t order,
                               const SsFeDescriptor **descriptor) {
  size_t index;
  if (descriptor == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *descriptor = NULL;
  for (index = 0U; index < ss_fe_generated_descriptor_count; ++index) {
    if (ss_fe_generated_descriptors[index].dimension == dimension &&
        ss_fe_generated_descriptors[index].order == order) {
      *descriptor = &ss_fe_generated_descriptors[index];
      return SS_OK;
    }
  }
  return SS_ERROR_NOT_FOUND;
}

SsStatus ss_fe_basis_eval_scalar(const SsFeDescriptor *descriptor,
                                 const double *reference_point, double *values,
                                 double *gradients) {
  if (descriptor == NULL || reference_point == NULL || values == NULL ||
      descriptor->basis_scalar_eval == NULL ||
      (descriptor->dimension != 2U && descriptor->dimension != 3U) ||
      descriptor->node_count > SS_FE_MAX_NODES) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  descriptor->basis_scalar_eval(reference_point, values, gradients);
  return SS_OK;
}

SsStatus ss_fe_basis_eval(const SsFeDescriptor *descriptor, size_t point_count,
                          const double *reference_points, double *values,
                          double *gradients) {
  if (descriptor == NULL || (point_count != 0U && reference_points == NULL) ||
      (point_count != 0U && values == NULL) || descriptor->basis_batch_eval == NULL ||
      (descriptor->dimension != 2U && descriptor->dimension != 3U) ||
      descriptor->node_count > SS_FE_MAX_NODES) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (point_count != 0U) {
    descriptor->basis_batch_eval(point_count, reference_points, values, gradients);
  }
  return SS_OK;
}

static int ss_fe_supported_order(uint32_t order) {
  return order == 1U || order == 2U || order == 4U || order == 8U;
}

static SsStatus ss_fe_quadrature_minimum_degree(const SsFeQuadratureRequest *request,
                                                uint32_t *minimum) {
  uint32_t measure_degree;
  uint32_t cofactor_degree;
  uint32_t field_degree;
  uint32_t gradient_pair_degree;
  uint32_t rational_degree;
  int volume_operation;
  if (request == NULL || minimum == NULL ||
      (request->parent_dimension != 2U && request->parent_dimension != 3U) ||
      !ss_fe_supported_order(request->field_order) ||
      !ss_fe_supported_order(request->coordinate_order) ||
      request->coefficient_order > 2U ||
      (request->domain != SS_FE_QUADRATURE_VOLUME &&
       request->domain != SS_FE_QUADRATURE_FACET)) {
    return request != NULL && request->coefficient_order > 2U
               ? SS_ERROR_NOT_FOUND
               : SS_ERROR_INVALID_ARGUMENT;
  }
  volume_operation = request->operation == SS_FE_OPERATION_GEOMETRY_QUALITY ||
                     request->operation == SS_FE_OPERATION_MECHANICS_STIFFNESS ||
                     request->operation == SS_FE_OPERATION_MECHANICS_BODY ||
                     request->operation == SS_FE_OPERATION_HYDRAULIC_DIFFUSION ||
                     request->operation == SS_FE_OPERATION_MASS ||
                     request->operation == SS_FE_OPERATION_MATERIAL_PROJECTION ||
                     request->operation == SS_FE_OPERATION_COUPLING_TRANSFER;
  if ((request->domain == SS_FE_QUADRATURE_VOLUME && volume_operation == 0) ||
      (request->domain == SS_FE_QUADRATURE_FACET && volume_operation != 0)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (request->domain == SS_FE_QUADRATURE_FACET &&
      request->operation != SS_FE_OPERATION_MECHANICS_NATURAL &&
      request->operation != SS_FE_OPERATION_HYDRAULIC_FLUX) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  /* Polynomial degree of the measure density: det J on volumes and the
   * cofactor-mapped normal |J_f| n = cof(J) n_ref on facets (pressure-type
   * loads).  The physical gradient times det J is cof(J)^T grad_ref N, whose
   * coordinate-dependent factor has degree (d - 1)(g - 1). */
  measure_degree = (request->parent_dimension -
                    (request->domain == SS_FE_QUADRATURE_FACET ? 1U : 0U)) *
                   (request->coordinate_order - 1U);
  cofactor_degree = (request->parent_dimension - 1U) * (request->coordinate_order - 1U);
  field_degree = request->field_order + request->coefficient_order;
  /* Curved stiffness and diffusion contain the rational metric
   * cof(J) cof(J)^T / det J; no finite degree makes them exact, so they keep
   * the affine degree or the measure degree, whichever is larger.  Every
   * polynomial term uses the degree of its complete product. */
  gradient_pair_degree = 2U * (request->field_order - 1U) + request->coefficient_order;
  rational_degree =
      gradient_pair_degree > measure_degree ? gradient_pair_degree : measure_degree;
  switch (request->operation) {
  case SS_FE_OPERATION_GEOMETRY_QUALITY:
    if (request->coordinate_order == 1U) {
      *minimum = 1U;
    } else if (request->coordinate_order == 2U) {
      *minimum = request->parent_dimension == 2U ? 5U : 4U;
    } else {
      *minimum = request->parent_dimension == 2U ? 10U : 9U;
    }
    if (*minimum < measure_degree) {
      *minimum = measure_degree;
    }
    break;
  case SS_FE_OPERATION_MECHANICS_STIFFNESS: {
    /* One rule serves internal force, tangent, body load N_a f det J and the
     * load work in the potential; the operator must integrate the load. */
    const uint32_t body_degree = field_degree + measure_degree;
    *minimum = rational_degree > body_degree ? rational_degree : body_degree;
    break;
  }
  case SS_FE_OPERATION_HYDRAULIC_DIFFUSION: {
    /* One rule serves diffusion and the nonlinear gravity transport
     * k grad(N_i) gamma_w kappa(p_h) e_up det J.  Inside the ramp transition
     * kappa = p_h / epsilon, so the residual term and its Jacobian
     * k gamma_w / epsilon d_up(N_i) N_j det J have degree 2p - 1 + c plus the
     * cofactor degree (seven for affine P4). */
    const uint32_t gravity_degree =
        2U * request->field_order - 1U + request->coefficient_order + cofactor_degree;
    *minimum = rational_degree > gravity_degree ? rational_degree : gravity_degree;
    break;
  }
  case SS_FE_OPERATION_MASS:
  case SS_FE_OPERATION_MATERIAL_PROJECTION:
    *minimum = 2U * request->field_order + request->coefficient_order + measure_degree;
    break;
  case SS_FE_OPERATION_MECHANICS_BODY:
  case SS_FE_OPERATION_MECHANICS_NATURAL:
  case SS_FE_OPERATION_HYDRAULIC_FLUX:
  case SS_FE_OPERATION_COUPLING_TRANSFER:
    *minimum = field_degree + measure_degree;
    break;
  default:
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*minimum == 0U) {
    *minimum = 1U;
  }
  return SS_OK;
}

static const SsFeQuadrature *
ss_fe_quadrature_select_minimum(const SsFeQuadratureRequest *request,
                                uint32_t minimum_degree,
                                uint32_t strictly_greater_than) {
  const SsFeQuadrature *selected = NULL;
  size_t index;
  for (index = 0U; index < ss_fe_generated_quadrature_count; ++index) {
    const SsFeQuadrature *candidate = &ss_fe_generated_quadratures[index];
    if (candidate->domain != request->domain ||
        candidate->parent_dimension != request->parent_dimension ||
        candidate->dimension !=
            request->parent_dimension -
                (request->domain == SS_FE_QUADRATURE_FACET ? 1U : 0U) ||
        candidate->exact_degree < minimum_degree ||
        candidate->exact_degree <= strictly_greater_than ||
        candidate->has_negative_weight != 0) {
      continue;
    }
    /* Cost first: among positive rules exact to the required degree the fewest
     * points win, then the lower degree, then the identifier.  A higher-degree
     * symmetric rule can be cheaper than a lower-degree Duffy product
     * (tetrahedron degree 14 has 204 points, degree 13 has 512). */
    if (selected == NULL || candidate->point_count < selected->point_count ||
        (candidate->point_count == selected->point_count &&
         (candidate->exact_degree < selected->exact_degree ||
          (candidate->exact_degree == selected->exact_degree &&
           strcmp(candidate->id, selected->id) < 0)))) {
      selected = candidate;
    }
  }
  return selected;
}

SsStatus ss_fe_quadrature_plan(const SsFeQuadratureRequest *request,
                               SsFeQuadraturePlan *plan) {
  SsStatus status;
  const SsFeQuadrature *default_rule = NULL;
  uint32_t minimum_degree = 0U;
  if (plan == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(plan, 0, sizeof(*plan));
  status = ss_fe_quadrature_minimum_degree(request, &minimum_degree);
  if (status != SS_OK) {
    return status;
  }
  if (request->mode != SS_FE_QUADRATURE_MODE_DEFAULT &&
      request->mode != SS_FE_QUADRATURE_MODE_PAIRED_HIGHER) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (request->override_rule_id != NULL) {
    status = ss_fe_quadrature_find(request->override_rule_id, &default_rule);
    if (status != SS_OK) {
      return status;
    }
    if (default_rule->domain != request->domain ||
        default_rule->parent_dimension != request->parent_dimension) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    if (default_rule->exact_degree < minimum_degree ||
        default_rule->has_negative_weight != 0) {
      return SS_ERROR_INTEGRITY;
    }
    plan->used_override = 1;
  } else {
    default_rule = ss_fe_quadrature_select_minimum(request, minimum_degree, 0U);
  }
  if (default_rule == NULL) {
    return SS_ERROR_NOT_FOUND;
  }
  /* The paired control is the cheapest rule of strictly higher degree.  Only
   * a paired request needs one; the highest registered degree has none. */
  plan->paired_rule = ss_fe_quadrature_select_minimum(request, minimum_degree,
                                                      default_rule->exact_degree);
  if (plan->paired_rule == NULL &&
      request->mode == SS_FE_QUADRATURE_MODE_PAIRED_HIGHER) {
    return SS_ERROR_NOT_FOUND;
  }
  plan->minimum_exact_degree = minimum_degree;
  plan->default_rule = default_rule;
  plan->selected_rule = request->mode == SS_FE_QUADRATURE_MODE_PAIRED_HIGHER
                            ? plan->paired_rule
                            : plan->default_rule;
  return SS_OK;
}

SsStatus ss_fe_quadrature_default(uint32_t dimension, uint32_t order,
                                  const SsFeQuadrature **quadrature) {
  size_t index;
  uint32_t exact_degree;
  if (quadrature == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *quadrature = NULL;
  if (order == 1U) {
    exact_degree = 1U;
  } else if (order == 2U) {
    exact_degree = dimension == 2U ? 5U : 4U;
  } else if (order == 4U) {
    exact_degree = 6U;
  } else if (order == 8U) {
    /* P8 stiffness on affine simplices has degree 2 * (8 - 1) = 14. */
    exact_degree = 14U;
  } else {
    return SS_ERROR_NOT_FOUND;
  }
  for (index = 0U; index < ss_fe_generated_quadrature_count; ++index) {
    if (ss_fe_generated_quadratures[index].domain == SS_FE_QUADRATURE_VOLUME &&
        ss_fe_generated_quadratures[index].parent_dimension == dimension &&
        ss_fe_generated_quadratures[index].dimension == dimension &&
        ss_fe_generated_quadratures[index].exact_degree == exact_degree) {
      *quadrature = &ss_fe_generated_quadratures[index];
      return SS_OK;
    }
  }
  return SS_ERROR_NOT_FOUND;
}

SsStatus ss_fe_facet_quadrature_default(uint32_t parent_dimension, uint32_t order,
                                        const SsFeQuadrature **quadrature) {
  size_t index;
  uint32_t exact_degree;
  if (quadrature == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *quadrature = NULL;
  if (parent_dimension != 2U && parent_dimension != 3U) {
    return SS_ERROR_NOT_FOUND;
  }
  if (order == 1U) {
    exact_degree = 1U;
  } else if (order == 2U) {
    exact_degree = 5U;
  } else if (order == 4U) {
    exact_degree = parent_dimension == 2U ? 9U : 6U;
  } else if (order == 8U) {
    exact_degree = 15U;
  } else {
    return SS_ERROR_NOT_FOUND;
  }
  for (index = 0U; index < ss_fe_generated_quadrature_count; ++index) {
    const SsFeQuadrature *candidate = &ss_fe_generated_quadratures[index];
    if (candidate->domain == SS_FE_QUADRATURE_FACET &&
        candidate->parent_dimension == parent_dimension &&
        candidate->dimension + 1U == parent_dimension &&
        candidate->exact_degree == exact_degree) {
      *quadrature = candidate;
      return SS_OK;
    }
  }
  return SS_ERROR_NOT_FOUND;
}

SsStatus ss_fe_quadrature_find(const char *id, const SsFeQuadrature **quadrature) {
  size_t index;
  if (id == NULL || quadrature == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *quadrature = NULL;
  for (index = 0U; index < ss_fe_generated_quadrature_count; ++index) {
    if (strcmp(ss_fe_generated_quadratures[index].id, id) == 0) {
      *quadrature = &ss_fe_generated_quadratures[index];
      return SS_OK;
    }
  }
  return SS_ERROR_NOT_FOUND;
}

size_t ss_fe_quadrature_registry_count(void) {
  return ss_fe_generated_quadrature_count;
}

SsStatus ss_fe_quadrature_registry_at(size_t index, const SsFeQuadrature **quadrature) {
  if (quadrature == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *quadrature = NULL;
  if (index >= ss_fe_generated_quadrature_count) {
    return SS_ERROR_NOT_FOUND;
  }
  *quadrature = &ss_fe_generated_quadratures[index];
  return SS_OK;
}

SsStatus ss_fe_quadrature_validate(const SsFeQuadrature *quadrature,
                                   size_t *bad_index) {
  const double tolerance = 4.0e-12;
  size_t point;
  uint32_t a;
  uint32_t b;
  uint32_t c;
  int observed_negative = 0;
  if (bad_index != NULL) {
    *bad_index = SIZE_MAX;
  }
  if (quadrature == NULL || bad_index == NULL || quadrature->points == NULL ||
      quadrature->weights == NULL ||
      (quadrature->dimension < 1U || quadrature->dimension > 3U) ||
      (quadrature->domain != SS_FE_QUADRATURE_VOLUME &&
       quadrature->domain != SS_FE_QUADRATURE_FACET) ||
      ((quadrature->domain == SS_FE_QUADRATURE_VOLUME &&
        quadrature->parent_dimension != quadrature->dimension) ||
       (quadrature->domain == SS_FE_QUADRATURE_FACET &&
        quadrature->parent_dimension != quadrature->dimension + 1U)) ||
      quadrature->point_count == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (point = 0U; point < quadrature->point_count; ++point) {
    double sum = 0.0;
    uint32_t coordinate;
    if (!isfinite(quadrature->weights[point])) {
      *bad_index = point;
      return SS_ERROR_INTEGRITY;
    }
    if (quadrature->weights[point] < 0.0) {
      observed_negative = 1;
    }
    for (coordinate = 0U; coordinate < quadrature->dimension; ++coordinate) {
      const double value =
          quadrature->points[point * (size_t)quadrature->dimension + coordinate];
      if (!isfinite(value) || value < -tolerance) {
        *bad_index = point;
        return SS_ERROR_INTEGRITY;
      }
      sum += value;
    }
    if (sum > 1.0 + tolerance) {
      *bad_index = point;
      return SS_ERROR_INTEGRITY;
    }
  }
  if (observed_negative != quadrature->has_negative_weight) {
    return SS_ERROR_INTEGRITY;
  }
  for (a = 0U; a <= quadrature->exact_degree; ++a) {
    const uint32_t b_limit =
        quadrature->dimension >= 2U ? quadrature->exact_degree - a : 0U;
    for (b = 0U; b <= b_limit; ++b) {
      const uint32_t c_limit =
          quadrature->dimension == 3U ? quadrature->exact_degree - a - b : 0U;
      for (c = 0U; c <= c_limit; ++c) {
        double actual = 0.0;
        for (point = 0U; point < quadrature->point_count; ++point) {
          const double *xi = quadrature->points + point * (size_t)quadrature->dimension;
          actual += quadrature->weights[point] * ss_fe_integer_power(xi[0], a) *
                    ss_fe_integer_power(quadrature->dimension >= 2U ? xi[1] : 1.0, b) *
                    ss_fe_integer_power(quadrature->dimension == 3U ? xi[2] : 1.0, c);
        }
        if (!isfinite(actual) ||
            fabs(actual - ss_fe_monomial_integral(quadrature->dimension, a, b, c)) >
                tolerance) {
          return SS_ERROR_INTEGRITY;
        }
      }
    }
  }
  return SS_OK;
}

static double ss_fe_determinant_inverse(uint32_t dimension, const double *matrix,
                                        double *inverse_transpose) {
  double determinant;
  if (dimension == 2U) {
    determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2];
    if (determinant != 0.0 && isfinite(determinant)) {
      inverse_transpose[0] = matrix[3] / determinant;
      inverse_transpose[1] = -matrix[2] / determinant;
      inverse_transpose[2] = -matrix[1] / determinant;
      inverse_transpose[3] = matrix[0] / determinant;
    }
  } else {
    const double cofactor00 = matrix[4] * matrix[8] - matrix[5] * matrix[7];
    const double cofactor01 = matrix[5] * matrix[6] - matrix[3] * matrix[8];
    const double cofactor02 = matrix[3] * matrix[7] - matrix[4] * matrix[6];
    const double cofactor10 = matrix[2] * matrix[7] - matrix[1] * matrix[8];
    const double cofactor11 = matrix[0] * matrix[8] - matrix[2] * matrix[6];
    const double cofactor12 = matrix[1] * matrix[6] - matrix[0] * matrix[7];
    const double cofactor20 = matrix[1] * matrix[5] - matrix[2] * matrix[4];
    const double cofactor21 = matrix[2] * matrix[3] - matrix[0] * matrix[5];
    const double cofactor22 = matrix[0] * matrix[4] - matrix[1] * matrix[3];
    determinant =
        matrix[0] * cofactor00 + matrix[1] * cofactor01 + matrix[2] * cofactor02;
    if (determinant != 0.0 && isfinite(determinant)) {
      inverse_transpose[0] = cofactor00 / determinant;
      inverse_transpose[1] = cofactor01 / determinant;
      inverse_transpose[2] = cofactor02 / determinant;
      inverse_transpose[3] = cofactor10 / determinant;
      inverse_transpose[4] = cofactor11 / determinant;
      inverse_transpose[5] = cofactor12 / determinant;
      inverse_transpose[6] = cofactor20 / determinant;
      inverse_transpose[7] = cofactor21 / determinant;
      inverse_transpose[8] = cofactor22 / determinant;
    }
  }
  return determinant;
}

SsStatus ss_fe_geometry_eval(const SsFeDescriptor *descriptor,
                             const double *coordinates, const double *reference_point,
                             const SsFeGeometryPolicy *policy, SsFeGeometry *geometry) {
  double values[SS_FE_MAX_NODES];
  double gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  double matrix_norm_squared = 0.0;
  double inverse_norm_squared = 0.0;
  size_t node;
  uint32_t physical_coordinate;
  uint32_t reference_coordinate;
  if (descriptor == NULL || coordinates == NULL || reference_point == NULL ||
      policy == NULL || geometry == NULL ||
      policy->absolute_determinant_tolerance < 0.0 ||
      policy->maximum_condition <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(geometry, 0, sizeof(*geometry));
  geometry->dimension = descriptor->dimension;
  if (ss_fe_basis_eval(descriptor, 1U, reference_point, values, gradients) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  for (node = 0U; node < descriptor->node_count; ++node) {
    for (physical_coordinate = 0U; physical_coordinate < descriptor->dimension;
         ++physical_coordinate) {
      const double coordinate =
          coordinates[node * (size_t)descriptor->dimension + physical_coordinate];
      if (!isfinite(coordinate)) {
        return SS_ERROR_INTEGRITY;
      }
      geometry->physical[physical_coordinate] += values[node] * coordinate;
      for (reference_coordinate = 0U; reference_coordinate < descriptor->dimension;
           ++reference_coordinate) {
        geometry->jacobian[physical_coordinate * (size_t)descriptor->dimension +
                           reference_coordinate] +=
            coordinate *
            gradients[node * (size_t)descriptor->dimension + reference_coordinate];
      }
    }
  }
  geometry->determinant = ss_fe_determinant_inverse(
      descriptor->dimension, geometry->jacobian, geometry->inverse_transpose);
  if (!isfinite(geometry->determinant)) {
    return SS_ERROR_INTEGRITY;
  }
  for (physical_coordinate = 0U; physical_coordinate < descriptor->dimension;
       ++physical_coordinate) {
    for (reference_coordinate = 0U; reference_coordinate < descriptor->dimension;
         ++reference_coordinate) {
      const size_t offset =
          physical_coordinate * (size_t)descriptor->dimension + reference_coordinate;
      matrix_norm_squared += geometry->jacobian[offset] * geometry->jacobian[offset];
      inverse_norm_squared +=
          geometry->inverse_transpose[offset] * geometry->inverse_transpose[offset];
    }
  }
  geometry->condition_estimate = sqrt(matrix_norm_squared * inverse_norm_squared);
  if (geometry->determinant <= policy->absolute_determinant_tolerance ||
      !isfinite(geometry->condition_estimate) ||
      geometry->condition_estimate > policy->maximum_condition) {
    return SS_ERROR_INTEGRITY;
  }
  return SS_OK;
}

SsStatus ss_fe_geometry_audit(const SsFeDescriptor *descriptor,
                              const double *coordinates, size_t point_count,
                              const double *reference_points,
                              const SsFeGeometryPolicy *policy, size_t *bad_index) {
  size_t point;
  if (bad_index == NULL || descriptor == NULL ||
      (point_count != 0U && reference_points == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *bad_index = SIZE_MAX;
  for (point = 0U; point < point_count; ++point) {
    SsFeGeometry geometry;
    const SsStatus status = ss_fe_geometry_eval(
        descriptor, coordinates,
        reference_points + point * (size_t)descriptor->dimension, policy, &geometry);
    if (status != SS_OK) {
      *bad_index = point;
      return status;
    }
  }
  return SS_OK;
}

void ss_fe_physical_gradients_apply(uint32_t dimension, size_t node_count,
                                    const double *inverse_transpose,
                                    const double *reference_gradients,
                                    double *physical_gradients) {
  size_t node;
  uint32_t physical_coordinate;
  for (node = 0U; node < node_count; ++node) {
    for (physical_coordinate = 0U; physical_coordinate < dimension;
         ++physical_coordinate) {
      uint32_t reference_coordinate;
      double value = 0.0;
      for (reference_coordinate = 0U; reference_coordinate < dimension;
           ++reference_coordinate) {
        value += inverse_transpose[physical_coordinate * (size_t)dimension +
                                   reference_coordinate] *
                 reference_gradients[node * (size_t)dimension + reference_coordinate];
      }
      physical_gradients[node * (size_t)dimension + physical_coordinate] = value;
    }
  }
}

SsStatus ss_fe_physical_gradients(const SsFeDescriptor *descriptor,
                                  const SsFeGeometry *geometry,
                                  const double *reference_gradients,
                                  double *physical_gradients) {
  if (descriptor == NULL || geometry == NULL || reference_gradients == NULL ||
      physical_gradients == NULL || geometry->dimension != descriptor->dimension) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_fe_physical_gradients_apply(descriptor->dimension, descriptor->node_count,
                                 geometry->inverse_transpose, reference_gradients,
                                 physical_gradients);
  return SS_OK;
}

static void ss_fe_reference_vertex(uint32_t dimension, uint32_t vertex,
                                   double result[SS_FE_MAX_DIMENSION]) {
  uint32_t coordinate;
  for (coordinate = 0U; coordinate < dimension; ++coordinate) {
    result[coordinate] = vertex == coordinate + 1U ? 1.0 : 0.0;
  }
}

SsStatus ss_fe_facet_parent_reference(uint32_t dimension, uint32_t facet_index,
                                      const double *facet_reference_point,
                                      double *parent_reference_point) {
  static const uint32_t triangle_facets[3][2] = {
      {0U, 1U},
      {1U, 2U},
      {0U, 2U},
  };
  static const uint32_t tetrahedron_facets[4][3] = {
      {0U, 1U, 2U},
      {0U, 1U, 3U},
      {0U, 2U, 3U},
      {1U, 2U, 3U},
  };
  double first[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  double second[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  double third[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  uint32_t coordinate;
  if (facet_reference_point == NULL || parent_reference_point == NULL ||
      (dimension != 2U && dimension != 3U) || (dimension == 2U && facet_index >= 3U) ||
      (dimension == 3U && facet_index >= 4U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(parent_reference_point, 0,
         SS_FE_MAX_DIMENSION * sizeof(*parent_reference_point));
  if (dimension == 2U) {
    ss_fe_reference_vertex(dimension, triangle_facets[facet_index][0], first);
    ss_fe_reference_vertex(dimension, triangle_facets[facet_index][1], second);
    for (coordinate = 0U; coordinate < dimension; ++coordinate) {
      parent_reference_point[coordinate] =
          first[coordinate] +
          facet_reference_point[0] * (second[coordinate] - first[coordinate]);
    }
  } else {
    ss_fe_reference_vertex(dimension, tetrahedron_facets[facet_index][0], first);
    ss_fe_reference_vertex(dimension, tetrahedron_facets[facet_index][1], second);
    ss_fe_reference_vertex(dimension, tetrahedron_facets[facet_index][2], third);
    for (coordinate = 0U; coordinate < dimension; ++coordinate) {
      parent_reference_point[coordinate] =
          first[coordinate] +
          facet_reference_point[0] * (second[coordinate] - first[coordinate]) +
          facet_reference_point[1] * (third[coordinate] - first[coordinate]);
    }
  }
  for (coordinate = 0U; coordinate < dimension; ++coordinate) {
    if (!isfinite(parent_reference_point[coordinate])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  return SS_OK;
}

SsStatus ss_fe_facet_geometry_eval(const SsFeDescriptor *descriptor,
                                   uint32_t facet_index, const double *coordinates,
                                   const double *facet_reference_point,
                                   const double *parent_centroid,
                                   SsFeFacetGeometry *facet) {
  static const uint32_t triangle_facets[3][2] = {
      {0U, 1U},
      {1U, 2U},
      {0U, 2U},
  };
  static const uint32_t tetrahedron_facets[4][3] = {
      {0U, 1U, 2U},
      {0U, 1U, 3U},
      {0U, 2U, 3U},
      {1U, 2U, 3U},
  };
  /* Sign that turns the parameterization normal (rotated tangent in 2D, the
   * cross product of the two reference edge vectors in 3D) into the outward
   * reference normal.  The cofactor of a positive-determinant Jacobian maps
   * oriented normals to oriented normals, so the same sign is outward on every
   * admissible curved cell; no point-dependent centroid test is needed. */
  static const double triangle_orientation[3] = {1.0, 1.0, -1.0};
  static const double tetrahedron_orientation[4] = {-1.0, 1.0, -1.0, 1.0};
  const SsFeGeometryPolicy policy = {DBL_EPSILON, 1.0 / DBL_EPSILON};
  SsFeGeometry geometry;
  double reference_point[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  double first_vertex[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  double second_vertex[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  double third_vertex[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  double reference_tangent0[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  double reference_tangent1[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  double orientation;
  uint32_t coordinate;
  (void)parent_centroid;
  if (descriptor == NULL || coordinates == NULL || facet_reference_point == NULL ||
      facet == NULL || (descriptor->dimension != 2U && descriptor->dimension != 3U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if ((descriptor->dimension == 2U && facet_index >= 3U) ||
      (descriptor->dimension == 3U && facet_index >= 4U)) {
    return SS_ERROR_NOT_FOUND;
  }
  memset(facet, 0, sizeof(*facet));
  facet->dimension = descriptor->dimension;
  if (descriptor->dimension == 2U) {
    const double parameter = facet_reference_point[0];
    ss_fe_reference_vertex(2U, triangle_facets[facet_index][0], first_vertex);
    ss_fe_reference_vertex(2U, triangle_facets[facet_index][1], second_vertex);
    for (coordinate = 0U; coordinate < 2U; ++coordinate) {
      reference_tangent0[coordinate] =
          second_vertex[coordinate] - first_vertex[coordinate];
      reference_point[coordinate] =
          first_vertex[coordinate] + parameter * reference_tangent0[coordinate];
    }
  } else {
    const double first_parameter = facet_reference_point[0];
    const double second_parameter = facet_reference_point[1];
    ss_fe_reference_vertex(3U, tetrahedron_facets[facet_index][0], first_vertex);
    ss_fe_reference_vertex(3U, tetrahedron_facets[facet_index][1], second_vertex);
    ss_fe_reference_vertex(3U, tetrahedron_facets[facet_index][2], third_vertex);
    for (coordinate = 0U; coordinate < 3U; ++coordinate) {
      reference_tangent0[coordinate] =
          second_vertex[coordinate] - first_vertex[coordinate];
      reference_tangent1[coordinate] =
          third_vertex[coordinate] - first_vertex[coordinate];
      reference_point[coordinate] = first_vertex[coordinate] +
                                    first_parameter * reference_tangent0[coordinate] +
                                    second_parameter * reference_tangent1[coordinate];
    }
  }
  if (ss_fe_geometry_eval(descriptor, coordinates, reference_point, &policy,
                          &geometry) != SS_OK) {
    return SS_ERROR_INTEGRITY;
  }
  memcpy(facet->physical, geometry.physical, sizeof(facet->physical));
  for (coordinate = 0U; coordinate < descriptor->dimension; ++coordinate) {
    uint32_t reference_coordinate;
    for (reference_coordinate = 0U; reference_coordinate < descriptor->dimension;
         ++reference_coordinate) {
      facet->tangents[coordinate] +=
          geometry.jacobian[coordinate * (size_t)descriptor->dimension +
                            reference_coordinate] *
          reference_tangent0[reference_coordinate];
      if (descriptor->dimension == 3U) {
        facet->tangents[3U + coordinate] +=
            geometry.jacobian[coordinate * 3U + reference_coordinate] *
            reference_tangent1[reference_coordinate];
      }
    }
  }
  if (descriptor->dimension == 2U) {
    facet->outward_normal[0] = facet->tangents[1];
    facet->outward_normal[1] = -facet->tangents[0];
    facet->measure = hypot(facet->outward_normal[0], facet->outward_normal[1]);
  } else {
    facet->outward_normal[0] = facet->tangents[1] * facet->tangents[5] -
                               facet->tangents[2] * facet->tangents[4];
    facet->outward_normal[1] = facet->tangents[2] * facet->tangents[3] -
                               facet->tangents[0] * facet->tangents[5];
    facet->outward_normal[2] = facet->tangents[0] * facet->tangents[4] -
                               facet->tangents[1] * facet->tangents[3];
    facet->measure = sqrt(facet->outward_normal[0] * facet->outward_normal[0] +
                          facet->outward_normal[1] * facet->outward_normal[1] +
                          facet->outward_normal[2] * facet->outward_normal[2]);
  }
  if (!isfinite(facet->measure) || facet->measure <= DBL_EPSILON) {
    return SS_ERROR_INTEGRITY;
  }
  orientation = descriptor->dimension == 2U ? triangle_orientation[facet_index]
                                            : tetrahedron_orientation[facet_index];
  for (coordinate = 0U; coordinate < descriptor->dimension; ++coordinate) {
    /* Negation is exact, so straight-sided normals stay bit-identical. */
    facet->outward_normal[coordinate] =
        orientation * (facet->outward_normal[coordinate] / facet->measure);
  }
  return SS_OK;
}

SsStatus ss_fe_permutation_find(const SsFeDescriptor *descriptor, SsFeOrdering ordering,
                                const SsFePermutation **permutation) {
  size_t descriptor_index;
  if (descriptor == NULL || permutation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *permutation = NULL;
  for (descriptor_index = 0U; descriptor_index < ss_fe_generated_descriptor_count;
       ++descriptor_index) {
    if (descriptor->dimension ==
            ss_fe_generated_descriptors[descriptor_index].dimension &&
        descriptor->order == ss_fe_generated_descriptors[descriptor_index].order) {
      size_t permutation_index;
      for (permutation_index = descriptor_index * 5U;
           permutation_index < descriptor_index * 5U + 5U &&
           permutation_index < ss_fe_generated_permutation_count;
           ++permutation_index) {
        if (ss_fe_generated_permutations[permutation_index].ordering == ordering) {
          *permutation = &ss_fe_generated_permutations[permutation_index];
          return SS_OK;
        }
      }
      return SS_ERROR_NOT_FOUND;
    }
  }
  return SS_ERROR_NOT_FOUND;
}

SsStatus ss_fe_permute_to_external(const SsFePermutation *permutation,
                                   size_t components, const double *canonical,
                                   double *external) {
  size_t canonical_node;
  if (permutation == NULL || components == 0U || canonical == NULL ||
      external == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (canonical_node = 0U; canonical_node < permutation->node_count;
       ++canonical_node) {
    const size_t external_node = permutation->canonical_to_external[canonical_node];
    size_t component;
    for (component = 0U; component < components; ++component) {
      external[external_node * components + component] =
          canonical[canonical_node * components + component];
    }
  }
  return SS_OK;
}

SsStatus ss_fe_permute_to_canonical(const SsFePermutation *permutation,
                                    size_t components, const double *external,
                                    double *canonical) {
  size_t external_node;
  if (permutation == NULL || components == 0U || external == NULL ||
      canonical == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (external_node = 0U; external_node < permutation->node_count; ++external_node) {
    const size_t canonical_node = permutation->external_to_canonical[external_node];
    size_t component;
    for (component = 0U; component < components; ++component) {
      canonical[canonical_node * components + component] =
          external[external_node * components + component];
    }
  }
  return SS_OK;
}

SsStatus ss_fe_transfer_build(const SsFeDescriptor *coarse, const SsFeDescriptor *fine,
                              double *matrix, size_t capacity, size_t *rows,
                              size_t *columns) {
  double point[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  size_t fine_node;
  if (coarse == NULL || fine == NULL || matrix == NULL || rows == NULL ||
      columns == NULL || coarse->dimension != fine->dimension ||
      coarse->order >= fine->order ||
      capacity < coarse->node_count * fine->node_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *rows = fine->node_count;
  *columns = coarse->node_count;
  for (fine_node = 0U; fine_node < fine->node_count; ++fine_node) {
    uint32_t coordinate;
    for (coordinate = 0U; coordinate < fine->dimension; ++coordinate) {
      const size_t offset =
          fine_node * (size_t)(fine->dimension + 1U) + coordinate + 1U;
      point[coordinate] =
          (double)fine->barycentric_indices[offset] / (double)fine->order;
    }
    if (ss_fe_basis_eval(coarse, 1U, point, matrix + fine_node * coarse->node_count,
                         NULL) != SS_OK) {
      return SS_ERROR_INTERNAL;
    }
  }
  return SS_OK;
}

SsStatus ss_fe_transfer_apply_constrained(size_t rows, size_t columns,
                                          const double *matrix, size_t components,
                                          const double *coarse_values,
                                          const uint8_t *constrained_rows,
                                          double *fine_values) {
  size_t row;
  if (rows == 0U || columns == 0U || matrix == NULL || components == 0U ||
      coarse_values == NULL || constrained_rows == NULL || fine_values == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (row = 0U; row < rows; ++row) {
    size_t component;
    for (component = 0U; component < components; ++component) {
      double value = 0.0;
      size_t column;
      if (constrained_rows[row] == 0U) {
        for (column = 0U; column < columns; ++column) {
          value += matrix[row * columns + column] *
                   coarse_values[column * components + component];
        }
      }
      fine_values[row * components + component] = value;
    }
  }
  return SS_OK;
}
