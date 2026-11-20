/* Finite-element tests: bases, P4 derivatives, quadrature, curved geometry,
 * facets, permutations and transfers.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/fe/fe.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__,             \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                        \
  do {                                                                                 \
    const double check_actual = (actual);                                              \
    const double check_expected = (expected);                                          \
    if (!isfinite(check_actual) ||                                                     \
        fabs(check_actual - check_expected) > (tolerance)) {                           \
      fprintf(stderr, "%s:%d: |%.17g - %.17g| > %.3g\n", __FILE__, __LINE__,           \
              check_actual, check_expected, (double)(tolerance));                      \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static double integer_power(double value, uint32_t exponent) {
  double result = 1.0;
  uint32_t index;
  for (index = 0U; index < exponent; ++index) {
    result *= value;
  }
  return result;
}

static double factorial(uint32_t value) {
  double result = 1.0;
  uint32_t index;
  for (index = 2U; index <= value; ++index) {
    result *= (double)index;
  }
  return result;
}

static double simplex_monomial(uint32_t dimension, uint32_t a, uint32_t b, uint32_t c) {
  const uint32_t degree = a + b + (dimension == 3U ? c : 0U);
  double numerator = factorial(a) * factorial(b);
  if (dimension == 3U) {
    numerator *= factorial(c);
  }
  return numerator / factorial(dimension + degree);
}

static double node_coordinate(const SsFeDescriptor *descriptor, size_t node,
                              uint32_t coordinate) {
  const size_t offset = node * (size_t)(descriptor->dimension + 1U);
  return (double)descriptor->barycentric_indices[offset + (size_t)coordinate + 1U] /
         (double)descriptor->order;
}

static int descriptor_matrix(const SsFeDescriptor *descriptors[8]) {
  const uint32_t dimensions[] = {2U, 2U, 2U, 2U, 3U, 3U, 3U, 3U};
  const uint32_t orders[] = {1U, 2U, 4U, 8U, 1U, 2U, 4U, 8U};
  const uint32_t nodes[] = {3U, 6U, 15U, 45U, 4U, 10U, 35U, 165U};
  const uint32_t facet_nodes[] = {2U, 3U, 5U, 9U, 3U, 6U, 15U, 45U};
  const SsFeDescriptor *unsupported = NULL;
  size_t index;

  for (index = 0U; index < 8U; ++index) {
    const SsFeDescriptor *descriptor = NULL;
    CHECK(ss_fe_descriptor_find(dimensions[index], orders[index], &descriptor) ==
          SS_OK);
    CHECK(descriptor != NULL);
    CHECK(descriptor->dimension == dimensions[index]);
    CHECK(descriptor->order == orders[index]);
    CHECK(descriptor->node_count == nodes[index]);
    CHECK(descriptor->facet_node_count == facet_nodes[index]);
    CHECK(descriptor->barycentric_indices != NULL);
    CHECK(descriptor->entity_dimensions != NULL);
    CHECK(descriptor->entity_indices != NULL);
    CHECK(descriptor->entity_slots != NULL);
    CHECK(descriptor->basis_scalar_eval != NULL);
    CHECK(descriptor->basis_batch_eval != NULL);
    CHECK(descriptor->basis_digest != NULL);
    CHECK(strncmp(descriptor->basis_digest, "sha256:", 7U) == 0);
    CHECK(descriptor->id != NULL && descriptor->generator_digest != NULL);
    CHECK(strncmp(descriptor->generator_digest, "sha256:", 7U) == 0);
    descriptors[index] = descriptor;
  }
  CHECK(ss_fe_descriptor_find(2U, 3U, NULL) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_fe_descriptor_find(2U, 3U, &unsupported) == SS_ERROR_NOT_FOUND);
  CHECK(unsupported == NULL);
  return 0;
}

static int check_basis_descriptor(const SsFeDescriptor *descriptor) {
  double values[SS_FE_MAX_NODES];
  double gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  double plus_values[SS_FE_MAX_NODES];
  double minus_values[SS_FE_MAX_NODES];
  double xi[SS_FE_MAX_DIMENSION] = {0.17, 0.23, 0.11};
  size_t node;
  size_t basis;
  uint32_t coordinate;

  for (node = 0U; node < descriptor->node_count; ++node) {
    for (coordinate = 0U; coordinate < descriptor->dimension; ++coordinate) {
      xi[coordinate] = node_coordinate(descriptor, node, coordinate);
    }
    CHECK(ss_fe_basis_eval(descriptor, 1U, xi, values, gradients) == SS_OK);
    for (basis = 0U; basis < descriptor->node_count; ++basis) {
      CHECK_NEAR(values[basis], basis == node ? 1.0 : 0.0, 2.0e-12);
    }
  }

  xi[0] = 0.17;
  xi[1] = 0.23;
  xi[2] = 0.11;
  CHECK(ss_fe_basis_eval(descriptor, 1U, xi, values, gradients) == SS_OK);
  {
    double scalar_values[SS_FE_MAX_NODES] = {0.0};
    double scalar_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
    size_t value;
    CHECK(ss_fe_basis_eval_scalar(descriptor, xi, scalar_values, scalar_gradients) ==
          SS_OK);
    for (value = 0U; value < descriptor->node_count * (size_t)descriptor->dimension;
         ++value) {
      CHECK(scalar_gradients[value] == gradients[value]);
    }
    for (value = 0U; value < descriptor->node_count; ++value) {
      CHECK(scalar_values[value] == values[value]);
    }
  }
  {
    double value_sum = 0.0;
    double gradient_sum[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
    for (basis = 0U; basis < descriptor->node_count; ++basis) {
      value_sum += values[basis];
      for (coordinate = 0U; coordinate < descriptor->dimension; ++coordinate) {
        gradient_sum[coordinate] +=
            gradients[basis * descriptor->dimension + coordinate];
      }
    }
    CHECK_NEAR(value_sum, 1.0, 2.0e-12);
    for (coordinate = 0U; coordinate < descriptor->dimension; ++coordinate) {
      CHECK_NEAR(gradient_sum[coordinate], 0.0, 2.0e-12);
    }
  }

  for (coordinate = 0U; coordinate < descriptor->dimension; ++coordinate) {
    double plus[SS_FE_MAX_DIMENSION] = {xi[0], xi[1], xi[2]};
    double minus[SS_FE_MAX_DIMENSION] = {xi[0], xi[1], xi[2]};
    plus[coordinate] += 1.0e-7;
    minus[coordinate] -= 1.0e-7;
    CHECK(ss_fe_basis_eval(descriptor, 1U, plus, plus_values, NULL) == SS_OK);
    CHECK(ss_fe_basis_eval(descriptor, 1U, minus, minus_values, NULL) == SS_OK);
    for (basis = 0U; basis < descriptor->node_count; ++basis) {
      const double finite_difference =
          (plus_values[basis] - minus_values[basis]) / 2.0e-7;
      CHECK_NEAR(gradients[basis * descriptor->dimension + coordinate],
                 finite_difference, 2.0e-7);
    }
  }

  {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    for (a = 0U; a <= descriptor->order; ++a) {
      for (b = 0U; b + a <= descriptor->order; ++b) {
        const uint32_t c_limit =
            descriptor->dimension == 3U ? descriptor->order - a - b : 0U;
        for (c = 0U; c <= c_limit; ++c) {
          double interpolated = 0.0;
          const double expected = integer_power(xi[0], a) * integer_power(xi[1], b) *
                                  integer_power(xi[2], c);
          for (node = 0U; node < descriptor->node_count; ++node) {
            const double nodal_value =
                integer_power(node_coordinate(descriptor, node, 0U), a) *
                integer_power(node_coordinate(descriptor, node, 1U), b) *
                integer_power(descriptor->dimension == 3U
                                  ? node_coordinate(descriptor, node, 2U)
                                  : 1.0,
                              c);
            interpolated += values[node] * nodal_value;
          }
          CHECK_NEAR(interpolated, expected, 3.0e-12);
          if (descriptor->dimension == 2U) {
            break;
          }
        }
      }
    }
  }
  return 0;
}

static int test_basis(void) {
  const SsFeDescriptor *descriptors[8] = {0};
  size_t index;
  CHECK(descriptor_matrix(descriptors) == 0);
  for (index = 0U; index < 8U; ++index) {
    CHECK(check_basis_descriptor(descriptors[index]) == 0);
  }
  return 0;
}

static int test_p4_derivative(void) {
  const SsFeDescriptor *descriptor = NULL;
  const double xi[2] = {0.063089014491502, 0.873821971016996};
  double values[SS_FE_MAX_NODES];
  double gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  double plus_values[SS_FE_MAX_NODES];
  double minus_values[SS_FE_MAX_NODES];
  double plus[2] = {xi[0], xi[1] + 1.0e-7};
  double minus[2] = {xi[0], xi[1] - 1.0e-7};
  size_t defect_basis = SS_FE_MAX_NODES;
  size_t basis;

  CHECK(ss_fe_descriptor_find(2U, 4U, &descriptor) == SS_OK);
  for (basis = 0U; basis < descriptor->node_count; ++basis) {
    const size_t offset = basis * 3U;
    if (descriptor->barycentric_indices[offset] == 3U &&
        descriptor->barycentric_indices[offset + 1U] == 0U &&
        descriptor->barycentric_indices[offset + 2U] == 1U) {
      defect_basis = basis;
      break;
    }
  }
  CHECK(defect_basis < descriptor->node_count);
  CHECK(ss_fe_basis_eval(descriptor, 1U, xi, values, gradients) == SS_OK);
  CHECK(ss_fe_basis_eval(descriptor, 1U, plus, plus_values, NULL) == SS_OK);
  CHECK(ss_fe_basis_eval(descriptor, 1U, minus, minus_values, NULL) == SS_OK);
  CHECK_NEAR(gradients[defect_basis * 2U + 1U],
             (plus_values[defect_basis] - minus_values[defect_basis]) / 2.0e-7, 2.0e-7);
  CHECK_NEAR(gradients[defect_basis * 2U + 1U], -1.3575189060366366, 2.0e-8);
  CHECK(fabs(gradients[defect_basis * 2U + 1U] - 11.040961296672785) > 1.0);
  return 0;
}

static int check_quadrature(uint32_t dimension, uint32_t order, size_t expected_points,
                            uint32_t expected_degree, int expected_negative) {
  const SsFeQuadrature *quadrature = NULL;
  size_t bad_index = SIZE_MAX;
  uint32_t a;
  uint32_t b;
  uint32_t c;

  CHECK(ss_fe_quadrature_default(dimension, order, &quadrature) == SS_OK);
  CHECK(quadrature != NULL && quadrature->dimension == dimension);
  CHECK(quadrature->domain == SS_FE_QUADRATURE_VOLUME);
  CHECK(quadrature->parent_dimension == dimension);
  CHECK(quadrature->point_count == expected_points);
  CHECK(quadrature->exact_degree == expected_degree);
  CHECK(quadrature->has_negative_weight == expected_negative);
  CHECK(ss_fe_quadrature_validate(quadrature, &bad_index) == SS_OK);
  CHECK(bad_index == SIZE_MAX);

  for (a = 0U; a <= expected_degree; ++a) {
    for (b = 0U; a + b <= expected_degree; ++b) {
      const uint32_t c_limit = dimension == 3U ? expected_degree - a - b : 0U;
      for (c = 0U; c <= c_limit; ++c) {
        double actual = 0.0;
        size_t point;
        for (point = 0U; point < quadrature->point_count; ++point) {
          const double *xi = quadrature->points + point * (size_t)quadrature->dimension;
          actual += quadrature->weights[point] * integer_power(xi[0], a) *
                    integer_power(xi[1], b) *
                    integer_power(dimension == 3U ? xi[2] : 1.0, c);
        }
        CHECK_NEAR(actual, simplex_monomial(dimension, a, b, c), 3.0e-12);
        if (dimension == 2U) {
          break;
        }
      }
    }
  }
  return 0;
}

static int check_quadrature_plan(uint32_t dimension, SsFeQuadratureDomain domain,
                                 SsFeOperationKind operation, uint32_t field_order,
                                 uint32_t coordinate_order, uint32_t coefficient_order,
                                 uint32_t expected_minimum,
                                 uint32_t expected_default_degree,
                                 uint32_t expected_paired_degree) {
  SsFeQuadratureRequest request = {
      dimension,
      domain,
      operation,
      field_order,
      coordinate_order,
      coefficient_order,
      SS_FE_QUADRATURE_MODE_DEFAULT,
      NULL,
  };
  SsFeQuadraturePlan plan = {0};
  CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_OK);
  CHECK(plan.minimum_exact_degree == expected_minimum);
  CHECK(plan.default_rule != NULL && plan.paired_rule != NULL);
  CHECK(plan.selected_rule == plan.default_rule);
  CHECK(plan.default_rule->parent_dimension == dimension);
  CHECK(plan.default_rule->domain == domain);
  CHECK(plan.default_rule->exact_degree == expected_default_degree);
  CHECK(plan.default_rule->has_negative_weight == 0);
  CHECK(plan.paired_rule->exact_degree == expected_paired_degree);
  CHECK(plan.paired_rule->has_negative_weight == 0);
  CHECK(plan.paired_rule->exact_degree > plan.default_rule->exact_degree);
  CHECK(plan.used_override == 0);

  request.mode = SS_FE_QUADRATURE_MODE_PAIRED_HIGHER;
  memset(&plan, 0, sizeof(plan));
  CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_OK);
  CHECK(plan.selected_rule == plan.paired_rule);
  CHECK(plan.selected_rule->exact_degree == expected_paired_degree);
  return 0;
}

static int test_quadrature_planner(void) {
  static const SsFeOperationKind volume_operations[] = {
      SS_FE_OPERATION_GEOMETRY_QUALITY,
      SS_FE_OPERATION_MECHANICS_STIFFNESS,
      SS_FE_OPERATION_MECHANICS_BODY,
      SS_FE_OPERATION_HYDRAULIC_DIFFUSION,
      SS_FE_OPERATION_MASS,
      SS_FE_OPERATION_MATERIAL_PROJECTION,
      SS_FE_OPERATION_COUPLING_TRANSFER,
  };
  static const SsFeOperationKind facet_operations[] = {
      SS_FE_OPERATION_MECHANICS_NATURAL,
      SS_FE_OPERATION_HYDRAULIC_FLUX,
  };
  static const uint32_t dimensions[] = {2U, 3U};
  static const uint32_t orders[] = {1U, 2U, 4U};
  size_t dimension_index;
  size_t field_index;
  size_t coordinate_index;
  size_t operation_index;

  {
    static const char *const high_rule_ids[] = {
        "triangle-degree-8-duffy-v1",
        "triangle-degree-10-duffy-v1",
        "triangle-degree-12-duffy-v1",
        "tetrahedron-degree-9-duffy-v1",
        "tetrahedron-degree-11-duffy-v1",
        "tetrahedron-degree-13-duffy-v1",
        "triangle-degree-14-petsc-minsym-v1",
        "triangle-degree-15-petsc-minsym-v1",
        "triangle-degree-16-petsc-minsym-v1",
        "triangle-degree-17-petsc-minsym-v1",
        "triangle-degree-18-petsc-minsym-v1",
        "triangle-degree-19-petsc-minsym-v1",
        "tetrahedron-degree-14-petsc-minsym-v1",
        "tetrahedron-degree-15-petsc-minsym-v1",
        "tetrahedron-degree-16-petsc-minsym-v1",
        "tetrahedron-degree-17-petsc-minsym-v1",
        "tetrahedron-degree-18-petsc-minsym-v1",
        "tetrahedron-degree-19-petsc-minsym-v1",
    };
    static const size_t high_rule_points[] = {
        25U, 36U, 49U, 216U, 343U, 512U, 42U,  49U,  55U,
        60U, 67U, 73U, 204U, 264U, 304U, 364U, 436U, 487U,
    };
    static const uint32_t high_rule_degrees[] = {
        8U,  10U, 12U, 9U,  11U, 13U, 14U, 15U, 16U,
        17U, 18U, 19U, 14U, 15U, 16U, 17U, 18U, 19U,
    };
    size_t rule;
    for (rule = 0U; rule < sizeof(high_rule_ids) / sizeof(high_rule_ids[0]); ++rule) {
      const SsFeQuadrature *quadrature = NULL;
      size_t bad_index = SIZE_MAX;
      CHECK(ss_fe_quadrature_find(high_rule_ids[rule], &quadrature) == SS_OK);
      CHECK(quadrature != NULL);
      CHECK(quadrature->point_count == high_rule_points[rule]);
      CHECK(quadrature->exact_degree == high_rule_degrees[rule]);
      CHECK(quadrature->has_negative_weight == 0);
      CHECK(ss_fe_quadrature_validate(quadrature, &bad_index) == SS_OK);
      CHECK(bad_index == SIZE_MAX);
    }
  }

  CHECK(check_quadrature_plan(2U, SS_FE_QUADRATURE_VOLUME,
                              SS_FE_OPERATION_MECHANICS_STIFFNESS, 4U, 1U, 0U, 6U, 6U,
                              8U) == 0);
  CHECK(check_quadrature_plan(3U, SS_FE_QUADRATURE_VOLUME,
                              SS_FE_OPERATION_MECHANICS_STIFFNESS, 2U, 1U, 0U, 2U, 6U,
                              14U) == 0);
  /* Mass on P4/G4 needs 2p + d (g - 1): 8 + 6 in 2D and 8 + 9 in 3D. */
  CHECK(check_quadrature_plan(2U, SS_FE_QUADRATURE_VOLUME, SS_FE_OPERATION_MASS, 4U, 4U,
                              0U, 14U, 14U, 15U) == 0);
  CHECK(check_quadrature_plan(3U, SS_FE_QUADRATURE_VOLUME, SS_FE_OPERATION_MASS, 4U, 4U,
                              0U, 17U, 17U, 18U) == 0);
  CHECK(check_quadrature_plan(2U, SS_FE_QUADRATURE_VOLUME,
                              SS_FE_OPERATION_GEOMETRY_QUALITY, 1U, 4U, 0U, 10U, 10U,
                              14U) == 0);
  CHECK(check_quadrature_plan(3U, SS_FE_QUADRATURE_VOLUME,
                              SS_FE_OPERATION_GEOMETRY_QUALITY, 1U, 4U, 0U, 9U, 14U,
                              15U) == 0);
  /* Facet moments carry the cofactor normal: p + (d - 1)(g - 1). */
  CHECK(check_quadrature_plan(2U, SS_FE_QUADRATURE_FACET,
                              SS_FE_OPERATION_MECHANICS_NATURAL, 4U, 4U, 0U, 7U, 9U,
                              15U) == 0);
  CHECK(check_quadrature_plan(3U, SS_FE_QUADRATURE_FACET,
                              SS_FE_OPERATION_HYDRAULIC_FLUX, 4U, 4U, 0U, 10U, 10U,
                              14U) == 0);

  CHECK(check_quadrature_plan(2U, SS_FE_QUADRATURE_VOLUME,
                              SS_FE_OPERATION_MECHANICS_STIFFNESS, 8U, 1U, 0U, 14U, 14U,
                              15U) == 0);
  CHECK(check_quadrature_plan(3U, SS_FE_QUADRATURE_VOLUME,
                              SS_FE_OPERATION_MECHANICS_STIFFNESS, 8U, 1U, 0U, 14U, 14U,
                              15U) == 0);
  CHECK(check_quadrature_plan(2U, SS_FE_QUADRATURE_VOLUME,
                              SS_FE_OPERATION_MATERIAL_PROJECTION, 8U, 1U, 0U, 16U, 16U,
                              17U) == 0);
  CHECK(check_quadrature_plan(3U, SS_FE_QUADRATURE_VOLUME,
                              SS_FE_OPERATION_MATERIAL_PROJECTION, 8U, 1U, 0U, 16U, 16U,
                              17U) == 0);

  for (dimension_index = 0U; dimension_index < 2U; ++dimension_index) {
    for (field_index = 0U; field_index < 3U; ++field_index) {
      for (coordinate_index = 0U; coordinate_index < 3U; ++coordinate_index) {
        for (operation_index = 0U;
             operation_index < sizeof(volume_operations) / sizeof(volume_operations[0]);
             ++operation_index) {
          SsFeQuadratureRequest request = {
              dimensions[dimension_index],        SS_FE_QUADRATURE_VOLUME,
              volume_operations[operation_index], orders[field_index],
              orders[coordinate_index],           2U,
              SS_FE_QUADRATURE_MODE_DEFAULT,      NULL,
          };
          SsFeQuadraturePlan plan = {0};
          CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_OK);
          CHECK(plan.selected_rule != NULL);
          CHECK(plan.selected_rule->exact_degree >= plan.minimum_exact_degree);
          /* Only the highest registered degree (19) has no paired control. */
          CHECK(plan.paired_rule != NULL || plan.default_rule->exact_degree == 19U);
          CHECK(plan.paired_rule == NULL ||
                plan.paired_rule->exact_degree > plan.default_rule->exact_degree);
        }
        for (operation_index = 0U;
             operation_index < sizeof(facet_operations) / sizeof(facet_operations[0]);
             ++operation_index) {
          SsFeQuadratureRequest request = {
              dimensions[dimension_index],
              SS_FE_QUADRATURE_FACET,
              facet_operations[operation_index],
              orders[field_index],
              orders[coordinate_index],
              2U,
              SS_FE_QUADRATURE_MODE_PAIRED_HIGHER,
              NULL,
          };
          SsFeQuadraturePlan plan = {0};
          CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_OK);
          CHECK(plan.selected_rule == plan.paired_rule);
        }
      }
    }
  }
  {
    SsFeQuadratureRequest request = {
        2U, SS_FE_QUADRATURE_VOLUME,       SS_FE_OPERATION_MASS,          4U, 1U,
        0U, SS_FE_QUADRATURE_MODE_DEFAULT, "triangle-degree-10-duffy-v1",
    };
    SsFeQuadraturePlan plan = {0};
    CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_OK);
    CHECK(plan.used_override != 0);
    CHECK(strcmp(plan.selected_rule->id, request.override_rule_id) == 0);
    CHECK(plan.paired_rule->exact_degree == 14U);
    request.override_rule_id = "triangle-degree-6-v1";
    CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_ERROR_INTEGRITY);
    request.override_rule_id = NULL;
    request.coefficient_order = 3U;
    CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_ERROR_NOT_FOUND);
    request.coefficient_order = 0U;
    request.operation = SS_FE_OPERATION_MECHANICS_NATURAL;
    CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_ERROR_INVALID_ARGUMENT);
  }
  return 0;
}

static int test_quadrature(void) {
  const uint32_t parent_dimensions[] = {2U, 3U};
  const uint32_t orders[] = {1U, 2U, 4U, 8U};
  const size_t facet_points[2][4] = {
      {1U, 3U, 5U, 8U},
      {1U, 7U, 12U, 49U},
  };
  const uint32_t facet_degrees[2][4] = {
      {1U, 5U, 9U, 15U},
      {1U, 5U, 6U, 15U},
  };
  size_t parent_index;
  size_t order_index;
  size_t registry_index;
  CHECK(ss_fe_quadrature_registry_count() > 0U);
  for (registry_index = 0U; registry_index < ss_fe_quadrature_registry_count();
       ++registry_index) {
    const SsFeQuadrature *registered = NULL;
    size_t bad_index = SIZE_MAX;
    CHECK(ss_fe_quadrature_registry_at(registry_index, &registered) == SS_OK);
    CHECK(registered != NULL);
    CHECK(ss_fe_quadrature_validate(registered, &bad_index) == SS_OK);
    CHECK(bad_index == SIZE_MAX);
  }
  {
    const SsFeQuadrature *missing = NULL;
    CHECK(ss_fe_quadrature_registry_at(ss_fe_quadrature_registry_count(), &missing) ==
          SS_ERROR_NOT_FOUND);
    CHECK(missing == NULL);
  }
  CHECK(check_quadrature(2U, 1U, 1U, 1U, 0) == 0);
  CHECK(check_quadrature(2U, 2U, 7U, 5U, 0) == 0);
  CHECK(check_quadrature(2U, 4U, 12U, 6U, 0) == 0);
  CHECK(check_quadrature(2U, 8U, 42U, 14U, 0) == 0);
  CHECK(check_quadrature(3U, 1U, 1U, 1U, 0) == 0);
  CHECK(check_quadrature(3U, 2U, 11U, 4U, 1) == 0);
  CHECK(check_quadrature(3U, 4U, 24U, 6U, 0) == 0);
  CHECK(check_quadrature(3U, 8U, 204U, 14U, 0) == 0);
  for (parent_index = 0U; parent_index < 2U; ++parent_index) {
    for (order_index = 0U; order_index < 4U; ++order_index) {
      const SsFeQuadrature *quadrature = NULL;
      size_t bad_index = SIZE_MAX;
      CHECK(ss_fe_facet_quadrature_default(parent_dimensions[parent_index],
                                           orders[order_index], &quadrature) == SS_OK);
      CHECK(quadrature != NULL);
      CHECK(quadrature->domain == SS_FE_QUADRATURE_FACET);
      CHECK(quadrature->parent_dimension == parent_dimensions[parent_index]);
      CHECK(quadrature->dimension == parent_dimensions[parent_index] - 1U);
      CHECK(quadrature->point_count == facet_points[parent_index][order_index]);
      CHECK(quadrature->exact_degree == facet_degrees[parent_index][order_index]);
      CHECK(ss_fe_quadrature_validate(quadrature, &bad_index) == SS_OK);
      CHECK(bad_index == SIZE_MAX);
    }
  }
  {
    const SsFeQuadrature *paired = NULL;
    size_t bad_index = SIZE_MAX;
    CHECK(ss_fe_quadrature_find("line-degree-15-v1", &paired) == SS_OK);
    CHECK(paired != NULL && paired->domain == SS_FE_QUADRATURE_FACET);
    CHECK(paired->parent_dimension == 2U && paired->dimension == 1U);
    CHECK(paired->point_count == 8U && paired->exact_degree == 15U);
    CHECK(ss_fe_quadrature_validate(paired, &bad_index) == SS_OK);
  }
  CHECK(test_quadrature_planner() == 0);
  return 0;
}

static int test_bad_quadrature(void) {
  static const double points[] = {
      0.25, 0.25, 0.25, 0.60, 0.30, 0.30,
  };
  static const double weights[] = {1.0 / 12.0, 1.0 / 12.0};
  const SsFeQuadrature malformed = {
      "malformed-tetrahedral-branch",
      SS_FE_QUADRATURE_VOLUME,
      3U,
      3U,
      2U,
      2U,
      0,
      points,
      weights,
      "rejection-fixture",
  };
  size_t bad_index = SIZE_MAX;
  CHECK(ss_fe_quadrature_validate(&malformed, &bad_index) == SS_ERROR_INTEGRITY);
  CHECK(bad_index == 1U);
  return 0;
}

static void fill_geometry_nodes(const SsFeDescriptor *descriptor, double *coordinates) {
  size_t node;
  for (node = 0U; node < descriptor->node_count; ++node) {
    const double r = node_coordinate(descriptor, node, 0U);
    const double s = node_coordinate(descriptor, node, 1U);
    coordinates[node * descriptor->dimension] = r;
    coordinates[node * descriptor->dimension + 1U] = s;
    if (descriptor->dimension == 2U) {
      coordinates[node * 2U + 1U] = s * (1.0 + 0.3 * r);
    } else {
      const double t = node_coordinate(descriptor, node, 2U);
      coordinates[node * 3U + 2U] = t * (1.0 + 0.4 * r);
    }
  }
}

static int check_curved_geometry(uint32_t dimension, uint32_t order) {
  const SsFeDescriptor *descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  const SsFeGeometryPolicy policy = {1.0e-13, 1.0e12};
  double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
  double measure = 0.0;
  size_t point;

  CHECK(ss_fe_descriptor_find(dimension, order, &descriptor) == SS_OK);
  CHECK(ss_fe_quadrature_default(dimension, order, &quadrature) == SS_OK);
  fill_geometry_nodes(descriptor, coordinates);
  for (point = 0U; point < quadrature->point_count; ++point) {
    const double *xi = quadrature->points + point * (size_t)quadrature->dimension;
    SsFeGeometry geometry = {0};
    CHECK(ss_fe_geometry_eval(descriptor, coordinates, xi, &policy, &geometry) ==
          SS_OK);
    CHECK(geometry.dimension == dimension);
    CHECK_NEAR(geometry.physical[0], xi[0], 3.0e-12);
    CHECK_NEAR(geometry.physical[1],
               dimension == 2U ? xi[1] * (1.0 + 0.3 * xi[0]) : xi[1], 3.0e-12);
    if (dimension == 2U) {
      CHECK_NEAR(geometry.jacobian[0], 1.0, 3.0e-12);
      CHECK_NEAR(geometry.jacobian[1], 0.0, 3.0e-12);
      CHECK_NEAR(geometry.jacobian[2], 0.3 * xi[1], 3.0e-12);
      CHECK_NEAR(geometry.jacobian[3], 1.0 + 0.3 * xi[0], 3.0e-12);
      CHECK_NEAR(geometry.determinant, 1.0 + 0.3 * xi[0], 3.0e-12);
    } else {
      CHECK_NEAR(geometry.physical[2], xi[2] * (1.0 + 0.4 * xi[0]), 3.0e-12);
      CHECK_NEAR(geometry.jacobian[6], 0.4 * xi[2], 3.0e-12);
      CHECK_NEAR(geometry.jacobian[8], 1.0 + 0.4 * xi[0], 3.0e-12);
      CHECK_NEAR(geometry.determinant, 1.0 + 0.4 * xi[0], 3.0e-12);
    }
    measure += quadrature->weights[point] * geometry.determinant;
  }
  CHECK_NEAR(measure, dimension == 2U ? 0.55 : 0.18333333333333332, 3.0e-12);
  return 0;
}

static int test_curved(void) {
  const SsFeDescriptor *descriptor = NULL;
  const SsFeGeometryPolicy policy = {1.0e-12, 1.0e12};
  double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
  double xi[2] = {0.2, 0.3};
  SsFeGeometry geometry = {0};
  size_t bad_index = SIZE_MAX;

  CHECK(check_curved_geometry(2U, 2U) == 0);
  CHECK(check_curved_geometry(2U, 4U) == 0);
  CHECK(check_curved_geometry(3U, 2U) == 0);
  CHECK(check_curved_geometry(3U, 4U) == 0);

  CHECK(ss_fe_descriptor_find(2U, 1U, &descriptor) == SS_OK);
  fill_geometry_nodes(descriptor, coordinates);
  coordinates[2U] = -1.0;
  CHECK(ss_fe_geometry_eval(descriptor, coordinates, xi, &policy, &geometry) ==
        SS_ERROR_INTEGRITY);
  CHECK(ss_fe_geometry_audit(descriptor, coordinates, 1U, xi, &policy, &bad_index) ==
        SS_ERROR_INTEGRITY);
  CHECK(bad_index == 0U);
  {
    const uint32_t dimensions[] = {2U, 3U};
    size_t dimension_index;
    for (dimension_index = 0U; dimension_index < 2U; ++dimension_index) {
      const uint32_t dimension = dimensions[dimension_index];
      const double fold_root = 0.92;
      const double fold_scale = 1.0 / (fold_root * fold_root * fold_root);
      const SsFeQuadrature *parity = NULL;
      SsFeQuadratureRequest request = {
          dimension,
          SS_FE_QUADRATURE_VOLUME,
          SS_FE_OPERATION_GEOMETRY_QUALITY,
          1U,
          4U,
          0U,
          SS_FE_QUADRATURE_MODE_DEFAULT,
          NULL,
      };
      SsFeQuadraturePlan plan = {0};
      size_t node;
      CHECK(ss_fe_descriptor_find(dimension, 4U, &descriptor) == SS_OK);
      for (node = 0U; node < descriptor->node_count; ++node) {
        const double r = node_coordinate(descriptor, node, 0U);
        const double s = node_coordinate(descriptor, node, 1U);
        coordinates[node * dimension] = r;
        coordinates[node * dimension + 1U] =
            dimension == 2U ? s * (1.0 - fold_scale * r * r * r) : s;
        if (dimension == 3U) {
          const double t = node_coordinate(descriptor, node, 2U);
          coordinates[node * 3U + 2U] = t * (1.0 - fold_scale * r * r * r);
        }
      }
      CHECK(ss_fe_quadrature_default(dimension, 4U, &parity) == SS_OK);
      CHECK(ss_fe_geometry_audit(descriptor, coordinates, parity->point_count,
                                 parity->points, &policy, &bad_index) == SS_OK);
      CHECK(bad_index == SIZE_MAX);
      CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_OK);
      CHECK(plan.selected_rule->exact_degree == (dimension == 2U ? 10U : 14U));
      CHECK(ss_fe_geometry_audit(
                descriptor, coordinates, plan.selected_rule->point_count,
                plan.selected_rule->points, &policy, &bad_index) == SS_ERROR_INTEGRITY);
      CHECK(bad_index < plan.selected_rule->point_count);
    }
  }
  return 0;
}

static int check_curved_tetrahedron_facet(uint32_t order) {
  const double curvature = 0.2;
  const double squared_slope = 4.0 * curvature * curvature;
  const double exact_measure =
      0.5 * (sqrt(1.0 + squared_slope) +
             asinh(sqrt(squared_slope)) / sqrt(squared_slope)) -
      (pow(1.0 + squared_slope, 1.5) - 1.0) / (3.0 * squared_slope);
  const SsFeDescriptor *descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
  const double parent_centroid[] = {
      0.25,
      0.25,
      0.25 + curvature * 0.25 * 0.25,
  };
  double measure = 0.0;
  double resultant[3] = {0.0, 0.0, 0.0};
  size_t node;
  size_t point;
  CHECK(ss_fe_descriptor_find(3U, order, &descriptor) == SS_OK);
  CHECK(ss_fe_facet_quadrature_default(3U, order, &quadrature) == SS_OK);
  for (node = 0U; node < descriptor->node_count; ++node) {
    const double r = node_coordinate(descriptor, node, 0U);
    const double s = node_coordinate(descriptor, node, 1U);
    const double t = node_coordinate(descriptor, node, 2U);
    coordinates[node * 3U] = r;
    coordinates[node * 3U + 1U] = s;
    coordinates[node * 3U + 2U] = t + curvature * r * r;
  }
  for (point = 0U; point < quadrature->point_count; ++point) {
    const double *eta = quadrature->points + point * 2U;
    const double expected_slope = 2.0 * curvature * eta[0];
    const double expected_measure = sqrt(1.0 + expected_slope * expected_slope);
    SsFeFacetGeometry facet = {0};
    uint32_t coordinate;
    CHECK(ss_fe_facet_geometry_eval(descriptor, 0U, coordinates, eta, parent_centroid,
                                    &facet) == SS_OK);
    CHECK_NEAR(facet.physical[0], eta[0], 3.0e-12);
    CHECK_NEAR(facet.physical[1], eta[1], 3.0e-12);
    CHECK_NEAR(facet.physical[2], curvature * eta[0] * eta[0], 3.0e-12);
    CHECK_NEAR(facet.measure, expected_measure, 3.0e-12);
    CHECK_NEAR(facet.outward_normal[0], expected_slope / expected_measure, 3.0e-12);
    CHECK_NEAR(facet.outward_normal[1], 0.0, 3.0e-12);
    CHECK_NEAR(facet.outward_normal[2], -1.0 / expected_measure, 3.0e-12);
    measure += quadrature->weights[point] * facet.measure;
    for (coordinate = 0U; coordinate < 3U; ++coordinate) {
      resultant[coordinate] +=
          quadrature->weights[point] * facet.measure * facet.outward_normal[coordinate];
    }
  }
  CHECK_NEAR(measure, exact_measure, 2.0e-8);
  CHECK_NEAR(resultant[0], curvature / 3.0, 3.0e-12);
  CHECK_NEAR(resultant[1], 0.0, 3.0e-12);
  CHECK_NEAR(resultant[2], -0.5, 3.0e-12);
  return 0;
}

static int test_facet(void) {
  static const double triangle_parent[3][3] = {
      {0.25, 0.0, 0.0},
      {0.75, 0.25, 0.0},
      {0.0, 0.25, 0.0},
  };
  static const double tetrahedron_parent[4][3] = {
      {0.2, 0.3, 0.0},
      {0.2, 0.0, 0.3},
      {0.0, 0.2, 0.3},
      {0.5, 0.2, 0.3},
  };
  const SsFeDescriptor *triangle = NULL;
  const SsFeDescriptor *tetrahedron = NULL;
  const SsFeQuadrature *line_quadrature = NULL;
  double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
  const double triangle_centroid[3] = {1.0 / 3.0, 4.0 / 9.0, 0.0};
  const double tetra_centroid[3] = {0.25, 0.25, 0.25};
  double length = 0.0;
  double resultant[3] = {0.0, 0.0, 0.0};
  size_t point;
  uint32_t facet_index;

  for (facet_index = 0U; facet_index < 3U; ++facet_index) {
    const double eta[2] = {0.25, 0.0};
    double parent[3] = {NAN, NAN, NAN};
    CHECK(ss_fe_facet_parent_reference(2U, facet_index, eta, parent) == SS_OK);
    CHECK_NEAR(parent[0], triangle_parent[facet_index][0], 0.0);
    CHECK_NEAR(parent[1], triangle_parent[facet_index][1], 0.0);
    CHECK_NEAR(parent[2], triangle_parent[facet_index][2], 0.0);
  }
  for (facet_index = 0U; facet_index < 4U; ++facet_index) {
    const double eta[2] = {0.2, 0.3};
    double parent[3] = {NAN, NAN, NAN};
    CHECK(ss_fe_facet_parent_reference(3U, facet_index, eta, parent) == SS_OK);
    CHECK_NEAR(parent[0], tetrahedron_parent[facet_index][0], 0.0);
    CHECK_NEAR(parent[1], tetrahedron_parent[facet_index][1], 0.0);
    CHECK_NEAR(parent[2], tetrahedron_parent[facet_index][2], 0.0);
  }
  {
    const double eta[2] = {0.25, 0.0};
    const double bad_eta[2] = {NAN, 0.0};
    double parent[3] = {0.0, 0.0, 0.0};
    CHECK(ss_fe_facet_parent_reference(1U, 0U, eta, parent) ==
          SS_ERROR_INVALID_ARGUMENT);
    CHECK(ss_fe_facet_parent_reference(2U, 3U, eta, parent) ==
          SS_ERROR_INVALID_ARGUMENT);
    CHECK(ss_fe_facet_parent_reference(3U, 4U, eta, parent) ==
          SS_ERROR_INVALID_ARGUMENT);
    CHECK(ss_fe_facet_parent_reference(2U, 0U, NULL, parent) ==
          SS_ERROR_INVALID_ARGUMENT);
    CHECK(ss_fe_facet_parent_reference(2U, 0U, eta, NULL) == SS_ERROR_INVALID_ARGUMENT);
    CHECK(ss_fe_facet_parent_reference(2U, 0U, bad_eta, parent) ==
          SS_ERROR_INVALID_ARGUMENT);
  }

  CHECK(ss_fe_descriptor_find(2U, 4U, &triangle) == SS_OK);
  CHECK(ss_fe_quadrature_find("line-degree-15-v1", &line_quadrature) == SS_OK);
  fill_geometry_nodes(triangle, coordinates);
  for (point = 0U; point < line_quadrature->point_count; ++point) {
    const double eta[2] = {line_quadrature->points[point], 0.0};
    SsFeFacetGeometry facet = {0};
    CHECK(ss_fe_facet_geometry_eval(triangle, 1U, coordinates, eta, triangle_centroid,
                                    &facet) == SS_OK);
    CHECK(facet.dimension == 2U && facet.measure > 0.0);
    CHECK(facet.outward_normal[0] > 0.0 && facet.outward_normal[1] > 0.0);
    length += line_quadrature->weights[point] * facet.measure;
    resultant[0] +=
        line_quadrature->weights[point] * facet.measure * facet.outward_normal[0];
    resultant[1] +=
        line_quadrature->weights[point] * facet.measure * facet.outward_normal[1];
  }
  CHECK_NEAR(length, 1.4195701087187969, 2.0e-12);
  CHECK_NEAR(resultant[0], 1.0, 2.0e-12);
  CHECK_NEAR(resultant[1], 1.0, 2.0e-12);

  CHECK(ss_fe_descriptor_find(3U, 1U, &tetrahedron) == SS_OK);
  fill_geometry_nodes(tetrahedron, coordinates);
  {
    const double eta[2] = {1.0 / 3.0, 1.0 / 3.0};
    SsFeFacetGeometry facet = {0};
    CHECK(ss_fe_facet_geometry_eval(tetrahedron, 3U, coordinates, eta, tetra_centroid,
                                    &facet) == SS_OK);
    CHECK_NEAR(facet.measure, sqrt(3.0), 2.0e-12);
    CHECK_NEAR(facet.outward_normal[0], 1.0 / sqrt(3.0), 2.0e-12);
    CHECK_NEAR(facet.outward_normal[1], 1.0 / sqrt(3.0), 2.0e-12);
    CHECK_NEAR(facet.outward_normal[2], 1.0 / sqrt(3.0), 2.0e-12);
  }
  CHECK(check_curved_tetrahedron_facet(2U) == 0);
  CHECK(check_curved_tetrahedron_facet(4U) == 0);
  return 0;
}

static int test_permutations(void) {
  const SsFeOrdering orderings[] = {
      SS_FE_ORDER_CANONICAL, SS_FE_ORDER_PETSC,     SS_FE_ORDER_GMSH,
      SS_FE_ORDER_VTK,       SS_FE_ORDER_REFERENCE,
  };
  const uint8_t vtk_p4_triangle_expected[] = {
      0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 11U, 10U, 9U, 12U, 13U, 14U,
  };
  const uint8_t gmsh_p4_tetra_expected[] = {
      0U,  1U,  2U,  3U,  4U,  5U,  6U,  7U,  8U,  9U,  12U, 11U,
      10U, 21U, 20U, 19U, 18U, 17U, 16U, 15U, 14U, 13U, 22U, 24U,
      23U, 25U, 26U, 27U, 28U, 30U, 29U, 33U, 31U, 32U, 34U,
  };
  const uint8_t vtk_p4_tetra_expected[] = {
      0U,  1U,  2U,  3U,  4U,  5U,  6U,  7U,  8U,  9U,  12U, 11U,
      10U, 19U, 20U, 21U, 13U, 14U, 15U, 16U, 17U, 18U, 25U, 26U,
      27U, 32U, 33U, 31U, 28U, 30U, 29U, 22U, 24U, 23U, 34U,
  };
  const uint8_t legacy_p4_triangle_to_vtk_expected[] = {
      0U, 1U, 2U, 6U, 3U, 7U, 8U, 4U, 9U, 10U, 5U, 11U, 12U, 13U, 14U,
  };
  const size_t vtk_p8_triangle_slots[] = {24U, 25U, 39U, 44U};
  const uint8_t vtk_p8_triangle_barycentric[][3] = {
      {6U, 1U, 1U},
      {1U, 6U, 1U},
      {4U, 2U, 2U},
      {3U, 2U, 3U},
  };
  const size_t vtk_p8_tetra_slots[] = {46U, 47U, 61U, 66U, 130U, 131U, 152U, 164U};
  const uint8_t vtk_p8_tetra_barycentric[][4] = {
      {6U, 1U, 0U, 1U}, {1U, 6U, 0U, 1U}, {4U, 2U, 0U, 2U}, {3U, 2U, 0U, 3U},
      {5U, 1U, 1U, 1U}, {1U, 5U, 1U, 1U}, {3U, 2U, 1U, 2U}, {2U, 2U, 2U, 2U},
  };
  const uint32_t dimensions[] = {2U, 3U};
  const uint32_t orders[] = {1U, 2U, 4U, 8U};
  size_t dimension_index;
  size_t order_index;
  size_t ordering_index;

  for (dimension_index = 0U; dimension_index < 2U; ++dimension_index) {
    for (order_index = 0U; order_index < 4U; ++order_index) {
      const SsFeDescriptor *descriptor = NULL;
      double canonical[SS_FE_MAX_NODES * 3U];
      double external[SS_FE_MAX_NODES * 3U];
      double roundtrip[SS_FE_MAX_NODES * 3U];
      size_t value;
      CHECK(ss_fe_descriptor_find(dimensions[dimension_index], orders[order_index],
                                  &descriptor) == SS_OK);
      for (value = 0U; value < descriptor->node_count * 3U; ++value) {
        canonical[value] =
            sin(0.37 * (double)(value + 1U)) + 0.01 * (double)(value * value);
      }
      for (ordering_index = 0U;
           ordering_index < sizeof(orderings) / sizeof(orderings[0]);
           ++ordering_index) {
        const SsFePermutation *permutation = NULL;
        CHECK(ss_fe_permutation_find(descriptor, orderings[ordering_index],
                                     &permutation) == SS_OK);
        CHECK(permutation != NULL && permutation->node_count == descriptor->node_count);
        CHECK(permutation->canonical_to_external != NULL);
        CHECK(permutation->external_to_canonical != NULL);
        CHECK(permutation->source_version != NULL &&
              permutation->node_coordinate_digest != NULL);
        CHECK(ss_fe_permute_to_external(permutation, 3U, canonical, external) == SS_OK);
        CHECK(ss_fe_permute_to_canonical(permutation, 3U, external, roundtrip) ==
              SS_OK);
        for (value = 0U; value < descriptor->node_count * 3U; ++value) {
          CHECK(roundtrip[value] == canonical[value]);
        }
        if (dimensions[dimension_index] == 2U && orders[order_index] == 4U &&
            orderings[ordering_index] == SS_FE_ORDER_VTK) {
          for (value = 0U; value < descriptor->node_count; ++value) {
            CHECK(permutation->external_to_canonical[value] ==
                  vtk_p4_triangle_expected[value]);
          }
        }
        if (dimensions[dimension_index] == 3U && orders[order_index] == 4U &&
            orderings[ordering_index] == SS_FE_ORDER_GMSH) {
          for (value = 0U; value < descriptor->node_count; ++value) {
            CHECK(permutation->external_to_canonical[value] ==
                  gmsh_p4_tetra_expected[value]);
          }
        }
        if (dimensions[dimension_index] == 3U && orders[order_index] == 4U &&
            orderings[ordering_index] == SS_FE_ORDER_VTK) {
          for (value = 0U; value < descriptor->node_count; ++value) {
            CHECK(permutation->external_to_canonical[value] ==
                  vtk_p4_tetra_expected[value]);
          }
        }
        if (orders[order_index] == 8U && orderings[ordering_index] == SS_FE_ORDER_VTK) {
          const size_t *slots = dimensions[dimension_index] == 2U
                                    ? vtk_p8_triangle_slots
                                    : vtk_p8_tetra_slots;
          const size_t sample_count =
              dimensions[dimension_index] == 2U
                  ? sizeof(vtk_p8_triangle_slots) / sizeof(vtk_p8_triangle_slots[0])
                  : sizeof(vtk_p8_tetra_slots) / sizeof(vtk_p8_tetra_slots[0]);
          size_t sample;
          for (sample = 0U; sample < sample_count; ++sample) {
            const size_t canonical_slot =
                permutation->external_to_canonical[slots[sample]];
            uint32_t barycentric;
            for (barycentric = 0U; barycentric <= dimensions[dimension_index];
                 ++barycentric) {
              const uint8_t expected =
                  dimensions[dimension_index] == 2U
                      ? vtk_p8_triangle_barycentric[sample][barycentric]
                      : vtk_p8_tetra_barycentric[sample][barycentric];
              CHECK(descriptor
                        ->barycentric_indices[canonical_slot *
                                                  (dimensions[dimension_index] + 1U) +
                                              barycentric] == expected);
            }
          }
        }
      }
    }
  }
  {
    const SsFeDescriptor *triangle = NULL;
    const SsFePermutation *legacy = NULL;
    const SsFePermutation *vtk = NULL;
    size_t vtk_slot;
    CHECK(ss_fe_descriptor_find(2U, 4U, &triangle) == SS_OK);
    CHECK(ss_fe_permutation_find(triangle, SS_FE_ORDER_REFERENCE, &legacy) == SS_OK);
    CHECK(ss_fe_permutation_find(triangle, SS_FE_ORDER_VTK, &vtk) == SS_OK);
    for (vtk_slot = 0U; vtk_slot < triangle->node_count; ++vtk_slot) {
      const size_t canonical = vtk->external_to_canonical[vtk_slot];
      CHECK(legacy->canonical_to_external[canonical] ==
            legacy_p4_triangle_to_vtk_expected[vtk_slot]);
    }
    CHECK(legacy_p4_triangle_to_vtk_expected[3] != vtk_p4_triangle_expected[3]);
  }
  return 0;
}

static double polynomial_value(uint32_t dimension, uint32_t degree, const double *xi) {
  double value = 1.25 - 0.7 * xi[0] + 0.35 * xi[1];
  if (dimension == 3U) {
    value += 0.55 * xi[2];
  }
  if (degree >= 2U) {
    value += 0.4 * xi[0] * xi[0] - 0.2 * xi[0] * xi[1] + 0.3 * xi[1] * xi[1];
    if (dimension == 3U) {
      value += 0.15 * xi[2] * xi[2] - 0.25 * xi[0] * xi[2];
    }
  }
  return value;
}

static int check_transfer(uint32_t dimension, uint32_t coarse_order,
                          uint32_t fine_order) {
  const SsFeDescriptor *coarse = NULL;
  const SsFeDescriptor *fine = NULL;
  double matrix[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
  double coarse_values[SS_FE_MAX_NODES] = {0.0};
  double minimum = 1.0;
  size_t rows = 0U;
  size_t columns = 0U;
  size_t row;
  size_t column;

  CHECK(ss_fe_descriptor_find(dimension, coarse_order, &coarse) == SS_OK);
  CHECK(ss_fe_descriptor_find(dimension, fine_order, &fine) == SS_OK);
  CHECK(ss_fe_transfer_build(coarse, fine, matrix, SS_FE_MAX_NODES * SS_FE_MAX_NODES,
                             &rows, &columns) == SS_OK);
  CHECK(rows == fine->node_count && columns == coarse->node_count);
  for (column = 0U; column < columns; ++column) {
    double xi[3] = {
        node_coordinate(coarse, column, 0U),
        node_coordinate(coarse, column, 1U),
        dimension == 3U ? node_coordinate(coarse, column, 2U) : 0.0,
    };
    coarse_values[column] = polynomial_value(dimension, coarse_order, xi);
  }
  for (row = 0U; row < rows; ++row) {
    double actual = 0.0;
    double row_sum = 0.0;
    const double xi[3] = {
        node_coordinate(fine, row, 0U),
        node_coordinate(fine, row, 1U),
        dimension == 3U ? node_coordinate(fine, row, 2U) : 0.0,
    };
    for (column = 0U; column < columns; ++column) {
      const double entry = matrix[row * columns + column];
      actual += entry * coarse_values[column];
      row_sum += entry;
      if (entry < minimum) {
        minimum = entry;
      }
    }
    CHECK_NEAR(row_sum, 1.0, 3.0e-12);
    CHECK_NEAR(actual, polynomial_value(dimension, coarse_order, xi), 3.0e-12);
  }
  if (coarse_order == 2U && fine_order == 4U) {
    CHECK_NEAR(minimum, -0.125, 3.0e-12);
  }
  return 0;
}

static int test_transfer(void) {
  const uint32_t dimensions[] = {2U, 3U};
  size_t dimension_index;
  for (dimension_index = 0U; dimension_index < 2U; ++dimension_index) {
    const uint32_t dimension = dimensions[dimension_index];
    const SsFeDescriptor *p1 = NULL;
    const SsFeDescriptor *p2 = NULL;
    const SsFeDescriptor *p4 = NULL;
    double p12[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
    double p24[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
    double p14[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
    size_t r12 = 0U;
    size_t c12 = 0U;
    size_t r24 = 0U;
    size_t c24 = 0U;
    size_t r14 = 0U;
    size_t c14 = 0U;
    size_t row;
    size_t column;
    size_t middle;

    CHECK(check_transfer(dimension, 1U, 2U) == 0);
    CHECK(check_transfer(dimension, 2U, 4U) == 0);
    CHECK(check_transfer(dimension, 1U, 4U) == 0);
    CHECK(check_transfer(dimension, 4U, 8U) == 0);
    CHECK(check_transfer(dimension, 2U, 8U) == 0);
    CHECK(check_transfer(dimension, 1U, 8U) == 0);
    CHECK(ss_fe_descriptor_find(dimension, 1U, &p1) == SS_OK);
    CHECK(ss_fe_descriptor_find(dimension, 2U, &p2) == SS_OK);
    CHECK(ss_fe_descriptor_find(dimension, 4U, &p4) == SS_OK);
    CHECK(ss_fe_transfer_build(p1, p2, p12, SS_FE_MAX_NODES * SS_FE_MAX_NODES, &r12,
                               &c12) == SS_OK);
    CHECK(ss_fe_transfer_build(p2, p4, p24, SS_FE_MAX_NODES * SS_FE_MAX_NODES, &r24,
                               &c24) == SS_OK);
    CHECK(ss_fe_transfer_build(p1, p4, p14, SS_FE_MAX_NODES * SS_FE_MAX_NODES, &r14,
                               &c14) == SS_OK);
    CHECK(c24 == r12 && r24 == r14 && c12 == c14);
    for (row = 0U; row < r14; ++row) {
      for (column = 0U; column < c14; ++column) {
        double composed = 0.0;
        for (middle = 0U; middle < c24; ++middle) {
          composed += p24[row * c24 + middle] * p12[middle * c12 + column];
        }
        CHECK_NEAR(composed, p14[row * c14 + column], 3.0e-12);
      }
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fputs("usage: ss_test_fe TEST-ID\n", stderr);
    return 2;
  }
  if (strcmp(argv[1], "FE-UNIT-BASIS-001") == 0) {
    return test_basis();
  }
  if (strcmp(argv[1], "FE-REG-P4-DERIV-001") == 0) {
    return test_p4_derivative();
  }
  if (strcmp(argv[1], "FE-UNIT-QUADRATURE-001") == 0) {
    return test_quadrature();
  }
  if (strcmp(argv[1], "FE-NEG-QUADRATURE-001") == 0) {
    return test_bad_quadrature();
  }
  if (strcmp(argv[1], "FE-UNIT-CURVED-001") == 0) {
    return test_curved();
  }
  if (strcmp(argv[1], "FE-UNIT-FACET-001") == 0) {
    return test_facet();
  }
  if (strcmp(argv[1], "FE-UNIT-PERMUTE-001") == 0) {
    return test_permutations();
  }
  if (strcmp(argv[1], "FE-UNIT-TRANSFER-001") == 0) {
    return test_transfer();
  }
  fprintf(stderr, "unknown FE test: %s\n", argv[1]);
  return 2;
}
