/* Independent oracles for finite-element, hydraulic and constitutive kernels.
 *
 * Each test compares a production kernel against a reference that does not
 * share its approximation: exact polynomial integration through a registered
 * high-degree rule, analytic geometry, positive homogeneity, rotation
 * covariance and central differences.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/assembly/assembly.h"
#include "slope_internal/fe/fe.h"
#include "slope_internal/hydraulic/hydraulic.h"
#include "slope_internal/material/material.h"

#include <float.h>
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
      fprintf(stderr, "%s:%d: |%.17g - %.17g| = %.3g > %.3g\n", __FILE__, __LINE__,    \
              check_actual, check_expected, fabs(check_actual - check_expected),       \
              (double)(tolerance));                                                    \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

typedef void (*OracleMap)(uint32_t dimension, const double *reference, double *physical,
                          double *jacobian);

/* x_i = xi_i + xi_i^4 / 2: positive Jacobian, determinant of degree 3d. */
static void quartic_map(uint32_t dimension, const double *reference, double *physical,
                        double *jacobian) {
  uint32_t row;
  for (row = 0U; row < dimension; ++row) {
    const double xi = reference[row];
    uint32_t column;
    physical[row] = xi + 0.5 * xi * xi * xi * xi;
    for (column = 0U; column < dimension; ++column) {
      jacobian[row * dimension + column] =
          row == column ? 1.0 + 2.0 * xi * xi * xi : 0.0;
    }
  }
}

/* Determinant-one shear (r, s - 2 r^2) or (r, s, t - 2 r^2); exact in G2/G4. */
static void shear_map(uint32_t dimension, const double *reference, double *physical,
                      double *jacobian) {
  const uint32_t last = dimension - 1U;
  uint32_t row;
  for (row = 0U; row < dimension; ++row) {
    uint32_t column;
    physical[row] = reference[row];
    for (column = 0U; column < dimension; ++column) {
      jacobian[row * dimension + column] = row == column ? 1.0 : 0.0;
    }
  }
  physical[last] -= 2.0 * reference[0] * reference[0];
  jacobian[last * dimension] = -4.0 * reference[0];
}

static void identity_map(uint32_t dimension, const double *reference, double *physical,
                         double *jacobian) {
  uint32_t row;
  for (row = 0U; row < dimension; ++row) {
    uint32_t column;
    physical[row] = reference[row];
    for (column = 0U; column < dimension; ++column) {
      jacobian[row * dimension + column] = row == column ? 1.0 : 0.0;
    }
  }
}

static double node_coordinate(const SsFeDescriptor *descriptor, size_t node,
                              uint32_t axis) {
  return (double)descriptor
             ->barycentric_indices[node * (descriptor->dimension + 1U) + axis + 1U] /
         (double)descriptor->order;
}

/* Interpolate the map at the coordinate nodes; exact when the map lies in the
 * coordinate space.  An affine scale/translation follows the map. */
static void fill_nodes(const SsFeDescriptor *descriptor, OracleMap map, double scale,
                       const double *translation, double *coordinates) {
  size_t node;
  for (node = 0U; node < descriptor->node_count; ++node) {
    double reference[3] = {0.0, 0.0, 0.0};
    double physical[3] = {0.0, 0.0, 0.0};
    double jacobian[9];
    uint32_t axis;
    for (axis = 0U; axis < descriptor->dimension; ++axis) {
      reference[axis] = node_coordinate(descriptor, node, axis);
    }
    map(descriptor->dimension, reference, physical, jacobian);
    for (axis = 0U; axis < descriptor->dimension; ++axis) {
      coordinates[node * descriptor->dimension + axis] =
          scale * physical[axis] + (translation == NULL ? 0.0 : translation[axis]);
    }
  }
}

static const SsFeQuadrature *reference_rule(uint32_t dimension,
                                            SsFeQuadratureDomain domain) {
  const SsFeQuadrature *rule = NULL;
  const char *id = domain == SS_FE_QUADRATURE_VOLUME
                       ? (dimension == 2U ? "triangle-degree-19-petsc-minsym-v1"
                                          : "tetrahedron-degree-19-petsc-minsym-v1")
                       : (dimension == 2U ? "line-degree-15-v1"
                                          : "triangle-facet-degree-19-petsc-minsym-v1");
  return ss_fe_quadrature_find(id, &rule) == SS_OK ? rule : NULL;
}

static const SsFeQuadrature *planned_rule(uint32_t dimension,
                                          SsFeOperationKind operation,
                                          uint32_t field_order,
                                          uint32_t coordinate_order) {
  SsFeQuadratureRequest request = {dimension,
                                   SS_FE_QUADRATURE_VOLUME,
                                   operation,
                                   field_order,
                                   coordinate_order,
                                   0U,
                                   SS_FE_QUADRATURE_MODE_DEFAULT,
                                   NULL};
  SsFeQuadraturePlan plan = {0};
  return ss_fe_quadrature_plan(&request, &plan) == SS_OK ? plan.selected_rule : NULL;
}

/* Body moments int N_a det(J) and the mass trace int sum_a N_a^2 det(J). */
static int moments(const SsFeDescriptor *field, const SsFeDescriptor *coordinates_space,
                   const double *coordinates, const SsFeQuadrature *rule, double *body,
                   double *mass_trace) {
  const SsFeGeometryPolicy policy = {1.0e-13, 1.0e12};
  double values[SS_FE_MAX_NODES];
  size_t point;
  size_t node;
  memset(body, 0, field->node_count * sizeof(*body));
  *mass_trace = 0.0;
  for (point = 0U; point < rule->point_count; ++point) {
    const double *xi = rule->points + point * (size_t)rule->dimension;
    SsFeGeometry geometry;
    double weight;
    CHECK(ss_fe_geometry_eval(coordinates_space, coordinates, xi, &policy, &geometry) ==
          SS_OK);
    CHECK(ss_fe_basis_eval_scalar(field, xi, values, NULL) == SS_OK);
    weight = rule->weights[point] * geometry.determinant;
    for (node = 0U; node < field->node_count; ++node) {
      body[node] += weight * values[node];
      *mass_trace += weight * values[node] * values[node];
    }
  }
  return 0;
}

static int check_product_degree_case(uint32_t dimension, uint32_t field_order,
                                     uint32_t coordinate_order) {
  const SsFeDescriptor *field = NULL;
  const SsFeDescriptor *geometry = NULL;
  const SsFeQuadrature *exact = reference_rule(dimension, SS_FE_QUADRATURE_VOLUME);
  const SsFeQuadrature *operator_rule = planned_rule(
      dimension, SS_FE_OPERATION_MECHANICS_STIFFNESS, field_order, coordinate_order);
  const SsFeQuadrature *body_rule = planned_rule(
      dimension, SS_FE_OPERATION_MECHANICS_BODY, field_order, coordinate_order);
  const SsFeQuadrature *mass_rule =
      planned_rule(dimension, SS_FE_OPERATION_MASS, field_order, coordinate_order);
  static double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  static double expected[SS_FE_MAX_NODES];
  static double actual[SS_FE_MAX_NODES];
  double expected_trace = 0.0;
  double actual_trace = 0.0;
  const SsFeQuadrature *rules[2];
  size_t rule_index;
  size_t node;
  CHECK(exact != NULL && operator_rule != NULL && body_rule != NULL &&
        mass_rule != NULL);
  CHECK(ss_fe_descriptor_find(dimension, field_order, &field) == SS_OK);
  CHECK(ss_fe_descriptor_find(dimension, coordinate_order, &geometry) == SS_OK);
  fill_nodes(geometry, quartic_map, 1.0, NULL, coordinates);
  CHECK(moments(field, geometry, coordinates, exact, expected, &expected_trace) == 0);
  /* The body integrand N_a det(J) has degree p + d (g - 1).  The fused
   * mechanics operator rule must integrate the load it shares exactly. */
  rules[0] = body_rule;
  rules[1] = operator_rule;
  for (rule_index = 0U; rule_index < 2U; ++rule_index) {
    CHECK(rules[rule_index]->exact_degree >=
          field_order + dimension * (coordinate_order - 1U));
    CHECK(moments(field, geometry, coordinates, rules[rule_index], actual,
                  &actual_trace) == 0);
    for (node = 0U; node < field->node_count; ++node) {
      CHECK_NEAR(actual[node], expected[node], 1.0e-12);
    }
  }
  CHECK(mass_rule->exact_degree >=
        2U * field_order + dimension * (coordinate_order - 1U));
  CHECK(moments(field, geometry, coordinates, mass_rule, actual, &actual_trace) == 0);
  CHECK_NEAR(actual_trace, expected_trace, 1.0e-12 * fmax(1.0, expected_trace));
  return 0;
}

/* Production element path: zero displacement makes the mechanics residual the
 * negative consistent body load computed with the operator rule. */
static int check_mechanics_body_load(uint32_t dimension, uint32_t order) {
  const SsMcParameterInput material_input = {
      40000.0,
      0.3,
      10.0,
      0.5,
      0.0,
      SS_MC_DAVIS_B,
      17U,
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"};
  const double body_force[3] = {0.25, -20.0, 1.5};
  const SsFeDescriptor *field = NULL;
  const SsFeQuadrature *exact = reference_rule(dimension, SS_FE_QUADRATURE_VOLUME);
  static double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  static double displacement[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  static double residual[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  static double expected[SS_FE_MAX_NODES];
  double trace = 0.0;
  SsMcParameters *material = NULL;
  SsMechanicsElementInput input;
  SsElementOutput output;
  SsErrorRecord error;
  size_t node;
  CHECK(ss_fe_descriptor_find(dimension, order, &field) == SS_OK);
  CHECK(ss_mc_parameters_create(&material_input, &material, NULL) == SS_OK);
  fill_nodes(field, quartic_map, 1.0, NULL, coordinates);
  CHECK(moments(field, field, coordinates, exact, expected, &trace) == 0);
  memset(displacement, 0, sizeof(displacement));
  memset(&input, 0, sizeof(input));
  input.dimension = dimension;
  input.field_descriptor = field;
  input.coordinate_descriptor = field;
  input.quadrature =
      planned_rule(dimension, SS_FE_OPERATION_MECHANICS_STIFFNESS, order, order);
  input.coordinate_values = coordinates;
  input.field_values = displacement;
  input.body_force = body_force;
  input.material = material;
  input.reduction_factor = 1.0;
  input.external_load_factor = 1.0;
  input.stable_cell_id = 9U;
  input.point_force_mode = SS_MECHANICS_POINT_FORCE_FIXED;
  memset(&output, 0, sizeof(output));
  output.closure_capacity = field->node_count * dimension;
  output.residual = residual;
  CHECK(input.quadrature != NULL);
  CHECK(ss_assembly_mechanics_element(&input, SS_ASSEMBLY_RESIDUAL, &output, &error) ==
        SS_OK);
  for (node = 0U; node < field->node_count; ++node) {
    uint32_t component;
    for (component = 0U; component < dimension; ++component) {
      CHECK_NEAR(residual[node * dimension + component],
                 -expected[node] * body_force[component],
                 1.0e-12 * fabs(body_force[component]));
    }
  }
  CHECK(ss_mc_parameters_destroy(&material) == SS_OK);
  return 0;
}

static int test_product_degree(void) {
  static const uint32_t pairs[][2] = {{2U, 2U}, {4U, 2U}, {4U, 4U}};
  uint32_t dimension;
  size_t pair;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    for (pair = 0U; pair < sizeof(pairs) / sizeof(pairs[0]); ++pair) {
      if (check_product_degree_case(dimension, pairs[pair][0], pairs[pair][1]) != 0) {
        fprintf(stderr, "product-degree case d=%u P%u/G%u failed\n", dimension,
                pairs[pair][0], pairs[pair][1]);
        return 1;
      }
    }
    CHECK(check_mechanics_body_load(dimension, 2U) == 0);
    CHECK(check_mechanics_body_load(dimension, 4U) == 0);
  }
  return 0;
}

/* Seepage element on the unit simplex with a quartic pressure entirely inside
 * the ramp transition: the gravity transport term grad(N_i) p_h has degree
 * 2p - 1 and its Jacobian grad(N_i) N_j has the same degree. */
static int hydraulic_element(const SsFeDescriptor *field, const double *coordinates,
                             const double *pressure, const SsFeQuadrature *rule,
                             const SsHydraulicPointModel *model, double *residual,
                             double *jacobian) {
  SsScalarElementInput input;
  SsElementOutput output;
  SsErrorRecord error;
  memset(&input, 0, sizeof(input));
  input.dimension = field->dimension;
  input.field_descriptor = field;
  input.coordinate_descriptor = field;
  input.quadrature = rule;
  input.coordinate_values = coordinates;
  input.field_values = pressure;
  input.point_kernel = ss_hydraulic_point_kernel;
  input.point_context = model;
  input.stable_cell_id = 3U;
  memset(&output, 0, sizeof(output));
  output.closure_capacity = field->node_count;
  output.matrix_capacity = field->node_count * field->node_count;
  output.residual = residual;
  output.jacobian = jacobian;
  CHECK(ss_assembly_scalar_element(&input, SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN,
                                   &output, &error) == SS_OK);
  return 0;
}

static int check_hydraulic_gravity(uint32_t dimension, uint32_t order) {
  const SsHydraulicPointModel model = {dimension, 1U, 9.81, 1.0, 4.905};
  const SsFeDescriptor *field = NULL;
  const SsFeQuadrature *exact = reference_rule(dimension, SS_FE_QUADRATURE_VOLUME);
  const SsFeQuadrature *rule =
      planned_rule(dimension, SS_FE_OPERATION_HYDRAULIC_DIFFUSION, order, 1U);
  static double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  static double pressure[SS_FE_MAX_NODES];
  static double expected_residual[SS_FE_MAX_NODES];
  static double actual_residual[SS_FE_MAX_NODES];
  static double expected_jacobian[SS_FE_MAX_NODES * SS_FE_MAX_NODES];
  static double actual_jacobian[SS_FE_MAX_NODES * SS_FE_MAX_NODES];
  double scale = 0.0;
  size_t node;
  CHECK(exact != NULL && rule != NULL);
  CHECK(rule->exact_degree >= 2U * order - 1U);
  CHECK(ss_fe_descriptor_find(dimension, order, &field) == SS_OK);
  fill_nodes(field, identity_map, 1.0, NULL, coordinates);
  for (node = 0U; node < field->node_count; ++node) {
    const double r = node_coordinate(field, node, 0U);
    const double polynomial = order >= 4U ? r * r * r * r : r * r;
    pressure[node] = model.epsilon * (0.5 + 0.2 * polynomial);
  }
  CHECK(hydraulic_element(field, coordinates, pressure, exact, &model,
                          expected_residual, expected_jacobian) == 0);
  CHECK(hydraulic_element(field, coordinates, pressure, rule, &model, actual_residual,
                          actual_jacobian) == 0);
  for (node = 0U; node < field->node_count; ++node) {
    scale = fmax(scale, fabs(expected_residual[node]));
  }
  for (node = 0U; node < field->node_count; ++node) {
    size_t column;
    CHECK_NEAR(actual_residual[node], expected_residual[node], 1.0e-12 * scale);
    for (column = 0U; column < field->node_count; ++column) {
      const size_t entry = node * field->node_count + column;
      CHECK_NEAR(actual_jacobian[entry], expected_jacobian[entry], 1.0e-11);
    }
  }
  return 0;
}

static int test_hydraulic_gravity_degree(void) {
  uint32_t dimension;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    CHECK(check_hydraulic_gravity(dimension, 2U) == 0);
    CHECK(check_hydraulic_gravity(dimension, 4U) == 0);
  }
  return 0;
}

static const uint32_t triangle_facets[3][2] = {{0U, 1U}, {1U, 2U}, {0U, 2U}};
static const uint32_t tetrahedron_facets[4][3] = {
    {0U, 1U, 2U}, {0U, 1U, 3U}, {0U, 2U, 3U}, {1U, 2U, 3U}};

/* Outward unit normal of the reference simplex facet, independent of the
 * vertex ordering used to parameterize it. */
static void reference_outward_normal(uint32_t dimension, uint32_t facet,
                                     double *normal) {
  uint32_t vertex_sum = 0U;
  uint32_t axis;
  uint32_t local;
  const uint32_t count = dimension;
  memset(normal, 0, 3U * sizeof(*normal));
  for (local = 0U; local < count; ++local) {
    vertex_sum |= 1U << (dimension == 2U ? triangle_facets[facet][local]
                                         : tetrahedron_facets[facet][local]);
  }
  if ((vertex_sum & 1U) == 0U) {
    for (axis = 0U; axis < dimension; ++axis) {
      normal[axis] = 1.0 / sqrt((double)dimension);
    }
    return;
  }
  for (axis = 0U; axis < dimension; ++axis) {
    if ((vertex_sum & (1U << (axis + 1U))) == 0U) {
      normal[axis] = -1.0;
    }
  }
}

static void inverse_transpose(uint32_t dimension, const double *jacobian,
                              double *result) {
  if (dimension == 2U) {
    const double determinant = jacobian[0] * jacobian[3] - jacobian[1] * jacobian[2];
    result[0] = jacobian[3] / determinant;
    result[1] = -jacobian[2] / determinant;
    result[2] = -jacobian[1] / determinant;
    result[3] = jacobian[0] / determinant;
  } else {
    const double *a = jacobian;
    const double cofactor[9] = {a[4] * a[8] - a[5] * a[7], a[5] * a[6] - a[3] * a[8],
                                a[3] * a[7] - a[4] * a[6], a[2] * a[7] - a[1] * a[8],
                                a[0] * a[8] - a[2] * a[6], a[1] * a[6] - a[0] * a[7],
                                a[1] * a[5] - a[2] * a[4], a[2] * a[3] - a[0] * a[5],
                                a[0] * a[4] - a[1] * a[3]};
    const double determinant =
        a[0] * cofactor[0] + a[1] * cofactor[1] + a[2] * cofactor[2];
    size_t entry;
    for (entry = 0U; entry < 9U; ++entry) {
      result[entry] = cofactor[entry] / determinant;
    }
  }
}

static int check_facet_orientation(uint32_t dimension, uint32_t order, OracleMap map,
                                   double scale, const double *translation) {
  const SsFeDescriptor *geometry = NULL;
  const SsFeQuadrature *facet_rule = reference_rule(dimension, SS_FE_QUADRATURE_FACET);
  const SsFeQuadrature *volume_rule =
      reference_rule(dimension, SS_FE_QUADRATURE_VOLUME);
  const SsFeGeometryPolicy policy = {1.0e-13, 1.0e12};
  const double reference_centroid[3] = {
      1.0 / (dimension + 1.0), 1.0 / (dimension + 1.0), 1.0 / (dimension + 1.0)};
  double centroid[3] = {0.0, 0.0, 0.0};
  static double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  double closed_normal[3] = {0.0, 0.0, 0.0};
  double flux_of_position = 0.0;
  double volume = 0.0;
  uint32_t facet;
  size_t point;
  CHECK(facet_rule != NULL && volume_rule != NULL);
  CHECK(ss_fe_descriptor_find(dimension, order, &geometry) == SS_OK);
  fill_nodes(geometry, map, scale, translation, coordinates);
  /* Production passes the mapped reference centroid of the parent cell. */
  {
    double jacobian[9];
    uint32_t axis;
    map(dimension, reference_centroid, centroid, jacobian);
    for (axis = 0U; axis < dimension; ++axis) {
      centroid[axis] =
          scale * centroid[axis] + (translation == NULL ? 0.0 : translation[axis]);
    }
  }
  for (facet = 0U; facet < dimension + 1U; ++facet) {
    for (point = 0U; point < facet_rule->point_count; ++point) {
      const double *eta = facet_rule->points + point * (size_t)facet_rule->dimension;
      double parent[3] = {0.0, 0.0, 0.0};
      double physical[3] = {0.0, 0.0, 0.0};
      double jacobian[9];
      double transform[9];
      double reference_normal[3];
      double exact[3] = {0.0, 0.0, 0.0};
      double norm = 0.0;
      SsFeFacetGeometry sample;
      uint32_t row;
      CHECK(ss_fe_facet_parent_reference(dimension, facet, eta, parent) == SS_OK);
      map(dimension, parent, physical, jacobian);
      inverse_transpose(dimension, jacobian, transform);
      reference_outward_normal(dimension, facet, reference_normal);
      for (row = 0U; row < dimension; ++row) {
        uint32_t column;
        for (column = 0U; column < dimension; ++column) {
          exact[row] += transform[row * dimension + column] * reference_normal[column];
        }
        norm += exact[row] * exact[row];
      }
      norm = sqrt(norm);
      CHECK(ss_fe_facet_geometry_eval(geometry, facet, coordinates, eta, centroid,
                                      &sample) == SS_OK);
      for (row = 0U; row < dimension; ++row) {
        if (fabs(sample.outward_normal[row] - exact[row] / norm) > 1.0e-12) {
          fprintf(stderr, "d=%u G%u facet %u point %zu: normal[%u] %.6f != %.6f\n",
                  dimension, order, facet, point, row, sample.outward_normal[row],
                  exact[row] / norm);
        }
        CHECK_NEAR(sample.outward_normal[row], exact[row] / norm, 1.0e-12);
        CHECK_NEAR(sample.physical[row],
                   scale * physical[row] +
                       (translation == NULL ? 0.0 : translation[row]),
                   1.0e-12 * fmax(1.0, scale));
        closed_normal[row] +=
            facet_rule->weights[point] * sample.measure * sample.outward_normal[row];
        flux_of_position += facet_rule->weights[point] * sample.measure *
                            sample.outward_normal[row] * sample.physical[row];
      }
    }
  }
  for (point = 0U; point < volume_rule->point_count; ++point) {
    SsFeGeometry sample;
    CHECK(ss_fe_geometry_eval(geometry, coordinates,
                              volume_rule->points + point * (size_t)dimension, &policy,
                              &sample) == SS_OK);
    volume += volume_rule->weights[point] * sample.determinant;
  }
  /* Divergence theorem: int n = 0 and int x.n = d |Omega| on closed cells. */
  for (facet = 0U; facet < dimension; ++facet) {
    CHECK_NEAR(closed_normal[facet], 0.0, 1.0e-12 * scale * scale);
  }
  CHECK_NEAR(flux_of_position, (double)dimension * volume, 1.0e-11 * fmax(1.0, volume));
  return 0;
}

static int test_facet_orientation(void) {
  const double translation[3] = {5.0, -2.0, 0.75};
  uint32_t dimension;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    CHECK(check_facet_orientation(dimension, 2U, shear_map, 1.0, NULL) == 0);
    CHECK(check_facet_orientation(dimension, 4U, shear_map, 1.0, NULL) == 0);
    CHECK(check_facet_orientation(dimension, 4U, shear_map, 3.0, translation) == 0);
    CHECK(check_facet_orientation(dimension, 4U, quartic_map, 1.0, NULL) == 0);
    CHECK(check_facet_orientation(dimension, 1U, identity_map, 2.0, translation) == 0);
  }
  return 0;
}

static void engineering_to_tensor(const double *strain, double *tensor) {
  tensor[0] = strain[0];
  tensor[4] = strain[1];
  tensor[8] = strain[2];
  tensor[1] = tensor[3] = 0.5 * strain[3];
  tensor[5] = tensor[7] = 0.5 * strain[4];
  tensor[2] = tensor[6] = 0.5 * strain[5];
}

static void tensor_to_engineering(const double *tensor, double *strain) {
  strain[0] = tensor[0];
  strain[1] = tensor[4];
  strain[2] = tensor[8];
  strain[3] = tensor[1] + tensor[3];
  strain[4] = tensor[5] + tensor[7];
  strain[5] = tensor[2] + tensor[6];
}

static void tensor_to_stress(const double *tensor, double *stress) {
  stress[0] = tensor[0];
  stress[1] = tensor[4];
  stress[2] = tensor[8];
  stress[3] = 0.5 * (tensor[1] + tensor[3]);
  stress[4] = 0.5 * (tensor[5] + tensor[7]);
  stress[5] = 0.5 * (tensor[2] + tensor[6]);
}

static void stress_to_tensor(const double *stress, double *tensor) {
  tensor[0] = stress[0];
  tensor[4] = stress[1];
  tensor[8] = stress[2];
  tensor[1] = tensor[3] = stress[3];
  tensor[5] = tensor[7] = stress[4];
  tensor[2] = tensor[6] = stress[5];
}

/* result = R A R^T */
static void rotate_tensor(const double *rotation, const double *tensor,
                          double *result) {
  double work[9] = {0.0};
  size_t row;
  memset(result, 0, 9U * sizeof(*result));
  for (row = 0U; row < 3U; ++row) {
    size_t column;
    for (column = 0U; column < 3U; ++column) {
      size_t inner;
      for (inner = 0U; inner < 3U; ++inner) {
        work[row * 3U + column] +=
            rotation[row * 3U + inner] * tensor[inner * 3U + column];
      }
    }
  }
  for (row = 0U; row < 3U; ++row) {
    size_t column;
    for (column = 0U; column < 3U; ++column) {
      size_t inner;
      for (inner = 0U; inner < 3U; ++inner) {
        result[row * 3U + column] +=
            work[row * 3U + inner] * rotation[column * 3U + inner];
      }
    }
  }
}

static void tangent_action(const double *tangent, const double *direction,
                           double *result) {
  size_t row;
  for (row = 0U; row < 6U; ++row) {
    size_t column;
    result[row] = 0.0;
    for (column = 0U; column < 6U; ++column) {
      result[row] += tangent[row + 6U * column] * direction[column];
    }
  }
}

static double relative_difference(const double *actual, const double *expected,
                                  size_t count) {
  double difference = 0.0;
  double norm = 0.0;
  size_t index;
  for (index = 0U; index < count; ++index) {
    difference += (actual[index] - expected[index]) * (actual[index] - expected[index]);
    norm += expected[index] * expected[index];
  }
  return sqrt(difference) / fmax(sqrt(norm), DBL_MIN);
}

static int evaluate_material(const SsMcParameters *material, const double *strain,
                             SsMcResponse *response) {
  CHECK(ss_mc_evaluate(material, SS_MC_DIMENSION_3, 1.0, strain, response, NULL) ==
        SS_OK);
  return 0;
}

static int check_tangent_scaling(double friction_degrees, double cohesion,
                                 const double *direction) {
  const double pi = 3.14159265358979323846;
  const SsMcParameterInput input = {
      100000.0,
      0.3,
      cohesion,
      friction_degrees * pi / 180.0,
      0.0,
      SS_MC_DAVIS_B,
      17U,
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"};
  /* A generic rotation (unit quaternion (0.8, 0.2, 0.4, 0.4)). */
  const double rotation[9] = {0.36, -0.48, 0.80, 0.80, 0.60, 0.00, -0.48, 0.64, 0.60};
  static const double scales[] = {1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15, 1.0e-20};
  const double probe[6] = {0.3, -0.7, 0.2, 0.5, -0.4, 0.9};
  SsMcParameters *material = NULL;
  SsMcResponse anchor;
  size_t scale_index;
  CHECK(ss_mc_parameters_create(&input, &material, NULL) == SS_OK);
  {
    double strain[6];
    size_t component;
    for (component = 0U; component < 6U; ++component) {
      strain[component] = 1.0e-3 * direction[component];
    }
    CHECK(evaluate_material(material, strain, &anchor) == 0);
    CHECK(anchor.branch != SS_MC_BRANCH_ELASTIC);
  }
  for (scale_index = 0U; scale_index < sizeof(scales) / sizeof(scales[0]);
       ++scale_index) {
    const double scale = scales[scale_index];
    double strain[6];
    double expected_stress[6];
    double rotated_strain_tensor[9];
    double strain_tensor[9];
    double rotated_strain[6];
    double probe_tensor[9];
    double rotated_probe_tensor[9];
    double rotated_probe[6];
    double action[6];
    double action_tensor[9];
    double rotated_action_tensor[9];
    double expected_rotated_action[6];
    double rotated_action[6];
    double central[6];
    SsMcResponse response;
    SsMcResponse rotated;
    size_t component;
    for (component = 0U; component < 6U; ++component) {
      strain[component] = scale * direction[component];
      expected_stress[component] = anchor.stress[component] * (scale / 1.0e-3);
    }
    CHECK(evaluate_material(material, strain, &response) == 0);
    if (cohesion == 0.0) {
      /* Positive homogeneity: same branch, scaled stress, unchanged tangent. */
      if (response.branch != anchor.branch) {
        fprintf(stderr, "branch %d != %d at scale %.1e\n", (int)response.branch,
                (int)anchor.branch, scale);
      }
      CHECK(response.branch == anchor.branch);
      CHECK(relative_difference(response.stress, expected_stress, 6U) < 1.0e-10);
      if (relative_difference(response.tangent, anchor.tangent, 36U) >= 1.0e-8) {
        fprintf(stderr, "tangent homogeneity lost at scale %.1e: %.3e\n", scale,
                relative_difference(response.tangent, anchor.tangent, 36U));
        CHECK(0);
      }
    }
    /* Rotation covariance of the tangent action. */
    engineering_to_tensor(strain, strain_tensor);
    rotate_tensor(rotation, strain_tensor, rotated_strain_tensor);
    tensor_to_engineering(rotated_strain_tensor, rotated_strain);
    engineering_to_tensor(probe, probe_tensor);
    rotate_tensor(rotation, probe_tensor, rotated_probe_tensor);
    tensor_to_engineering(rotated_probe_tensor, rotated_probe);
    CHECK(evaluate_material(material, rotated_strain, &rotated) == 0);
    CHECK(rotated.branch == response.branch);
    tangent_action(response.tangent, probe, action);
    stress_to_tensor(action, action_tensor);
    rotate_tensor(rotation, action_tensor, rotated_action_tensor);
    tensor_to_stress(rotated_action_tensor, expected_rotated_action);
    tangent_action(rotated.tangent, rotated_probe, rotated_action);
    CHECK(relative_difference(rotated_action, expected_rotated_action, 6U) < 1.0e-8);
    /* Central difference with a step relative to the strain scale. */
    {
      const double step = 1.0e-6 * scale;
      double plus_strain[6];
      double minus_strain[6];
      SsMcResponse plus;
      SsMcResponse minus;
      for (component = 0U; component < 6U; ++component) {
        plus_strain[component] = strain[component] + step * probe[component];
        minus_strain[component] = strain[component] - step * probe[component];
      }
      CHECK(evaluate_material(material, plus_strain, &plus) == 0);
      CHECK(evaluate_material(material, minus_strain, &minus) == 0);
      CHECK(plus.branch == response.branch && minus.branch == response.branch);
      for (component = 0U; component < 6U; ++component) {
        central[component] =
            (plus.stress[component] - minus.stress[component]) / (2.0 * step);
      }
      if (relative_difference(action, central, 6U) >= 1.0e-5) {
        fprintf(stderr, "central difference mismatch at scale %.1e: %.3e\n", scale,
                relative_difference(action, central, 6U));
        CHECK(0);
      }
    }
  }
  CHECK(ss_mc_parameters_destroy(&material) == SS_OK);
  return 0;
}

static int test_tangent_scaling(void) {
  /* Distinct eigenvalues on the left edge, an edge with a near pair, the
   * smooth face and the right edge of the zero-cohesion cone. */
  static const double directions[][6] = {
      {3.0, 1.0, -4.0, 0.0, 0.0, 0.0},
      {1.0, 0.2, -2.5, 0.4, -0.3, 0.2},
      {2.0, -0.5, -3.0, 0.0, 0.0, 0.0},
      {4.0, -1.0, -1.2, 0.0, 0.0, 0.0},
  };
  size_t direction;
  for (direction = 0U; direction < sizeof(directions) / sizeof(directions[0]);
       ++direction) {
    if (check_tangent_scaling(30.0, 0.0, directions[direction]) != 0) {
      fprintf(stderr, "tangent scaling direction %zu failed\n", direction);
      return 1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s TEST-ID\n", argv[0]);
    return 2;
  }
  if (strcmp(argv[1], "FE-ORACLE-PRODUCT-DEGREE-001") == 0) {
    return test_product_degree();
  }
  if (strcmp(argv[1], "HYD-ORACLE-GRAVITY-DEGREE-001") == 0) {
    return test_hydraulic_gravity_degree();
  }
  if (strcmp(argv[1], "FE-ORACLE-FACET-NORMAL-001") == 0) {
    return test_facet_orientation();
  }
  if (strcmp(argv[1], "MAT-ORACLE-TANGENT-SCALE-001") == 0) {
    return test_tangent_scaling();
  }
  fprintf(stderr, "unknown test %s\n", argv[1]);
  return 2;
}
