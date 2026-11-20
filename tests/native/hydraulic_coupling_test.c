/* Single-process seepage tests (ramps, elements, Jacobian, head, flux) and
 * one-way coupling into mechanics.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/assembly/assembly.h"
#include "slope_internal/build/config.h"
#include "slope_internal/coupling/coupling.h"
#include "slope_internal/fe/fe.h"
#include "slope_internal/hydraulic/hydraulic.h"
#include "slope_internal/linear/linear.h"

#include "fixtures/coupled_fixture.h"
#include "fixtures/hydraulic_fixture.h"
#include "fixtures/mechanics_fixture.h"

#include <petscsys.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_THREADS (SLOPE_ENABLE_OPENMP ? 2 : 1)

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
    const double checked_actual = (actual);                                            \
    const double checked_expected = (expected);                                        \
    if (!isfinite(checked_actual) ||                                                   \
        fabs(checked_actual - checked_expected) > (tolerance)) {                       \
      fprintf(stderr, "%s:%d: |%.17g - %.17g| > %.3g\n", __FILE__, __LINE__,           \
              checked_actual, checked_expected, (double)(tolerance));                  \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

typedef struct {
  SsTransferTargetPoint points[16];
  size_t point_count;
  uint32_t dimension;
  double affine_offset;
  uint64_t generation;
  uint64_t evaluations;
  int locate_mode;
  int same_mesh;
  int mutate_generation_on_evaluation;
  int fail_evaluation;
  char mesh_identity[SS_TRANSFER_ID_CAPACITY];
} TransferFixture;

static void simplex_reference_node(const SsFeDescriptor *descriptor, size_t node,
                                   double reference[SS_FE_MAX_DIMENSION]) {
  uint32_t coordinate;
  for (coordinate = 0U; coordinate < descriptor->dimension; ++coordinate) {
    reference[coordinate] =
        (double)descriptor
            ->barycentric_indices[node * (size_t)(descriptor->dimension + 1U) +
                                  coordinate + 1U] /
        (double)descriptor->order;
  }
}

static void curved_coordinate(uint32_t dimension, const double *reference,
                              double physical[SS_FE_MAX_DIMENSION]) {
  const double r = reference[0];
  const double s = reference[1];
  physical[0] = r + 0.08 * r * s;
  physical[1] = s + 0.06 * r * s;
  if (dimension == 3U) {
    const double t = reference[2];
    physical[0] += 0.03 * s * t;
    physical[1] += 0.02 * r * t;
    physical[2] = t + 0.05 * r * t + 0.04 * s * t;
  }
}

static double affine_pressure(const TransferFixture *fixture, const double *point) {
  double value = fixture->affine_offset + 2.0 * point[0] - 3.0 * point[1];
  if (fixture->dimension == 3U) {
    value += 0.5 * point[2];
  }
  return value;
}

static SsStatus fixture_target_count(void *opaque, size_t *count) {
  const TransferFixture *fixture = (const TransferFixture *)opaque;
  if (fixture == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *count = fixture->point_count;
  return SS_OK;
}

static SsStatus fixture_target_at(void *opaque, size_t index,
                                  SsTransferTargetPoint *target) {
  const TransferFixture *fixture = (const TransferFixture *)opaque;
  if (fixture == NULL || target == NULL || index >= fixture->point_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *target = fixture->points[index];
  return SS_OK;
}

static SsStatus
fixture_locate(void *opaque, const SsTransferTargetPoint *target,
               SsTransferSourceCandidate candidates[SS_TRANSFER_MAX_CANDIDATES],
               size_t *candidate_count) {
  const TransferFixture *fixture = (const TransferFixture *)opaque;
  uint32_t coordinate;
  if (fixture == NULL || target == NULL || candidates == NULL ||
      candidate_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (fixture->locate_mode == 1 &&
      target->stable_point_id == fixture->points[0].stable_point_id) {
    *candidate_count = 0U;
    return SS_OK;
  }
  memset(candidates, 0, SS_TRANSFER_MAX_CANDIDATES * sizeof(SsTransferSourceCandidate));
  candidates[0].stable_source_cell_id =
      fixture->same_mesh != 0
          ? target->stable_target_cell_id
          : UINT64_C(7000) + (target->physical[0] >= 0.5 ? UINT64_C(1) : UINT64_C(0));
  candidates[0].interior_margin = 0.25;
  candidates[0].owned = 1;
  candidates[0].ambiguous =
      fixture->locate_mode == 2 &&
      target->stable_point_id == fixture->points[0].stable_point_id;
  for (coordinate = 0U; coordinate < fixture->dimension; ++coordinate) {
    candidates[0].reference[coordinate] = target->physical[coordinate];
  }
  *candidate_count = 1U;
  return SS_OK;
}

static SsStatus fixture_evaluate(void *opaque, SsEntityId stable_source_cell_id,
                                 const double reference[SS_TRANSFER_MAX_DIMENSION],
                                 double *pressure,
                                 double gradient[SS_TRANSFER_MAX_DIMENSION]) {
  TransferFixture *fixture = (TransferFixture *)opaque;
  (void)stable_source_cell_id;
  if (fixture == NULL || reference == NULL || pressure == NULL || gradient == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ++fixture->evaluations;
  if (fixture->fail_evaluation != 0) {
    return SS_ERROR_INJECTED;
  }
  if (fixture->mutate_generation_on_evaluation != 0 && fixture->evaluations == 1U) {
    ++fixture->generation;
  }
  *pressure = affine_pressure(fixture, reference);
  gradient[0] = 2.0;
  gradient[1] = -3.0;
  gradient[2] = fixture->dimension == 3U ? 0.5 : 0.0;
  return SS_OK;
}

static SsStatus fixture_state(void *opaque, SsTransferSourceState *state) {
  const TransferFixture *fixture = (const TransferFixture *)opaque;
  if (fixture == NULL || state == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  state->accepted_generation = fixture->generation;
  state->mesh_generation = 11U;
  state->coordinate_generation = 12U;
  strcpy(state->mesh_identity, fixture->mesh_identity);
  strcpy(state->field_space_identity, "pressure-space-v1");
  strcpy(state->model_identity, "hydraulic-affine-v1");
  strcpy(state->pressure_unit, "kPa");
  return SS_OK;
}

static void fixture_init(TransferFixture *fixture, uint32_t dimension) {
  size_t cell;
  memset(fixture, 0, sizeof(*fixture));
  fixture->dimension = dimension;
  fixture->affine_offset = 1.0;
  fixture->generation = 23U;
  strcpy(fixture->mesh_identity, "mesh-source-v1");
  fixture->point_count = 8U;
  for (cell = 0U; cell < 4U; ++cell) {
    size_t point;
    for (point = 0U; point < 2U; ++point) {
      const size_t index = cell * 2U + point;
      SsTransferTargetPoint *target = &fixture->points[index];
      target->stable_point_id = UINT64_C(1000) + (SsEntityId)index;
      target->stable_target_cell_id = UINT64_C(2000) + (SsEntityId)cell;
      target->quadrature_index = (uint32_t)point;
      target->dimension = dimension;
      target->physical[0] = 0.12 * (double)cell + 0.04 * (double)point;
      target->physical[1] = cell == 1U ? 0.6 : 0.08 * (double)cell;
      target->physical[2] = dimension == 3U ? 0.03 * (double)(index + 1U) : 0.0;
      target->physical_weight = point == 0U ? 0.4 : 0.6;
      target->cell_epsilon = 2.0;
      target->saturated_unit_weight = 20.0;
      target->unsaturated_unit_weight = 18.0;
    }
  }
}

static SsCouplingPlanSpec fixture_spec(TransferFixture *fixture,
                                       SsCouplingTransferType transfer_type,
                                       int requested_threads) {
  SsCouplingPlanSpec spec;
  memset(&spec, 0, sizeof(spec));
  spec.dimension = fixture->dimension;
  spec.upward_axis = 1U;
  spec.transfer_type = transfer_type;
  spec.requested_threads = requested_threads;
  spec.openmp_work_threshold = 1U;
  fixture->same_mesh = transfer_type == SS_COUPLING_TRANSFER_SAME_MESH;
  strcpy(spec.target_mesh_identity,
         fixture->same_mesh != 0 ? fixture->mesh_identity : "mesh-target-v1");
  strcpy(spec.target_space_identity, "mechanics-space-v1");
  strcpy(spec.target_quadrature_identity, "mechanics-quadrature-v1");
  spec.geometry.context = fixture;
  spec.geometry.target_count = fixture_target_count;
  spec.geometry.target_at = fixture_target_at;
  spec.geometry.locate = fixture_locate;
  spec.geometry.evaluate_source = fixture_evaluate;
  spec.geometry.source_state = fixture_state;
  return spec;
}

static int test_ramp(void) {
  SsHydraulicRamp ramp = {
      .region = SS_HYDRAULIC_RAMP_DRY, .epsilon = 0.0, .value = 0.0, .derivative = 0.0};
  SsHydraulicInitializationRecord initialization = {0};
  CHECK(ss_hydraulic_ramp(-1.0, 2.0, &ramp) == SS_OK);
  CHECK(ramp.region == SS_HYDRAULIC_RAMP_DRY);
  CHECK_NEAR(ramp.value, 0.0, 0.0);
  CHECK_NEAR(ramp.derivative, 0.0, 0.0);
  CHECK(ss_hydraulic_ramp(0.0, 2.0, &ramp) == SS_OK);
  CHECK(ramp.region == SS_HYDRAULIC_RAMP_DRY);
  CHECK_NEAR(ramp.derivative, 0.0, 0.0);
  CHECK(ss_hydraulic_ramp(0.5, 2.0, &ramp) == SS_OK);
  CHECK(ramp.region == SS_HYDRAULIC_RAMP_TRANSITION);
  CHECK_NEAR(ramp.value, 0.25, 1.0e-15);
  CHECK_NEAR(ramp.derivative, 0.5, 1.0e-15);
  CHECK(ss_hydraulic_ramp(2.0, 2.0, &ramp) == SS_OK);
  CHECK(ramp.region == SS_HYDRAULIC_RAMP_SATURATED);
  CHECK_NEAR(ramp.value, 1.0, 0.0);
  CHECK_NEAR(ramp.derivative, 0.0, 0.0);
  CHECK(ss_hydraulic_ramp(3.0, 2.0, &ramp) == SS_OK);
  CHECK(ramp.region == SS_HYDRAULIC_RAMP_SATURATED);
  CHECK(ss_hydraulic_ramp(NAN, 2.0, &ramp) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_hydraulic_ramp(1.0, 0.0, &ramp) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_hydraulic_ramp(1.0, INFINITY, &ramp) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_hydraulic_initialization_select(11.725, 25.043, 1, &initialization) ==
        SS_OK);
  CHECK(initialization.choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR);
  CHECK(initialization.primary_choice == initialization.choice);
  CHECK(initialization.fallback_attempted == 0);
  CHECK(initialization.saturated_candidate_valid != 0);
  CHECK(ss_hydraulic_initialization_fallback(&initialization) == SS_OK);
  CHECK(initialization.primary_choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR);
  CHECK(initialization.choice == SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE);
  CHECK(initialization.fallback_attempted != 0);
  CHECK(ss_hydraulic_initialization_fallback(&initialization) ==
        SS_ERROR_INVALID_STATE);
  CHECK(ss_hydraulic_initialization_select(2650.8, 120.0, 1, &initialization) == SS_OK);
  CHECK(initialization.choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR);
  CHECK(ss_hydraulic_initialization_fallback(&initialization) == SS_OK);
  CHECK(initialization.choice == SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE);
  CHECK(ss_hydraulic_initialization_select(3.0, 3.0, 1, &initialization) == SS_OK);
  CHECK(initialization.choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR);
  CHECK(ss_hydraulic_initialization_select(5.0, 10.0, 1, &initialization) == SS_OK);
  CHECK(initialization.choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR);
  CHECK(ss_hydraulic_initialization_select(5.01, 10.0, 1, &initialization) == SS_OK);
  CHECK(initialization.choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR);
  CHECK(ss_hydraulic_initialization_select(3.0, NAN, 0, &initialization) == SS_OK);
  CHECK(initialization.choice == SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE);
  CHECK(initialization.saturated_candidate_valid == 0);
  CHECK(ss_hydraulic_initialization_fallback(&initialization) ==
        SS_ERROR_INVALID_STATE);
  CHECK(ss_hydraulic_initialization_fallback(NULL) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_hydraulic_initialization_select(NAN, 1.0, 1, &initialization) ==
        SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_hydraulic_initialization_select(1.0, NAN, 1, &initialization) ==
        SS_ERROR_INVALID_ARGUMENT);
  return 0;
}

static int test_tetrahedron_edge(void) {
  static const double coordinates[12] = {
      1.0, 2.0, 0.0, 4.0, 6.0, 0.0, 1.0, 2.0, 8.0, 2.0, 2.0, 12.0,
  };
  double minimum = 0.0;
  double epsilon = 0.0;
  const double corrected_edge = sqrt(17.0);
  const double historical_edge = 5.0;
  CHECK(ss_hydraulic_cell_epsilon(3U, coordinates, 9.81, &minimum, &epsilon) == SS_OK);
  CHECK_NEAR(minimum, corrected_edge, 2.0e-15);
  CHECK_NEAR(epsilon, 9.81 * corrected_edge / 2.0, 2.0e-14);
  CHECK(fabs(minimum - historical_edge) > 0.8);
  CHECK(ss_hydraulic_cell_epsilon(1U, coordinates, 9.81, &minimum, &epsilon) ==
        SS_ERROR_INVALID_ARGUMENT);
  return 0;
}

static int assemble_fan(uint32_t dimension, const double *coordinates,
                        const size_t *oriented_cells, const double *pressure,
                        const double *conductivity, const double *epsilon,
                        const double *expected_jacobian,
                        double expected_free_residual) {
  const SsFeDescriptor *descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  double assembled_residual[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double assembled_jacobian[25] = {0.0};
  size_t cell;
  CHECK(ss_fe_descriptor_find(dimension, 1U, &descriptor) == SS_OK);
  CHECK(ss_fe_quadrature_find(dimension == 2U ? "triangle-degree-1-v1"
                                              : "tetrahedron-degree-1-v1",
                              &quadrature) == SS_OK);
  for (cell = 0U; cell < 4U; ++cell) {
    double local_coordinates[12] = {0.0};
    double local_pressure[4] = {0.0};
    double residual[4] = {0.0};
    double jacobian[16] = {0.0};
    SsHydraulicPointModel model = {
        dimension, 1U, 9.81, conductivity[cell], epsilon[cell],
    };
    SsScalarElementInput element;
    SsElementOutput output;
    SsErrorRecord error = {0};
    size_t local_node;
    for (local_node = 0U; local_node < descriptor->node_count; ++local_node) {
      const size_t global_node =
          oriented_cells[cell * descriptor->node_count + local_node];
      uint32_t coordinate;
      local_pressure[local_node] = pressure[global_node];
      for (coordinate = 0U; coordinate < dimension; ++coordinate) {
        local_coordinates[local_node * dimension + coordinate] =
            coordinates[global_node * dimension + coordinate];
      }
    }
    memset(&element, 0, sizeof(element));
    element.dimension = dimension;
    element.field_descriptor = descriptor;
    element.coordinate_descriptor = descriptor;
    element.quadrature = quadrature;
    element.coordinate_values = local_coordinates;
    element.field_values = local_pressure;
    element.point_kernel = ss_hydraulic_point_kernel;
    element.point_context = &model;
    element.stable_cell_id = UINT64_C(3000) + (SsEntityId)cell;
    memset(&output, 0, sizeof(output));
    output.closure_capacity = 4U;
    output.matrix_capacity = 16U;
    output.residual = residual;
    output.jacobian = jacobian;
    CHECK(ss_assembly_scalar_element(&element, SS_ASSEMBLY_FUSED, &output, &error) ==
          SS_OK);
    CHECK(error.valid == 0);
    for (local_node = 0U; local_node < descriptor->node_count; ++local_node) {
      const size_t global_row =
          oriented_cells[cell * descriptor->node_count + local_node];
      size_t local_column;
      assembled_residual[global_row] += residual[local_node];
      for (local_column = 0U; local_column < descriptor->node_count; ++local_column) {
        const size_t global_column =
            oriented_cells[cell * descriptor->node_count + local_column];
        assembled_jacobian[global_row * 5U + global_column] +=
            jacobian[local_node * descriptor->node_count + local_column];
      }
    }
  }
  CHECK_NEAR(assembled_residual[4], expected_free_residual, 3.0e-13);
  for (cell = 0U; cell < 25U; ++cell) {
    CHECK_NEAR(assembled_jacobian[cell], expected_jacobian[cell], 4.0e-13);
  }
  return 0;
}

static int test_element_oracles(void) {
  static const double coordinates_2d[10] = {
      0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5,
  };
  static const size_t cells_2d[12] = {
      0U, 1U, 4U, 1U, 2U, 4U, 2U, 3U, 4U, 3U, 0U, 4U,
  };
  static const double pressure_2d[5] = {
      1.0, 1.0, 0.0, 0.0, 0.33754295039749843,
  };
  static const double conductivity[4] = {1.0, 2.0, 3.0, 4.0};
  static const double epsilon_2d[4] = {
      3.4683587617200162,
      3.4683587617200162,
      3.4683587617200162,
      3.4683587617200162,
  };
  static const double jacobian_2d[25] = {
      1.3214886980224207,
      -0.2357022603955158,
      0.0,
      -0.9428090415820636,
      -3.6785113019775793,
      -0.2357022603955159,
      0.7928932188134524,
      -0.4714045207910316,
      0.0,
      -2.2071067811865475,
      0.0,
      0.4714045207910318,
      3.6785113019775793,
      0.7071067811865475,
      -1.321488698022421,
      0.9428090415820632,
      0.0,
      0.7071067811865477,
      5.149915822768611,
      -1.8500841772313894,
      -2.028595479208968,
      -1.0285954792089684,
      -3.9142135623730954,
      -4.914213562373095,
      9.057190958417937,
  };
  static const double coordinates_3d[15] = {
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25, 0.25, 0.25,
  };
  static const size_t cells_3d[16] = {
      0U, 1U, 2U, 4U, 0U, 3U, 1U, 4U, 0U, 2U, 3U, 4U, 1U, 3U, 2U, 4U,
  };
  static const double pressure_3d[5] = {
      0.3, 0.3, 0.0, 0.3, 0.20819684934303137,
  };
  static const double epsilon_3d[4] = {
      2.1239273027813357,
      2.1239273027813357,
      2.1239273027813357,
      4.067011149173309,
  };
  static const double jacobian_3d[25] = {
      1.1150998205402494,   -0.11556261216234409, -0.025783423063208577,
      -0.12845432369394844, -2.3849001794597506,  0.02877495513506237,
      1.2542787366609836,   0.9755037815259211,   0.9209454033276501,
      -3.162387930005683,   0.35911675639654195,  1.1241200854843112,
      1.726790986115051,    1.3036784636825822,   -2.9398756805516153,
      0.11210828846839568,  0.9209454033276501,   1.0588371148592546,
      1.42094540332765,     -3.4957212633390164,  -1.6150998205402494,
      -3.1837816133106007,  -3.735348459437018,   -3.517114946643934,
      11.982885053356064,
  };
  CHECK(assemble_fan(2U, coordinates_2d, cells_2d, pressure_2d, conductivity,
                     epsilon_2d, jacobian_2d, 2.220446049250313e-16) == 0);
  CHECK(assemble_fan(3U, coordinates_3d, cells_3d, pressure_3d, conductivity,
                     epsilon_3d, jacobian_3d, 3.469446951953614e-17) == 0);
  {
    double quadrature_conductivity[2] = {1.5, 3.5};
    SsHydraulicCellCoefficients cell = {
        UINT64_C(0xb20000000001),
        2.0,
        2U,
        quadrature_conductivity,
    };
    SsHydraulicOperatorModelSpec spec;
    SsHydraulicOperatorModel *model = NULL;
    SsHydraulicOperatorModelSummary summary;
    SsScalarPointInput input;
    SsScalarPointOutput output;
    memset(&spec, 0, sizeof(spec));
    spec.dimension = 2U;
    spec.upward_axis = 1U;
    spec.gamma_water = 9.81;
    spec.cell_count = 1U;
    spec.cells = &cell;
    strcpy(spec.model_identity, "quadrature-heterogeneous-v1");
    strcpy(spec.pressure_unit, "kPa");
    strcpy(spec.coordinate_unit, "m");
    strcpy(spec.conductivity_unit, "m/s");
    CHECK(ss_hydraulic_operator_model_create(&spec, &model) == SS_OK);
    quadrature_conductivity[1] = 99.0;
    memset(&input, 0, sizeof(input));
    input.dimension = 2U;
    input.value = 0.5;
    input.gradient[0] = 1.0;
    input.gradient[1] = -2.0;
    input.stable_cell_id = cell.stable_cell_id;
    input.quadrature_index = 1U;
    memset(&output, 0, sizeof(output));
    CHECK(ss_hydraulic_operator_point_kernel(model, &input, &output) == SS_OK);
    CHECK_NEAR(output.flux[0], 3.5, 2.0e-15);
    CHECK_NEAR(output.flux[1], 3.5 * (-2.0 + 9.81 * 0.25), 3.0e-15);
    CHECK_NEAR(output.flux_derivative_gradient[0], 3.5, 0.0);
    CHECK_NEAR(output.flux_derivative_gradient[3], 3.5, 0.0);
    CHECK_NEAR(output.flux_derivative_value[1], 3.5 * 9.81 / 2.0, 3.0e-15);
    CHECK(ss_hydraulic_operator_model_set_gravity_mode(
              model, SS_HYDRAULIC_GRAVITY_DISABLED) == SS_OK);
    memset(&output, 0, sizeof(output));
    CHECK(ss_hydraulic_operator_point_kernel(model, &input, &output) == SS_OK);
    CHECK_NEAR(output.flux[0], 3.5, 2.0e-15);
    CHECK_NEAR(output.flux[1], -7.0, 2.0e-15);
    CHECK_NEAR(output.flux_derivative_value[1], 0.0, 0.0);
    CHECK(ss_hydraulic_operator_model_set_gravity_mode(
              model, SS_HYDRAULIC_GRAVITY_SATURATED) == SS_OK);
    memset(&output, 0, sizeof(output));
    CHECK(ss_hydraulic_operator_point_kernel(model, &input, &output) == SS_OK);
    CHECK_NEAR(output.flux[0], 3.5, 2.0e-15);
    CHECK_NEAR(output.flux[1], 3.5 * (-2.0 + 9.81), 3.0e-15);
    CHECK_NEAR(output.flux_derivative_value[1], 0.0, 0.0);
    CHECK(ss_hydraulic_operator_model_set_gravity_mode(
              model, SS_HYDRAULIC_GRAVITY_NONLINEAR) == SS_OK);
    CHECK(ss_hydraulic_operator_model_set_gravity_mode(
              model, (SsHydraulicGravityMode)0) == SS_ERROR_INVALID_ARGUMENT);
    CHECK(ss_hydraulic_operator_model_set_gravity_mode(
              NULL, SS_HYDRAULIC_GRAVITY_NONLINEAR) == SS_ERROR_INVALID_ARGUMENT);
    CHECK(ss_hydraulic_operator_model_summary(model, &summary) == SS_OK);
    CHECK(summary.quadrature_coefficient_count == 2U);
    CHECK_NEAR(summary.conductivity_minimum, 1.5, 0.0);
    CHECK_NEAR(summary.conductivity_maximum, 3.5, 0.0);
    input.quadrature_index = 2U;
    CHECK(ss_hydraulic_operator_point_kernel(model, &input, &output) ==
          SS_ERROR_NOT_FOUND);
    CHECK(ss_hydraulic_operator_model_destroy(&model) == SS_OK && model == NULL);
    spec.cell_count = 0U;
    spec.cells = NULL;
    CHECK(ss_hydraulic_operator_model_create(&spec, &model) == SS_OK);
    CHECK(ss_hydraulic_operator_model_summary(model, &summary) == SS_OK);
    CHECK(summary.cell_count == 0U);
    CHECK_NEAR(summary.conductivity_minimum, 0.0, 0.0);
    CHECK_NEAR(summary.conductivity_maximum, 0.0, 0.0);
    CHECK_NEAR(summary.epsilon_minimum, 0.0, 0.0);
    CHECK_NEAR(summary.epsilon_maximum, 0.0, 0.0);
    CHECK(ss_hydraulic_operator_point_kernel(model, &input, &output) ==
          SS_ERROR_NOT_FOUND);
    CHECK(ss_hydraulic_operator_model_destroy(&model) == SS_OK && model == NULL);
    spec.cell_count = SIZE_MAX;
    spec.cells = &cell;
    CHECK(ss_hydraulic_operator_model_create(&spec, &model) == SS_ERROR_OVERFLOW);
    CHECK(model == NULL);
    spec.cell_count = 1U;
    cell.conductivity_count = SIZE_MAX;
    CHECK(ss_hydraulic_operator_model_create(&spec, &model) == SS_ERROR_OVERFLOW);
    CHECK(model == NULL);
  }
  {
    HydraulicFixtureProfile profile_2d;
    HydraulicFixtureProfile profile_3d;
    CHECK(hydraulic_fixture_compact_profile(2U, TEST_THREADS, &profile_2d) == 0);
    CHECK(hydraulic_fixture_compact_profile(3U, TEST_THREADS, &profile_3d) == 0);
    CHECK_NEAR(profile_2d.accepted_free_pressure, pressure_2d[4], 3.0e-13);
    CHECK_NEAR(profile_3d.accepted_free_pressure, pressure_3d[4], 3.0e-13);
    CHECK(profile_2d.accepted_residual_norm < 1.0e-11);
    CHECK(profile_3d.accepted_residual_norm < 1.0e-11);
    CHECK_NEAR(profile_2d.accepted_jacobian_diagonal, jacobian_2d[24], 5.0e-12);
    CHECK_NEAR(profile_3d.accepted_jacobian_diagonal, jacobian_3d[24], 5.0e-12);
    CHECK_NEAR(profile_2d.conductivity_minimum, 1.0, 0.0);
    CHECK_NEAR(profile_2d.conductivity_maximum, 4.0, 0.0);
    CHECK_NEAR(profile_3d.epsilon_minimum, epsilon_3d[0], 5.0e-15);
    CHECK_NEAR(profile_3d.epsilon_maximum, epsilon_3d[3], 5.0e-15);
    CHECK(profile_2d.nonlinear_outcome == SS_NONLINEAR_CONVERGED);
    CHECK(profile_3d.nonlinear_outcome == SS_NONLINEAR_CONVERGED);
    CHECK(profile_2d.nonlinear_iterations > 0);
    CHECK(profile_3d.nonlinear_iterations > 0);
    CHECK(profile_2d.injected_failure_outcome != SS_NONLINEAR_CONVERGED);
    CHECK(profile_3d.injected_failure_outcome != SS_NONLINEAR_CONVERGED);
    CHECK(profile_2d.injected_failure_rolled_back == 1);
    CHECK(profile_3d.injected_failure_rolled_back == 1);
    CHECK(profile_2d.injected_failure_parent_unchanged == 1);
    CHECK(profile_3d.injected_failure_parent_unchanged == 1);
    CHECK(profile_2d.accepted_transaction_committed == 1);
    CHECK(profile_3d.accepted_transaction_committed == 1);
    CHECK(profile_2d.observed_threads == TEST_THREADS);
    CHECK(profile_3d.observed_threads == TEST_THREADS);
  }
  {
    HydraulicFixtureProfile case_007;
    HydraulicFixtureProfile case_009;
    CHECK(hydraulic_fixture_case_slice_profile("CASE-007", TEST_THREADS, &case_007) ==
          0);
    CHECK(hydraulic_fixture_case_slice_profile("CASE-009", TEST_THREADS, &case_009) ==
          0);
    CHECK(case_007.dimension == 2U && case_007.field_order == 1U);
    CHECK(case_009.dimension == 3U && case_009.field_order == 2U);
    CHECK(case_007.free_dof_count == 1);
    CHECK(case_009.free_dof_count == 5);
    CHECK_NEAR(case_007.accepted_free_pressure, 172.91649332964369, 2.0e-10);
    CHECK_NEAR(case_009.accepted_free_pressure, 218.27250000000004, 3.0e-10);
    CHECK_NEAR(case_007.accepted_pressure_norm, 172.91649332964369, 2.0e-10);
    CHECK_NEAR(case_009.accepted_pressure_norm, 539.59458906664372, 6.0e-10);
    CHECK(case_007.accepted_residual_norm < 1.0e-11);
    CHECK(case_009.accepted_residual_norm < 1.0e-10);
    CHECK_NEAR(case_007.accepted_jacobian_diagonal, 4.0, 2.0e-13);
    CHECK_NEAR(case_009.accepted_jacobian_diagonal, 2.4000000000000021, 3.0e-13);
    CHECK_NEAR(case_007.conductivity_minimum, 1.0, 0.0);
    CHECK_NEAR(case_007.conductivity_maximum, 1.0, 0.0);
    CHECK_NEAR(case_009.conductivity_minimum, 1.0, 0.0);
    CHECK_NEAR(case_009.conductivity_maximum, 1.0, 0.0);
    CHECK(case_009.epsilon_minimum < case_009.epsilon_maximum);
    CHECK(case_007.nonlinear_outcome == SS_NONLINEAR_CONVERGED);
    CHECK(case_009.nonlinear_outcome == SS_NONLINEAR_CONVERGED);
    CHECK(case_007.nonlinear_iterations == 1);
    CHECK(case_009.nonlinear_iterations == 2);
    CHECK(case_007.injected_failure_outcome != SS_NONLINEAR_CONVERGED);
    CHECK(case_009.injected_failure_outcome != SS_NONLINEAR_CONVERGED);
    CHECK(case_007.injected_failure_rolled_back == 1);
    CHECK(case_009.injected_failure_rolled_back == 1);
    CHECK(case_007.injected_failure_parent_unchanged == 1);
    CHECK(case_009.injected_failure_parent_unchanged == 1);
    CHECK(case_007.accepted_transaction_committed == 1);
    CHECK(case_009.accepted_transaction_committed == 1);
    CHECK(case_007.observed_threads == TEST_THREADS);
    CHECK(case_009.observed_threads == TEST_THREADS);
    CHECK(hydraulic_fixture_case_slice_profile("CASE-999", TEST_THREADS, &case_009) !=
          0);
  }
  return 0;
}

static int test_jacobian_property(void) {
  static const uint32_t orders[3] = {1U, 2U, 4U};
  uint32_t dimension;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    size_t field_order_index;
    for (field_order_index = 0U; field_order_index < 3U; ++field_order_index) {
      size_t geometry_order_index;
      for (geometry_order_index = 0U; geometry_order_index < 3U;
           ++geometry_order_index) {
        const SsFeDescriptor *field_descriptor = NULL;
        const SsFeDescriptor *coordinate_descriptor = NULL;
        SsFeQuadratureRequest request;
        SsFeQuadraturePlan plan = {0};
        SsHydraulicPointModel model = {
            dimension, 1U, 9.81, 2.7, 0.7,
        };
        double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
        double values[SS_FE_MAX_NODES] = {0.0};
        double plus_values[SS_FE_MAX_NODES] = {0.0};
        double minus_values[SS_FE_MAX_NODES] = {0.0};
        double direction[SS_FE_MAX_NODES] = {0.0};
        double residual[SS_FE_MAX_NODES] = {0.0};
        double plus_residual[SS_FE_MAX_NODES] = {0.0};
        double minus_residual[SS_FE_MAX_NODES] = {0.0};
        double jacobian[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
        SsScalarElementInput element;
        SsElementOutput output;
        SsElementOutput plus_output;
        SsElementOutput minus_output;
        SsErrorRecord error = {0};
        const double step = 2.0e-7;
        size_t node;
        CHECK(ss_fe_descriptor_find(dimension, orders[field_order_index],
                                    &field_descriptor) == SS_OK);
        CHECK(ss_fe_descriptor_find(dimension, orders[geometry_order_index],
                                    &coordinate_descriptor) == SS_OK);
        for (node = 0U; node < coordinate_descriptor->node_count; ++node) {
          double reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          double physical[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          uint32_t coordinate;
          simplex_reference_node(coordinate_descriptor, node, reference);
          curved_coordinate(dimension, reference, physical);
          for (coordinate = 0U; coordinate < dimension; ++coordinate) {
            coordinates[node * dimension + coordinate] = physical[coordinate];
          }
        }
        for (node = 0U; node < field_descriptor->node_count; ++node) {
          double reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          simplex_reference_node(field_descriptor, node, reference);
          values[node] = 0.28 + 0.05 * reference[0] + 0.03 * reference[1] +
                         (dimension == 3U ? 0.02 * reference[2] : 0.0);
          direction[node] = 0.07 * cos((double)(node + 1U) * 0.43);
          plus_values[node] = values[node] + step * direction[node];
          minus_values[node] = values[node] - step * direction[node];
        }
        memset(&request, 0, sizeof(request));
        request.parent_dimension = dimension;
        request.domain = SS_FE_QUADRATURE_VOLUME;
        request.operation = SS_FE_OPERATION_HYDRAULIC_DIFFUSION;
        request.field_order = orders[field_order_index];
        request.coordinate_order = orders[geometry_order_index];
        request.mode = SS_FE_QUADRATURE_MODE_DEFAULT;
        CHECK(ss_fe_quadrature_plan(&request, &plan) == SS_OK);
        memset(&element, 0, sizeof(element));
        element.dimension = dimension;
        element.field_descriptor = field_descriptor;
        element.coordinate_descriptor = coordinate_descriptor;
        element.quadrature = plan.selected_rule;
        element.coordinate_values = coordinates;
        element.field_values = values;
        element.point_kernel = ss_hydraulic_point_kernel;
        element.point_context = &model;
        element.stable_cell_id =
            UINT64_C(4000) +
            (SsEntityId)(dimension * 100U + orders[field_order_index] * 10U +
                         orders[geometry_order_index]);
        memset(&output, 0, sizeof(output));
        output.closure_capacity = SS_FE_MAX_NODES;
        output.matrix_capacity = SS_FE_MAX_NODES * SS_FE_MAX_NODES;
        output.residual = residual;
        output.jacobian = jacobian;
        memset(&plus_output, 0, sizeof(plus_output));
        plus_output.closure_capacity = SS_FE_MAX_NODES;
        plus_output.residual = plus_residual;
        memset(&minus_output, 0, sizeof(minus_output));
        minus_output.closure_capacity = SS_FE_MAX_NODES;
        minus_output.residual = minus_residual;
        CHECK(ss_assembly_scalar_element(&element, SS_ASSEMBLY_FUSED, &output,
                                         &error) == SS_OK);
        element.field_values = plus_values;
        CHECK(ss_assembly_scalar_element(&element, SS_ASSEMBLY_RESIDUAL, &plus_output,
                                         &error) == SS_OK);
        element.field_values = minus_values;
        CHECK(ss_assembly_scalar_element(&element, SS_ASSEMBLY_RESIDUAL, &minus_output,
                                         &error) == SS_OK);
        for (node = 0U; node < field_descriptor->node_count; ++node) {
          size_t column;
          double action = 0.0;
          const double difference =
              (plus_residual[node] - minus_residual[node]) / (2.0 * step);
          for (column = 0U; column < field_descriptor->node_count; ++column) {
            action += jacobian[node * field_descriptor->node_count + column] *
                      direction[column];
          }
          CHECK_NEAR(difference, action, 4.0e-7 * (1.0 + fabs(action)));
        }
      }
    }
  }
  return 0;
}

static int test_head(void) {
  const double physical[3] = {1.5, 2.0, -0.25};
  SsHydraulicHeadModel model;
  double pressure = -1.0;
  memset(&model, 0, sizeof(model));
  model.dimension = 3U;
  model.upward_axis = 1U;
  model.gamma_water = 9.81;
  model.kind = SS_HYDRAULIC_HEAD_DRY;
  strcpy(model.identity, "dry-v1");
  CHECK(ss_hydraulic_head_evaluate(&model, physical, &pressure) == SS_OK);
  CHECK_NEAR(pressure, 0.0, 0.0);
  model.kind = SS_HYDRAULIC_HEAD_CONSTANT_LEVEL;
  model.level = 3.0;
  strcpy(model.identity, "constant-level-v1");
  CHECK(ss_hydraulic_head_evaluate(&model, physical, &pressure) == SS_OK);
  CHECK_NEAR(pressure, 9.81, 2.0e-15);
  model.kind = SS_HYDRAULIC_HEAD_PIECEWISE_LINEAR_LEVEL;
  model.piecewise_axis = 0U;
  model.piecewise_point_count = 3U;
  model.piecewise_coordinates[0] = 0.0;
  model.piecewise_coordinates[1] = 1.0;
  model.piecewise_coordinates[2] = 2.0;
  model.piecewise_levels[0] = 2.0;
  model.piecewise_levels[1] = 2.4;
  model.piecewise_levels[2] = 2.8;
  strcpy(model.identity, "piecewise-segment-v1");
  CHECK(ss_hydraulic_head_evaluate(&model, physical, &pressure) == SS_OK);
  CHECK_NEAR(pressure, 9.81 * 0.6, 2.0e-14);
  CHECK(ss_hydraulic_head_evaluate(&model, (double[3]){-1.0, 1.0, 0.0}, &pressure) ==
        SS_OK);
  CHECK_NEAR(pressure, 9.81, 2.0e-14);
  CHECK(ss_hydraulic_head_evaluate(&model, (double[3]){3.0, 1.0, 0.0}, &pressure) ==
        SS_OK);
  CHECK_NEAR(pressure, 9.81 * 1.8, 3.0e-14);
  model.piecewise_coordinates[1] = 0.0;
  CHECK(ss_hydraulic_head_evaluate(&model, physical, &pressure) ==
        SS_ERROR_INVALID_ARGUMENT);
  model.piecewise_coordinates[1] = 1.0;
  model.kind = SS_HYDRAULIC_HEAD_COMPILED_AFFINE_LEVEL;
  model.affine_level[0] = 1.0;
  model.affine_level[1] = 0.2;
  model.affine_level[2] = 0.7;
  model.affine_level[3] = -0.1;
  strcpy(model.identity, "compiled-affine-v1");
  CHECK(ss_hydraulic_head_evaluate(&model, physical, &pressure) == SS_OK);
  CHECK_NEAR(pressure, 7.11225, 2.0e-14);
  model.kind = SS_HYDRAULIC_HEAD_NODAL_VALUE;
  model.nodal_value = 4.25;
  strcpy(model.identity, "nodal-pressure-v1");
  CHECK(ss_hydraulic_head_evaluate(&model, physical, &pressure) == SS_OK);
  CHECK_NEAR(pressure, 4.25, 0.0);
  model.gamma_water = 0.0;
  CHECK(ss_hydraulic_head_evaluate(&model, physical, &pressure) ==
        SS_ERROR_INVALID_ARGUMENT);
  return 0;
}

static int test_flux(void) {
  static const uint32_t orders[3] = {1U, 2U, 4U};
  double integrals[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  uint32_t dimension;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    size_t order_index;
    for (order_index = 0U; order_index < 3U; ++order_index) {
      const SsFeDescriptor *descriptor = NULL;
      const SsFeQuadrature *quadrature = NULL;
      double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
      const double parent_centroid[3] = {
          dimension == 2U ? 1.0 / 3.0 : 0.25,
          dimension == 2U ? 1.0 / 3.0 : 0.25,
          dimension == 2U ? 0.0 : 0.25,
      };
      SsHydraulicFluxModel model;
      size_t node;
      CHECK(ss_fe_descriptor_find(dimension, orders[order_index], &descriptor) ==
            SS_OK);
      CHECK(ss_fe_facet_quadrature_default(dimension, orders[order_index],
                                           &quadrature) == SS_OK);
      for (node = 0U; node < descriptor->node_count; ++node) {
        double reference[3] = {0.0, 0.0, 0.0};
        double physical[3] = {0.0, 0.0, 0.0};
        uint32_t coordinate;
        simplex_reference_node(descriptor, node, reference);
        curved_coordinate(dimension, reference, physical);
        for (coordinate = 0U; coordinate < dimension; ++coordinate) {
          coordinates[node * dimension + coordinate] = physical[coordinate];
        }
      }
      memset(&model, 0, sizeof(model));
      model.kind = SS_HYDRAULIC_FLUX_COMPILED_AFFINE;
      model.dimension = dimension;
      model.affine[0] = 1.25;
      model.affine[1] = 0.3;
      model.affine[2] = -0.2;
      model.affine[3] = 0.15;
      strcpy(model.identity, "curved-affine-flux-v1");
      for (node = 0U; node < quadrature->point_count; ++node) {
        const double *reference = quadrature->points + node * (size_t)(dimension - 1U);
        SsFeFacetGeometry geometry = {0};
        SsNaturalPointInput input;
        SsNaturalPointOutput output;
        double expected = 0.0;
        double normal_squared = 0.0;
        uint32_t coordinate;
        CHECK(ss_fe_facet_geometry_eval(descriptor, 1U, coordinates, reference,
                                        parent_centroid, &geometry) == SS_OK);
        CHECK(isfinite(geometry.measure) && geometry.measure > 0.0);
        memset(&input, 0, sizeof(input));
        input.dimension = dimension;
        input.components = 1U;
        memcpy(input.physical, geometry.physical, dimension * sizeof(double));
        memcpy(input.outward_normal, geometry.outward_normal,
               dimension * sizeof(double));
        for (coordinate = 0U; coordinate < dimension; ++coordinate) {
          normal_squared +=
              geometry.outward_normal[coordinate] * geometry.outward_normal[coordinate];
        }
        CHECK_NEAR(normal_squared, 1.0, 2.0e-14);
        memset(&output, 0, sizeof(output));
        CHECK(ss_hydraulic_flux_evaluate(&model, geometry.physical, &expected) ==
              SS_OK);
        CHECK(ss_hydraulic_flux_point_kernel(&model, &input, &output) == SS_OK);
        CHECK_NEAR(output.load[0], expected, 1.0e-15);
        CHECK_NEAR(output.signed_flux, expected, 1.0e-15);
        integrals[dimension - 2U][order_index] +=
            quadrature->weights[node] * geometry.measure * output.signed_flux;
      }
      CHECK(isfinite(integrals[dimension - 2U][order_index]) &&
            integrals[dimension - 2U][order_index] > 0.0);
    }
    CHECK(fabs(integrals[dimension - 2U][1] - integrals[dimension - 2U][0]) > 1.0e-6);
  }
  {
    SsHydraulicFluxModel zero;
    double value = 1.0;
    memset(&zero, 0, sizeof(zero));
    zero.kind = SS_HYDRAULIC_FLUX_ZERO;
    zero.dimension = 2U;
    strcpy(zero.identity, "natural-zero-v1");
    CHECK(ss_hydraulic_flux_evaluate(&zero, (double[2]){0.0, 0.0}, &value) == SS_OK);
    CHECK_NEAR(value, 0.0, 0.0);
  }
  return 0;
}

static SsLinearPolicy general_policy(void) {
  SsLinearPolicy policy;
  memset(&policy, 0, sizeof(policy));
  strcpy(policy.identity, "hydraulic-general-v1");
  strcpy(policy.options_prefix, "hyd_");
  policy.operator_class = SS_LINEAR_OPERATOR_GENERAL;
  policy.method = SS_LINEAR_METHOD_FGMRES;
  policy.pc = SS_LINEAR_PC_JACOBI;
  policy.side = SS_LINEAR_SIDE_RIGHT;
  policy.rtol = 1.0e-8;
  policy.atol = 1.0e-12;
  policy.dtol = 1.0e4;
  policy.maximum_iterations = 100;
  policy.restart = 30;
  policy.vector_memory_cap_bytes = 4096U;
  return policy;
}

static int test_nonsymmetric_contract(void) {
  const double gradient[2] = {0.4, -0.7};
  SsHydraulicPointModel model = {2U, 1U, 9.81, 3.0, 2.0};
  SsScalarPointInput input;
  SsScalarPointOutput output;
  SsHydraulicOperatorTraits traits;
  SsLinearOperatorProperties properties;
  SsLinearPolicy policy = general_policy();
  const double basis[2] = {0.2, 0.8};
  const double basis_gradient[4] = {-1.0, -1.0, 1.0, 1.0};
  double matrix[4] = {0.0, 0.0, 0.0, 0.0};
  size_t row;
  memset(&input, 0, sizeof(input));
  input.dimension = 2U;
  input.value = 0.5;
  memcpy(input.gradient, gradient, sizeof(gradient));
  memset(&output, 0, sizeof(output));
  CHECK(ss_hydraulic_point_kernel(&model, &input, &output) == SS_OK);
  for (row = 0U; row < 2U; ++row) {
    size_t column;
    for (column = 0U; column < 2U; ++column) {
      size_t coordinate;
      for (coordinate = 0U; coordinate < 2U; ++coordinate) {
        matrix[row * 2U + column] +=
            basis_gradient[row * 2U + coordinate] *
            (output.flux_derivative_value[coordinate] * basis[column] +
             output.flux_derivative_gradient[coordinate * 2U] *
                 basis_gradient[column * 2U] +
             output.flux_derivative_gradient[coordinate * 2U + 1U] *
                 basis_gradient[column * 2U + 1U]);
      }
    }
  }
  CHECK(fabs(matrix[1] - matrix[2]) > 1.0);
  memset(&traits, 0, sizeof(traits));
  CHECK(ss_hydraulic_operator_traits(&traits) == SS_OK);
  CHECK(traits.scalar == 1 && traits.real == 1 && traits.generally_nonsymmetric == 1 &&
        traits.pressure_dependent_values == 1);
  memset(&properties, 0, sizeof(properties));
  properties.finite_action = 1;
  properties.nonlinear_hydraulic = traits.pressure_dependent_values;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
  policy.operator_class = SS_LINEAR_OPERATOR_SPD_VERIFIED;
  policy.method = SS_LINEAR_METHOD_CG;
  policy.side = SS_LINEAR_SIDE_LEFT;
  properties.measured_symmetric = 1;
  properties.positive_energy = 1;
  properties.pc_symmetric = 1;
  properties.pc_positive = 1;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_TYPE_MISMATCH);
  return 0;
}

static int validate_snapshot(const TransferFixture *fixture,
                             const SsCouplingFieldSnapshot *snapshot,
                             SsCouplingMode mode) {
  SsCouplingSnapshotView view;
  SsCouplingDiagnostics diagnostics;
  size_t point;
  memset(&view, 0, sizeof(view));
  memset(&diagnostics, 0, sizeof(diagnostics));
  CHECK(ss_coupling_field_snapshot_view(snapshot, &view) == SS_OK);
  CHECK(ss_coupling_field_snapshot_diagnostics(snapshot, &diagnostics) == SS_OK);
  CHECK(view.point_count == fixture->point_count);
  CHECK(view.cell_count == 4U);
  CHECK(view.dimension == fixture->dimension);
  CHECK(view.mode == mode);
  CHECK(view.source_generation == fixture->generation);
  CHECK(strcmp(view.source_mesh_identity, "mesh-source-v1") == 0);
  CHECK(strcmp(view.source_space_identity, "pressure-space-v1") == 0);
  CHECK(strcmp(view.source_model_identity, "hydraulic-affine-v1") == 0);
  CHECK(strcmp(view.pressure_unit, "kPa") == 0);
  CHECK(strcmp(view.target_mesh_identity, "mesh-source-v1") == 0);
  CHECK(strcmp(view.target_space_identity, "mechanics-space-v1") == 0);
  CHECK(strcmp(view.target_quadrature_identity, "mechanics-quadrature-v1") == 0);
  CHECK(strlen(diagnostics.content_digest) == 71U);
  CHECK(diagnostics.unresolved_points == 0U && diagnostics.ambiguous_points == 0U);
  for (point = 0U; point < fixture->point_count; ++point) {
    uint32_t coordinate;
    double mechanics_force[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
    const double expected_pressure =
        affine_pressure(fixture, fixture->points[point].physical);
    CHECK(view.point_ids[point] == fixture->points[point].stable_point_id);
    CHECK_NEAR(view.pressure[point], expected_pressure, 2.0e-14);
    CHECK_NEAR(view.gradient[point * fixture->dimension], 2.0, 2.0e-14);
    CHECK_NEAR(view.gradient[point * fixture->dimension + 1U], -3.0, 2.0e-14);
    if (fixture->dimension == 3U) {
      CHECK_NEAR(view.gradient[point * fixture->dimension + 2U], 0.5, 2.0e-14);
    }
    CHECK(ss_coupling_field_snapshot_mechanics_force(
              snapshot, fixture->dimension,
              fixture->points[point].stable_target_cell_id,
              fixture->points[point].quadrature_index, fixture->points[point].physical,
              mechanics_force) == SS_OK);
    for (coordinate = 0U; coordinate < fixture->dimension; ++coordinate) {
      const double gradient = view.gradient[point * fixture->dimension + coordinate];
      const double expected =
          coordinate == 1U
              ? -view.selected_unit_weight[point / 2U] -
                    (mode == SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT ? gradient : 0.0)
              : (mode == SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT ? -gradient : 0.0);
      CHECK_NEAR(view.body_force[point * fixture->dimension + coordinate], expected,
                 2.0e-14);
      CHECK_NEAR(mechanics_force[coordinate], expected, 2.0e-14);
    }
  }
  CHECK(ss_coupling_field_snapshot_mechanics_force(
            snapshot, fixture->dimension, UINT64_C(999999), 0U,
            fixture->points[0].physical,
            (double[SS_FE_MAX_DIMENSION]){0.0, 0.0, 0.0}) == SS_ERROR_NOT_FOUND);
  return 0;
}

static int test_coupled_ssr_profiles(void) {
  double checksums[2] = {0.0, 0.0};
  uint32_t dimension;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    CoupledFixture *fixture = NULL;
    const SsCouplingFieldSnapshot *snapshot;
    CHECK(coupled_fixture_create(PETSC_COMM_SELF, dimension, TEST_THREADS, &fixture) ==
          0);
    snapshot = coupled_fixture_snapshot(fixture);
    CHECK(snapshot != NULL);
    CHECK(mechanics_fixture_coupled_ssr_profile(
              dimension, ss_coupling_field_snapshot_mechanics_force, snapshot,
              TEST_THREADS, &checksums[dimension - 2U]) == 0);
    CHECK(coupled_fixture_destroy(&fixture) == 0);
  }
  CHECK(isfinite(checksums[0]) && isfinite(checksums[1]));
  CHECK(fabs(checksums[0] - checksums[1]) > 1.0e-6);
  return 0;
}

static int test_same_mesh(void) {
  TransferFixture fixture;
  SsCouplingPlanSpec spec;
  SsCouplingTransferPlan *plan = NULL;
  SsCouplingFieldSnapshot *snapshot = NULL;
  SsCouplingPlanSummary summary;
  fixture_init(&fixture, 3U);
  spec = fixture_spec(&fixture, SS_COUPLING_TRANSFER_SAME_MESH, 1);
  memset(&summary, 0, sizeof(summary));
  CHECK(ss_coupling_plan_create(PETSC_COMM_SELF, &spec, &plan) == SS_OK);
  CHECK(ss_coupling_plan_summary(plan, &summary) == SS_OK);
  CHECK(summary.local_target_points == 8U && summary.global_target_points == 8U);
  CHECK(summary.local_target_cells == 4U && summary.global_target_cells == 4U);
  CHECK(summary.resident_target_points == 8U);
  CHECK(summary.resident_bytes > sizeof(SsCouplingPlanSummary));
  CHECK(summary.transfer_type == SS_COUPLING_TRANSFER_SAME_MESH);
  CHECK(ss_coupling_field_snapshot_build(plan, SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT,
                                         &snapshot) == SS_OK);
  CHECK(validate_snapshot(&fixture, snapshot,
                          SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT) == 0);
  CHECK(fixture.evaluations == fixture.point_count);
  CHECK(ss_coupling_field_snapshot_destroy(&snapshot) == SS_OK && snapshot == NULL);
  CHECK(ss_coupling_plan_destroy(&plan) == SS_OK && plan == NULL);
#if !SLOPE_ENABLE_OPENMP
  spec = fixture_spec(&fixture, SS_COUPLING_TRANSFER_SAME_MESH, 2);
  CHECK(ss_coupling_plan_create(PETSC_COMM_SELF, &spec, &plan) ==
        SS_ERROR_INVALID_ARGUMENT);
  CHECK(plan == NULL);
#endif
  CHECK(test_coupled_ssr_profiles() == 0);
  return 0;
}

static int test_modes(void) {
  static const double corrected_gradients[12] = {
      0.0,
      -0.3,
      -0.06721260262787443,
      0.0,
      -0.3672126026278745,
      0.0,
      -0.06721260262787443,
      -0.3,
      0.0,
      0.06721260262787443,
      -0.2327873973721255,
      0.06721260262787443,
  };
  static const double selected_weight[4] = {18.0, 20.0, 18.0, 18.0};
  static const double expected_full[12] = {
      0.0,
      -17.7,
      0.06721260262787443,
      0.0,
      -19.632787397372127,
      0.0,
      0.06721260262787443,
      -17.7,
      0.0,
      -0.06721260262787443,
      -17.767212602627875,
      -0.06721260262787443,
  };
  size_t point;
  for (point = 0U; point < 4U; ++point) {
    double full[3] = {0.0, 0.0, 0.0};
    double weight_only[3] = {0.0, 0.0, 0.0};
    uint32_t coordinate;
    CHECK(ss_coupling_body_force(3U, 1U, SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT,
                                 corrected_gradients + point * 3U,
                                 selected_weight[point], full) == SS_OK);
    CHECK(ss_coupling_body_force(3U, 1U, SS_COUPLING_WEIGHT_ONLY,
                                 corrected_gradients + point * 3U,
                                 selected_weight[point], weight_only) == SS_OK);
    for (coordinate = 0U; coordinate < 3U; ++coordinate) {
      CHECK_NEAR(full[coordinate], expected_full[point * 3U + coordinate], 2.0e-14);
      CHECK_NEAR(weight_only[coordinate],
                 coordinate == 1U ? -selected_weight[point] : 0.0, 0.0);
    }
  }
  CHECK(ss_coupling_body_force(3U, 3U, SS_COUPLING_WEIGHT_ONLY, corrected_gradients,
                               18.0, (double[3]){0.0, 0.0, 0.0}) ==
        SS_ERROR_INVALID_ARGUMENT);
  return 0;
}

static int test_immutable(void) {
  TransferFixture fixture;
  SsCouplingPlanSpec spec;
  SsCouplingTransferPlan *plan = NULL;
  SsCouplingFieldSnapshot *snapshot = NULL;
  SsCouplingFieldSnapshot *replacement = NULL;
  SsCouplingSnapshotView before;
  SsCouplingSnapshotView after;
  SsCouplingDiagnostics before_diagnostics;
  SsCouplingDiagnostics after_diagnostics;
  double retained_pressure[16] = {0.0};
  fixture_init(&fixture, 2U);
  spec = fixture_spec(&fixture, SS_COUPLING_TRANSFER_SAME_MESH, TEST_THREADS);
  CHECK(ss_coupling_plan_create(PETSC_COMM_SELF, &spec, &plan) == SS_OK);
  CHECK(ss_coupling_field_snapshot_build(plan, SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT,
                                         &snapshot) == SS_OK);
  CHECK(ss_coupling_field_snapshot_view(snapshot, &before) == SS_OK);
  CHECK(ss_coupling_field_snapshot_diagnostics(snapshot, &before_diagnostics) == SS_OK);
  memcpy(retained_pressure, before.pressure, before.point_count * sizeof(double));
  fixture.affine_offset += 9.0;
  ++fixture.generation;
  CHECK(ss_coupling_field_snapshot_build(plan, SS_COUPLING_WEIGHT_ONLY, &replacement) ==
        SS_ERROR_INVALID_STATE);
  CHECK(replacement == NULL);
  CHECK(ss_coupling_field_snapshot_view(snapshot, &after) == SS_OK);
  CHECK(ss_coupling_field_snapshot_diagnostics(snapshot, &after_diagnostics) == SS_OK);
  CHECK(memcmp(retained_pressure, after.pressure, after.point_count * sizeof(double)) ==
        0);
  CHECK(strcmp(before_diagnostics.content_digest, after_diagnostics.content_digest) ==
        0);
  CHECK(ss_coupling_field_snapshot_destroy(&snapshot) == SS_OK);
  CHECK(ss_coupling_plan_destroy(&plan) == SS_OK);
  spec = fixture_spec(&fixture, SS_COUPLING_TRANSFER_SAME_MESH, TEST_THREADS);
  CHECK(ss_coupling_plan_create(PETSC_COMM_SELF, &spec, &plan) == SS_OK);
  CHECK(ss_coupling_field_snapshot_build(plan, SS_COUPLING_WEIGHT_ONLY, &replacement) ==
        SS_OK);
  CHECK(ss_coupling_field_snapshot_view(replacement, &after) == SS_OK);
  CHECK(fabs(after.pressure[0] - retained_pressure[0]) > 8.0);
  CHECK(ss_coupling_field_snapshot_destroy(&replacement) == SS_OK);
  CHECK(ss_coupling_plan_destroy(&plan) == SS_OK);
  fixture_init(&fixture, 2U);
  spec = fixture_spec(&fixture, SS_COUPLING_TRANSFER_SAME_MESH, TEST_THREADS);
  CHECK(ss_coupling_plan_create(PETSC_COMM_SELF, &spec, &plan) == SS_OK);
  fixture.mutate_generation_on_evaluation = 1;
  CHECK(ss_coupling_field_snapshot_build(plan, SS_COUPLING_WEIGHT_ONLY, &replacement) ==
        SS_ERROR_INVALID_STATE);
  CHECK(replacement == NULL);
  CHECK(fixture.generation == 24U);
  CHECK(ss_coupling_plan_destroy(&plan) == SS_OK);
  fixture_init(&fixture, 2U);
  spec = fixture_spec(&fixture, SS_COUPLING_TRANSFER_SAME_MESH, TEST_THREADS);
  CHECK(ss_coupling_plan_create(PETSC_COMM_SELF, &spec, &plan) == SS_OK);
  fixture.fail_evaluation = 1;
  CHECK(ss_coupling_field_snapshot_build(plan, SS_COUPLING_WEIGHT_ONLY, &replacement) ==
        SS_ERROR_INJECTED);
  CHECK(replacement == NULL);
  CHECK(fixture.generation == 23U);
  CHECK(ss_coupling_plan_destroy(&plan) == SS_OK);
  fixture_init(&fixture, 2U);
  fixture.generation = 0U;
  spec = fixture_spec(&fixture, SS_COUPLING_TRANSFER_SAME_MESH, TEST_THREADS);
  CHECK(ss_coupling_plan_create(PETSC_COMM_SELF, &spec, &plan) == SS_ERROR_INTEGRITY);
  CHECK(plan == NULL);
  return 0;
}

static int dispatch(const char *test_id) {
  if (strcmp(test_id, "HYD-UNIT-RAMP-001") == 0) {
    return test_ramp();
  }
  if (strcmp(test_id, "HYD-REG-TETEDGE-001") == 0) {
    return test_tetrahedron_edge();
  }
  if (strcmp(test_id, "HYD-UNIT-ELEMENT-001") == 0) {
    return test_element_oracles();
  }
  if (strcmp(test_id, "HYD-PROP-JACOBIAN-001") == 0) {
    return test_jacobian_property();
  }
  if (strcmp(test_id, "HYD-UNIT-HEAD-001") == 0) {
    return test_head();
  }
  if (strcmp(test_id, "HYD-UNIT-FLUX-001") == 0) {
    return test_flux();
  }
  if (strcmp(test_id, "HYD-CONTRACT-NONSYM-001") == 0) {
    return test_nonsymmetric_contract();
  }
  if (strcmp(test_id, "CPL-UNIT-SAME-001") == 0) {
    return test_same_mesh();
  }
  if (strcmp(test_id, "CPL-UNIT-MODES-001") == 0) {
    return test_modes();
  }
  if (strcmp(test_id, "CPL-UNIT-IMMUTABLE-001") == 0) {
    return test_immutable();
  }
  fprintf(stderr, "unknown hydraulic or coupling test: %s\n", test_id);
  return 2;
}

int main(int argc, char **argv) {
  int result;
  if (argc != 2) {
    fprintf(stderr, "usage: %s TEST_ID\n", argv[0]);
    return 2;
  }
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 2;
  }
  result = dispatch(argv[1]);
  if (PetscFinalize() != PETSC_SUCCESS && result == 0) {
    result = 2;
  }
  return result;
}
