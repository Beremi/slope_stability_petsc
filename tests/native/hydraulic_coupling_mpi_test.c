/* Distributed seepage balance and hydraulic-to-mechanics coupling at MPI and
 * hybrid layouts.
 *
 * Launched by tests/native/support/mpi_layout_runner.py --suite hydraulic.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/coupling/coupling.h"
#include "slope_internal/hydraulic/hydraulic.h"

#include "fixtures/coupled_fixture.h"
#include "fixtures/mechanics_fixture.h"

#include <petscsys.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "rank %d %s:%d: assertion failed: %s\n", fixture_rank, __FILE__, \
              __LINE__, #expression);                                                  \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                        \
  do {                                                                                 \
    const double checked_actual = (actual);                                            \
    const double checked_expected = (expected);                                        \
    if (!isfinite(checked_actual) ||                                                   \
        fabs(checked_actual - checked_expected) > (tolerance)) {                       \
      fprintf(stderr, "rank %d %s:%d: |%.17g - %.17g| > %.3g\n", fixture_rank,         \
              __FILE__, __LINE__, checked_actual, checked_expected,                    \
              (double)(tolerance));                                                    \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static int fixture_rank = -1;

typedef struct {
  const char *test_id;
  const char *output;
  int expected_ranks;
  int expected_threads;
} Arguments;

typedef struct {
  MPI_Comm communicator;
  int rank;
  int size;
  int requested_threads;
  int locate_mode;
  SsTransferTargetPoint local_points[32];
  size_t local_point_count;
  uint64_t generation;
  uint64_t evaluations;
} DistributedFixture;

static int parse_integer(const char *text, int *value) {
  char *end = NULL;
  long parsed;
  if (text == NULL || value == NULL || text[0] == '\0') {
    return 0;
  }
  parsed = strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > 1024) {
    return 0;
  }
  *value = (int)parsed;
  return 1;
}

static int parse_arguments(int argc, char **argv, Arguments *arguments) {
  int index;
  memset(arguments, 0, sizeof(*arguments));
  arguments->expected_threads = 1;
  for (index = 1; index < argc; ++index) {
    if (strcmp(argv[index], "--test-id") == 0 && index + 1 < argc) {
      arguments->test_id = argv[++index];
    } else if (strcmp(argv[index], "--output") == 0 && index + 1 < argc) {
      arguments->output = argv[++index];
    } else if (strcmp(argv[index], "--expected-ranks") == 0 && index + 1 < argc) {
      if (!parse_integer(argv[++index], &arguments->expected_ranks)) {
        return 0;
      }
    } else if (strcmp(argv[index], "--expected-threads") == 0 && index + 1 < argc) {
      if (!parse_integer(argv[++index], &arguments->expected_threads)) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return arguments->test_id != NULL && arguments->output != NULL &&
         arguments->expected_ranks > 0;
}

static double expected_pressure(const double *physical) {
  return 1.25 + 2.0 * physical[0] - 3.0 * physical[1] + 0.5 * physical[2];
}

static void fixture_init(DistributedFixture *fixture, MPI_Comm communicator,
                         int requested_threads, int locate_mode) {
  size_t cell;
  memset(fixture, 0, sizeof(*fixture));
  fixture->communicator = communicator;
  fixture->requested_threads = requested_threads;
  fixture->locate_mode = locate_mode;
  fixture->generation = 31U;
  (void)MPI_Comm_rank(communicator, &fixture->rank);
  (void)MPI_Comm_size(communicator, &fixture->size);
  for (cell = 0U; cell < 12U; ++cell) {
    size_t point;
    if ((locate_mode == 5 && fixture->rank != 0) ||
        (locate_mode != 5 && (int)(cell % (size_t)fixture->size) != fixture->rank)) {
      continue;
    }
    for (point = 0U; point < 2U; ++point) {
      const size_t global_index = cell * 2U + point;
      SsTransferTargetPoint *target =
          &fixture->local_points[fixture->local_point_count++];
      target->stable_point_id = UINT64_C(10000) + (SsEntityId)global_index;
      target->stable_target_cell_id = UINT64_C(11000) + (SsEntityId)cell;
      target->quadrature_index = (uint32_t)point;
      target->dimension = 3U;
      target->physical[0] =
          cell == 6U ? 0.5 : (double)cell / 13.0 + 0.01 * (double)point;
      target->physical[1] = 0.04 * (double)(cell % 4U) + 0.02 * (double)point;
      target->physical[2] = 0.03 * (double)(cell % 3U) + 0.01 * (double)point;
      target->physical_weight = point == 0U ? 0.35 : 0.65;
      target->cell_epsilon = 2.0 + 0.01 * (double)cell;
      target->saturated_unit_weight = 20.0;
      target->unsaturated_unit_weight = 18.0;
    }
  }
  if (fixture->size > 1 && fixture->rank == 1 && fixture->local_point_count != 0U) {
    if (fixture->locate_mode == 3) {
      fixture->local_points[0].stable_point_id = UINT64_C(10000);
    } else if (fixture->locate_mode == 4) {
      fixture->local_points[0].stable_target_cell_id = UINT64_C(11000);
    }
  }
}

static SsStatus distributed_target_count(void *opaque, size_t *count) {
  const DistributedFixture *fixture = (const DistributedFixture *)opaque;
  if (fixture == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *count = fixture->local_point_count;
  return SS_OK;
}

static SsStatus distributed_target_at(void *opaque, size_t index,
                                      SsTransferTargetPoint *target) {
  const DistributedFixture *fixture = (const DistributedFixture *)opaque;
  if (fixture == NULL || target == NULL || index >= fixture->local_point_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *target = fixture->local_points[index];
  return SS_OK;
}

static void fill_candidate(const SsTransferTargetPoint *target, SsEntityId cell_id,
                           int rank, int size, SsTransferSourceCandidate *candidate) {
  uint32_t coordinate;
  memset(candidate, 0, sizeof(*candidate));
  candidate->stable_source_cell_id = cell_id;
  candidate->interior_margin = fabs(target->physical[0] - 0.5) < 1.0e-15 ? 0.0 : 0.1;
  candidate->owned = (int)(cell_id % (SsEntityId)size) == rank;
  for (coordinate = 0U; coordinate < 3U; ++coordinate) {
    candidate->reference[coordinate] = target->physical[coordinate];
  }
}

static SsStatus
distributed_locate(void *opaque, const SsTransferTargetPoint *target,
                   SsTransferSourceCandidate candidates[SS_TRANSFER_MAX_CANDIDATES],
                   size_t *candidate_count) {
  const DistributedFixture *fixture = (const DistributedFixture *)opaque;
  const int interface_point = fabs(target->physical[0] - 0.5) < 1.0e-15;
  if (fixture == NULL || target == NULL || candidates == NULL ||
      candidate_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *candidate_count = 0U;
  if (fixture->locate_mode == 1 && target->stable_point_id == UINT64_C(10000)) {
    return SS_OK;
  }
  if (interface_point != 0) {
    SsTransferSourceCandidate candidate;
    fill_candidate(target, UINT64_C(8000), fixture->rank, fixture->size, &candidate);
    if (candidate.owned != 0) {
      candidates[(*candidate_count)++] = candidate;
    }
    fill_candidate(target, UINT64_C(8001), fixture->rank, fixture->size, &candidate);
    if (candidate.owned != 0) {
      candidates[(*candidate_count)++] = candidate;
    }
  } else {
    const SsEntityId cell_id =
        target->physical[0] < 0.5 ? UINT64_C(8000) : UINT64_C(8001);
    SsTransferSourceCandidate candidate;
    fill_candidate(target, cell_id, fixture->rank, fixture->size, &candidate);
    if (candidate.owned != 0) {
      candidates[(*candidate_count)++] = candidate;
    }
  }
  if (fixture->locate_mode == 2 && target->stable_point_id == UINT64_C(10000) &&
      *candidate_count != 0U) {
    candidates[0].ambiguous = 1;
  }
  return SS_OK;
}

static SsStatus distributed_evaluate(void *opaque, SsEntityId stable_source_cell_id,
                                     const double reference[SS_TRANSFER_MAX_DIMENSION],
                                     double *pressure,
                                     double gradient[SS_TRANSFER_MAX_DIMENSION]) {
  DistributedFixture *fixture = (DistributedFixture *)opaque;
  if (fixture == NULL || reference == NULL || pressure == NULL || gradient == NULL ||
      (stable_source_cell_id != UINT64_C(8000) &&
       stable_source_cell_id != UINT64_C(8001))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ++fixture->evaluations;
  *pressure = expected_pressure(reference);
  gradient[0] = 2.0;
  gradient[1] = -3.0;
  gradient[2] = 0.5;
  return SS_OK;
}

static SsStatus distributed_state(void *opaque, SsTransferSourceState *state) {
  const DistributedFixture *fixture = (const DistributedFixture *)opaque;
  if (fixture == NULL || state == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  state->accepted_generation = fixture->generation;
  state->mesh_generation = 101U;
  state->coordinate_generation = 102U;
  strcpy(state->mesh_identity, "distributed-source-mesh-v1");
  strcpy(state->field_space_identity, "distributed-pressure-space-v1");
  strcpy(state->model_identity, "distributed-affine-hydraulic-v1");
  strcpy(state->pressure_unit, "kPa");
  return SS_OK;
}

static SsCouplingPlanSpec distributed_spec(DistributedFixture *fixture) {
  SsCouplingPlanSpec spec;
  memset(&spec, 0, sizeof(spec));
  spec.dimension = 3U;
  spec.upward_axis = 1U;
  spec.transfer_type = SS_COUPLING_TRANSFER_DIFFERENT_MESH;
  spec.requested_threads = fixture->requested_threads;
  spec.openmp_work_threshold = 1U;
  strcpy(spec.target_mesh_identity, "distributed-target-mesh-v1");
  strcpy(spec.target_space_identity, "distributed-mechanics-space-v1");
  strcpy(spec.target_quadrature_identity, "distributed-quadrature-v1");
  spec.geometry.context = fixture;
  spec.geometry.target_count = distributed_target_count;
  spec.geometry.target_at = distributed_target_at;
  spec.geometry.locate = distributed_locate;
  spec.geometry.evaluate_source = distributed_evaluate;
  spec.geometry.source_state = distributed_state;
  return spec;
}

static void hydraulic_reference_node(const SsFeDescriptor *descriptor, size_t node,
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

static void hydraulic_curved_coordinate(uint32_t dimension, const double *reference,
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

static int manufactured_balance_dimension(MPI_Comm communicator, uint32_t dimension,
                                          double *maximum_error, double *checksum) {
  const SsFeDescriptor *descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  const double desired_flux[3] = {0.37, -0.22, 0.19};
  double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
  double centroid_reference[3] = {0.0, 0.0, 0.0};
  double parent_centroid[3] = {0.0, 0.0, 0.0};
  SsHydraulicPointModel point_model;
  SsHydraulicPointResult point_result;
  SsHydraulicFluxModel flux_model;
  SsHydraulicBalanceLocal local;
  SsHydraulicBalance balance;
  int rank = 0;
  int size = 0;
  size_t node;
  uint32_t facet;
  CHECK(MPI_Comm_rank(communicator, &rank) == MPI_SUCCESS);
  CHECK(MPI_Comm_size(communicator, &size) == MPI_SUCCESS);
  CHECK(ss_fe_descriptor_find(dimension, 4U, &descriptor) == SS_OK);
  CHECK(ss_fe_facet_quadrature_default(dimension, 4U, &quadrature) == SS_OK);
  for (node = 0U; node < descriptor->node_count; ++node) {
    double reference[3] = {0.0, 0.0, 0.0};
    double physical[3] = {0.0, 0.0, 0.0};
    uint32_t coordinate;
    hydraulic_reference_node(descriptor, node, reference);
    hydraulic_curved_coordinate(dimension, reference, physical);
    for (coordinate = 0U; coordinate < dimension; ++coordinate) {
      coordinates[node * dimension + coordinate] = physical[coordinate];
    }
  }
  for (facet = 0U; facet < dimension; ++facet) {
    centroid_reference[facet] = 1.0 / (double)(dimension + 1U);
  }
  hydraulic_curved_coordinate(dimension, centroid_reference, parent_centroid);
  memset(&point_model, 0, sizeof(point_model));
  point_model.dimension = dimension;
  point_model.upward_axis = 1U;
  point_model.gamma_water = 9.81;
  point_model.conductivity = 1.0;
  point_model.epsilon = 1.0;
  {
    double gradient[3] = {
        desired_flux[0],
        desired_flux[1] - point_model.gamma_water,
        desired_flux[2],
    };
    memset(&point_result, 0, sizeof(point_result));
    CHECK(ss_hydraulic_point_evaluate(&point_model, 100.0, gradient, &point_result) ==
          SS_OK);
    for (facet = 0U; facet < dimension; ++facet) {
      CHECK_NEAR(point_result.pressure_flux[facet], desired_flux[facet], 2.0e-15);
    }
  }
  memset(&flux_model, 0, sizeof(flux_model));
  flux_model.kind = SS_HYDRAULIC_FLUX_COMPILED_VECTOR_NORMAL;
  flux_model.dimension = dimension;
  memcpy(flux_model.vector, desired_flux, dimension * sizeof(double));
  strcpy(flux_model.identity, "manufactured-curved-vector-normal-v1");
  memset(&local, 0, sizeof(local));
  for (facet = 0U; facet <= dimension; ++facet) {
    double integral = 0.0;
    if ((int)(facet % (uint32_t)size) != rank) {
      continue;
    }
    for (node = 0U; node < quadrature->point_count; ++node) {
      const double *reference = quadrature->points + node * (size_t)(dimension - 1U);
      SsFeFacetGeometry geometry = {0};
      SsNaturalPointInput input;
      SsNaturalPointOutput output;
      double independently_evaluated = 0.0;
      CHECK(ss_fe_facet_geometry_eval(descriptor, facet, coordinates, reference,
                                      parent_centroid, &geometry) == SS_OK);
      memset(&input, 0, sizeof(input));
      input.dimension = dimension;
      input.components = 1U;
      memcpy(input.physical, geometry.physical, dimension * sizeof(double));
      memcpy(input.outward_normal, geometry.outward_normal, dimension * sizeof(double));
      memset(&output, 0, sizeof(output));
      CHECK(ss_hydraulic_flux_evaluate_normal(&flux_model, input.physical,
                                              input.outward_normal,
                                              &independently_evaluated) == SS_OK);
      CHECK(ss_hydraulic_flux_point_kernel(&flux_model, &input, &output) == SS_OK);
      CHECK_NEAR(output.signed_flux, independently_evaluated, 2.0e-15);
      integral += quadrature->weights[node] * geometry.measure * output.signed_flux;
    }
    if (facet == 1U) {
      local.prescribed_boundary_flux += integral;
    } else {
      local.dirichlet_reaction_flux += integral;
    }
  }
  memset(&balance, 0, sizeof(balance));
  CHECK(ss_hydraulic_balance(communicator, &local, 1.0, 2.0e-12, 2.0e-12, &balance) ==
        SS_OK);
  CHECK(balance.accepted == 1);
  CHECK(fabs(balance.prescribed_boundary_flux) > 1.0e-4);
  CHECK_NEAR(balance.signed_imbalance, 0.0, 2.0e-12);
  *maximum_error = fabs(balance.signed_imbalance);
  *checksum = balance.prescribed_boundary_flux + 3.0 * balance.dirichlet_reaction_flux +
              0.1 * balance.normalization;
  return 0;
}

static int test_balance(MPI_Comm communicator, double *maximum_error,
                        double *checksum) {
  int size = 0;
  double manufactured_error_2d = 0.0;
  double manufactured_error_3d = 0.0;
  double manufactured_checksum_2d = 0.0;
  double manufactured_checksum_3d = 0.0;
  double rejected_imbalance = 0.0;
  SsHydraulicBalanceLocal local;
  SsHydraulicBalance balance;
  CHECK(MPI_Comm_size(communicator, &size) == MPI_SUCCESS);
  CHECK(manufactured_balance_dimension(communicator, 2U, &manufactured_error_2d,
                                       &manufactured_checksum_2d) == 0);
  CHECK(manufactured_balance_dimension(communicator, 3U, &manufactured_error_3d,
                                       &manufactured_checksum_3d) == 0);
  memset(&local, 0, sizeof(local));
  local.domain_source = 0.25 / (double)size;
  local.prescribed_boundary_flux = -0.5 / (double)size;
  local.natural_boundary_flux = 0.1 / (double)size;
  local.dirichlet_reaction_flux = 0.15 / (double)size;
  local.free_residual_l2_squared = 1.0e-20 / (double)size;
  memset(&balance, 0, sizeof(balance));
  CHECK(ss_hydraulic_balance(communicator, &local, 1.0, 1.0e-9, 1.0e-9, &balance) ==
        SS_OK);
  CHECK(balance.accepted == 1);
  CHECK_NEAR(balance.domain_source, 0.25, 2.0e-15);
  CHECK_NEAR(balance.prescribed_boundary_flux, -0.5, 2.0e-15);
  CHECK_NEAR(balance.natural_boundary_flux, 0.1, 2.0e-15);
  CHECK_NEAR(balance.dirichlet_reaction_flux, 0.15, 2.0e-15);
  CHECK_NEAR(balance.signed_imbalance, 0.0, 2.0e-15);
  CHECK_NEAR(balance.physical_scale, 1.0, 0.0);
  CHECK_NEAR(balance.normalization, 1.0, 2.0e-15);
  CHECK_NEAR(balance.free_residual_l2, 1.0e-10, 2.0e-22);
  local.natural_boundary_flux += 0.01 / (double)size;
  CHECK(ss_hydraulic_balance(communicator, &local, 1.0, 1.0e-12, 1.0e-12, &balance) ==
        SS_OK);
  CHECK(balance.accepted == 0);
  rejected_imbalance = balance.signed_imbalance;
  memset(&local, 0, sizeof(local));
  local.domain_source = 2.0e-4 / (double)size;
  CHECK(ss_hydraulic_balance(communicator, &local, 10.0, 3.0e-5, 0.0, &balance) ==
        SS_OK);
  CHECK(balance.accepted == 1);
  CHECK_NEAR(balance.physical_scale, 10.0, 0.0);
  CHECK_NEAR(balance.normalization, 10.0, 0.0);
  CHECK_NEAR(balance.normalized_imbalance, 2.0e-5, 2.0e-20);
  CHECK(ss_hydraulic_balance(communicator, &local, 0.0, 3.0e-5, 0.0, &balance) ==
        SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_hydraulic_balance(communicator, &local, 10.0, -1.0, 0.0, &balance) ==
        SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_hydraulic_balance(communicator, &local, 10.0, 3.0e-5, -1.0, &balance) ==
        SS_ERROR_INVALID_ARGUMENT);
  *maximum_error = fmax(fabs(rejected_imbalance - 0.01),
                        fmax(manufactured_error_2d, manufactured_error_3d));
  *checksum = balance.domain_source + balance.normalization + balance.free_residual_l2 +
              manufactured_checksum_2d + manufactured_checksum_3d;
  return 0;
}

static int test_coupled_trajectories(MPI_Comm communicator, int requested_threads,
                                     double *maximum_error, double *checksum) {
  double local_checksums[2] = {0.0, 0.0};
  double minimum_checksums[2] = {0.0, 0.0};
  double maximum_checksums[2] = {0.0, 0.0};
  uint32_t dimension;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    CoupledFixture *fixture = NULL;
    const SsCouplingFieldSnapshot *snapshot;
    CHECK(coupled_fixture_create(PETSC_COMM_SELF, dimension, requested_threads,
                                 &fixture) == 0);
    snapshot = coupled_fixture_snapshot(fixture);
    CHECK(snapshot != NULL);
    CHECK(mechanics_fixture_coupled_ssr_profile(
              dimension, ss_coupling_field_snapshot_mechanics_force, snapshot,
              requested_threads, &local_checksums[dimension - 2U]) == 0);
    CHECK(coupled_fixture_destroy(&fixture) == 0);
  }
  CHECK(MPI_Allreduce(local_checksums, minimum_checksums, 2, MPI_DOUBLE, MPI_MIN,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(local_checksums, maximum_checksums, 2, MPI_DOUBLE, MPI_MAX,
                      communicator) == MPI_SUCCESS);
  *maximum_error = fmax(maximum_checksums[0] - minimum_checksums[0],
                        maximum_checksums[1] - minimum_checksums[1]);
  *checksum = local_checksums[0] + local_checksums[1];
  return 0;
}

static int same_mesh_source_rank(const DistributedFixture *fixture, SsEntityId cell) {
  /* Permute source ownership independently of both target layouts. */
  return (int)(((cell - UINT64_C(11000)) + 1U) / 2U % (SsEntityId)fixture->size);
}

static SsStatus same_mesh_map(void *opaque, const SsTransferTargetPoint *target,
                              int *source_rank,
                              double reference[SS_TRANSFER_MAX_DIMENSION]) {
  const DistributedFixture *fixture = opaque;
  *source_rank = same_mesh_source_rank(fixture, target->stable_target_cell_id);
  memcpy(reference, target->physical, sizeof(target->physical));
  return SS_OK;
}

static SsStatus same_mesh_evaluate(void *opaque, SsEntityId cell,
                                   const double reference[SS_TRANSFER_MAX_DIMENSION],
                                   double *pressure,
                                   double gradient[SS_TRANSFER_MAX_DIMENSION]) {
  const DistributedFixture *fixture = opaque;
  if (fixture->rank != same_mesh_source_rank(fixture, cell)) {
    return SS_ERROR_INTEGRITY;
  }
  return distributed_evaluate(opaque, UINT64_C(8000), reference, pressure, gradient);
}

static int test_routed_same_mesh(MPI_Comm communicator, int threads) {
  DistributedFixture serial;
  fixture_init(&serial, PETSC_COMM_SELF, 1, 0);
  SsCouplingPlanSpec spec = distributed_spec(&serial);
  spec.transfer_type = SS_COUPLING_TRANSFER_SAME_MESH;
  strcpy(spec.target_mesh_identity, "distributed-source-mesh-v1");
  spec.geometry.locate = NULL;
  spec.geometry.evaluate_source = same_mesh_evaluate;
  spec.same_mesh_map = same_mesh_map;
  SsCouplingTransferPlan *reference = NULL;
  CHECK(ss_coupling_plan_create(PETSC_COMM_SELF, &spec, &reference) == SS_OK);
  for (int concentrated = 0; concentrated <= 1; ++concentrated) {
    DistributedFixture fixture;
    fixture_init(&fixture, communicator, threads, concentrated ? 5 : 0);
    spec.geometry.context = &fixture;
    spec.requested_threads = threads;
    SsCouplingTransferPlan *plan = NULL;
    CHECK(ss_coupling_plan_create(communicator, &spec, &plan) == SS_OK);
    for (int i = 0; i < 2; ++i) {
      const SsCouplingMode mode =
          i ? SS_COUPLING_WEIGHT_ONLY : SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT;
      SsCouplingFieldSnapshot *snapshot = NULL, *oracle = NULL;
      SsCouplingSnapshotView view, expected;
      CHECK(ss_coupling_field_snapshot_build(reference, mode, &oracle) == SS_OK);
      CHECK(ss_coupling_field_snapshot_build(plan, mode, &snapshot) == SS_OK);
      CHECK(ss_coupling_field_snapshot_view(oracle, &expected) == SS_OK);
      CHECK(ss_coupling_field_snapshot_view(snapshot, &view) == SS_OK);
      CHECK(view.point_count == fixture.local_point_count);
      CHECK(strcmp(view.snapshot_identity, expected.snapshot_identity) == 0);
      for (size_t p = 0; p < view.point_count; ++p) {
        const size_t serial_point = (size_t)(view.point_ids[p] - UINT64_C(10000));
        CHECK(serial_point < expected.point_count);
        CHECK(view.pressure[p] == expected.pressure[serial_point]);
        for (uint32_t d = 0; d < view.dimension; ++d) {
          CHECK(view.body_force[view.dimension * p + d] ==
                expected.body_force[view.dimension * serial_point + d]);
        }
      }
      CHECK(ss_coupling_field_snapshot_destroy(&snapshot) == SS_OK);
      CHECK(ss_coupling_field_snapshot_destroy(&oracle) == SS_OK);
    }
    uint64_t evaluations = 0;
    CHECK(MPI_Allreduce(&fixture.evaluations, &evaluations, 1, MPI_UINT64_T, MPI_SUM,
                        communicator) == MPI_SUCCESS);
    CHECK(evaluations == 2U * serial.local_point_count);
    CHECK(ss_coupling_plan_destroy(&plan) == SS_OK);
  }
  CHECK(ss_coupling_plan_destroy(&reference) == SS_OK);
  return 0;
}

static int test_transfer(MPI_Comm communicator, int requested_threads,
                         int run_coupled_trajectories, double *maximum_error,
                         double *checksum, int *observed_threads, char invariant[72]) {
  DistributedFixture fixture;
  SsCouplingPlanSpec spec;
  SsCouplingTransferPlan *plan = NULL;
  SsCouplingFieldSnapshot *snapshot = NULL;
  SsCouplingPlanSummary summary;
  SsCouplingSnapshotView view;
  SsCouplingDiagnostics diagnostics;
  double local_error = 0.0;
  double global_error = 0.0;
  double local_checksum = 0.0;
  double global_checksum = 0.0;
  double coupled_error = 0.0;
  double coupled_checksum = 0.0;
  uint64_t global_evaluations = 0U;
  size_t point;
  fixture_init(&fixture, communicator, requested_threads, 0);
  spec = distributed_spec(&fixture);
  memset(&summary, 0, sizeof(summary));
  CHECK(ss_coupling_plan_create(communicator, &spec, &plan) == SS_OK);
  CHECK(ss_coupling_plan_summary(plan, &summary) == SS_OK);
  CHECK(summary.local_target_points == fixture.local_point_count);
  CHECK(summary.global_target_points == 24U);
  CHECK(summary.global_target_cells == 12U);
  CHECK(summary.resident_target_points == fixture.local_point_count);
  CHECK(summary.resident_bytes >= sizeof(SsCouplingPlanSummary));
  CHECK(fixture.size == 1 ||
        summary.resident_target_points < summary.global_target_points);
  CHECK(summary.transfer_type == SS_COUPLING_TRANSFER_DIFFERENT_MESH);
  CHECK(ss_coupling_field_snapshot_build(plan, SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT,
                                         &snapshot) == SS_OK);
  memset(&view, 0, sizeof(view));
  memset(&diagnostics, 0, sizeof(diagnostics));
  CHECK(ss_coupling_field_snapshot_view(snapshot, &view) == SS_OK);
  CHECK(ss_coupling_field_snapshot_diagnostics(snapshot, &diagnostics) == SS_OK);
  CHECK(view.point_count == fixture.local_point_count);
  CHECK(view.cell_count * 2U == view.point_count);
  CHECK(diagnostics.global_target_points == 24U);
  CHECK(diagnostics.global_target_cells == 12U);
  CHECK(diagnostics.unresolved_points == 0U);
  CHECK(diagnostics.ambiguous_points == 0U);
  for (point = 0U; point < view.point_count; ++point) {
    const double expected = expected_pressure(fixture.local_points[point].physical);
    uint32_t coordinate;
    const double pressure_error = fabs(view.pressure[point] - expected);
    if (pressure_error > local_error) {
      local_error = pressure_error;
    }
    CHECK(view.point_ids[point] == fixture.local_points[point].stable_point_id);
    for (coordinate = 0U; coordinate < 3U; ++coordinate) {
      const double expected_gradient =
          coordinate == 0U ? 2.0 : (coordinate == 1U ? -3.0 : 0.5);
      const double gradient_error =
          fabs(view.gradient[point * 3U + coordinate] - expected_gradient);
      if (gradient_error > local_error) {
        local_error = gradient_error;
      }
      local_checksum += view.body_force[point * 3U + coordinate];
    }
    local_checksum += view.pressure[point];
  }
  CHECK(MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&fixture.evaluations, &global_evaluations, 1, MPI_UINT64_T,
                      MPI_SUM, communicator) == MPI_SUCCESS);
  CHECK(global_evaluations == 24U);
  CHECK(strlen(diagnostics.content_digest) == 71U);
  strcpy(invariant, diagnostics.content_digest);
  *observed_threads = diagnostics.observed_threads;
  *maximum_error = global_error;
  *checksum = global_checksum;
  CHECK(ss_coupling_field_snapshot_destroy(&snapshot) == SS_OK);
  CHECK(ss_coupling_plan_destroy(&plan) == SS_OK);
  if (run_coupled_trajectories != 0) {
    fixture_init(&fixture, communicator, requested_threads, 5);
    spec = distributed_spec(&fixture);
    memset(&summary, 0, sizeof(summary));
    CHECK(ss_coupling_plan_create(communicator, &spec, &plan) == SS_OK);
    CHECK(ss_coupling_plan_summary(plan, &summary) == SS_OK);
    CHECK(summary.local_target_points == fixture.local_point_count);
    CHECK(summary.global_target_points == 24U);
    CHECK(fixture.size == 1 || fixture.rank == 0 ||
          summary.resident_target_points == 0U);
    CHECK(ss_coupling_field_snapshot_build(
              plan, SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT, &snapshot) == SS_OK);
    memset(&view, 0, sizeof(view));
    memset(&diagnostics, 0, sizeof(diagnostics));
    CHECK(ss_coupling_field_snapshot_view(snapshot, &view) == SS_OK);
    CHECK(ss_coupling_field_snapshot_diagnostics(snapshot, &diagnostics) == SS_OK);
    CHECK(view.point_count == fixture.local_point_count);
    CHECK(diagnostics.global_target_points == 24U);
    CHECK(ss_coupling_field_snapshot_destroy(&snapshot) == SS_OK);
    CHECK(ss_coupling_plan_destroy(&plan) == SS_OK);
    CHECK(test_routed_same_mesh(communicator, requested_threads) == 0);
    CHECK(test_coupled_trajectories(communicator, requested_threads, &coupled_error,
                                    &coupled_checksum) == 0);
    if (coupled_error > *maximum_error) {
      *maximum_error = coupled_error;
    }
    *checksum += coupled_checksum;
  }
  return 0;
}

static int test_negative_coverage(MPI_Comm communicator, int requested_threads,
                                  double *maximum_error, double *checksum) {
  DistributedFixture fixture;
  SsCouplingPlanSpec spec;
  SsCouplingTransferPlan *plan = NULL;
  fixture_init(&fixture, communicator, requested_threads, 1);
  spec = distributed_spec(&fixture);
  CHECK(ss_coupling_plan_create(communicator, &spec, &plan) == SS_ERROR_NOT_FOUND);
  CHECK(plan == NULL);
  fixture_init(&fixture, communicator, requested_threads, 2);
  spec = distributed_spec(&fixture);
  CHECK(ss_coupling_plan_create(communicator, &spec, &plan) == SS_ERROR_DUPLICATE);
  CHECK(plan == NULL);
  if (fixture.size > 1) {
    fixture_init(&fixture, communicator, requested_threads, 3);
    spec = distributed_spec(&fixture);
    CHECK(ss_coupling_plan_create(communicator, &spec, &plan) == SS_ERROR_DUPLICATE);
    CHECK(plan == NULL);
    fixture_init(&fixture, communicator, requested_threads, 4);
    spec = distributed_spec(&fixture);
    CHECK(ss_coupling_plan_create(communicator, &spec, &plan) == SS_ERROR_DUPLICATE);
    CHECK(plan == NULL);
  }
  *maximum_error = 0.0;
  *checksum = 4.0;
  return 0;
}

static int publish_result(const Arguments *arguments, MPI_Comm communicator,
                          int observed_threads, double maximum_error, double checksum,
                          const char *invariant) {
  int rank = 0;
  FILE *stream;
  CHECK(MPI_Comm_rank(communicator, &rank) == MPI_SUCCESS);
  if (rank != 0) {
    return 0;
  }
  stream = fopen(arguments->output, "wb");
  if (stream == NULL) {
    fprintf(stderr, "cannot open %s\n", arguments->output);
    return 1;
  }
  if (fprintf(stream,
              "{\"test_id\":\"%s\",\"ranks\":%d,\"requested_threads\":%d,"
              "\"observed_threads\":%d,\"invariant\":\"%s\","
              "\"maximum_error\":%.17g,\"checksum\":%.17g}\n",
              arguments->test_id, arguments->expected_ranks,
              arguments->expected_threads, observed_threads, invariant, maximum_error,
              checksum) < 0 ||
      fclose(stream) != 0) {
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  Arguments arguments;
  MPI_Comm communicator;
  int size = 0;
  int result = 0;
  int global_result = 0;
  int observed_threads = 1;
  double maximum_error = 0.0;
  double checksum = 0.0;
  char invariant[72] =
      "sha256:8f0284bc8a9a5f3ee29365f4af8898f036730d0700702a2c97dc37d7137d9a55";
  if (!parse_arguments(argc, argv, &arguments)) {
    fprintf(stderr,
            "usage: %s --test-id ID --expected-ranks N --expected-threads N "
            "--output PATH\n",
            argv[0]);
    return 2;
  }
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 2;
  }
  communicator = PETSC_COMM_WORLD;
  (void)MPI_Comm_rank(communicator, &fixture_rank);
  (void)MPI_Comm_size(communicator, &size);
  if (size != arguments.expected_ranks) {
    result = 1;
  } else if (strcmp(arguments.test_id, "HYD-CONTRACT-BALANCE-001") == 0) {
    result = test_balance(communicator, &maximum_error, &checksum);
  } else if (strcmp(arguments.test_id, "CPL-CONTRACT-DIFFERENT-001") == 0 ||
             strcmp(arguments.test_id, "CPL-CONTRACT-LAYOUT-001") == 0) {
    result = test_transfer(communicator, arguments.expected_threads,
                           strcmp(arguments.test_id, "CPL-CONTRACT-LAYOUT-001") == 0,
                           &maximum_error, &checksum, &observed_threads, invariant);
  } else if (strcmp(arguments.test_id, "CPL-NEG-COVERAGE-001") == 0) {
    result = test_negative_coverage(communicator, arguments.expected_threads,
                                    &maximum_error, &checksum);
  } else {
    fprintf(stderr, "unknown hydraulic or coupling MPI test: %s\n", arguments.test_id);
    result = 2;
  }
  if (MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, communicator) !=
      MPI_SUCCESS) {
    global_result = 2;
  }
  if (global_result == 0) {
    global_result = publish_result(&arguments, communicator, observed_threads,
                                   maximum_error, checksum, invariant);
  }
  if (PetscFinalize() != PETSC_SUCCESS && global_result == 0) {
    global_result = 2;
  }
  return global_result;
}
