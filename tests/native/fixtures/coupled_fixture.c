/* Compact hydraulic-to-mechanics coupling fixture of the coupling tests.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "coupled_fixture.h"

#include "slope_internal/fe/fe.h"
#include "slope_internal/hydraulic/hydraulic.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct CoupledFixture {
  SsTransferTargetPoint points[16];
  size_t point_count;
  uint32_t dimension;
  uint64_t generation;
  char mesh_identity[SS_TRANSFER_ID_CAPACITY];
  SsCouplingTransferPlan *plan;
  SsCouplingFieldSnapshot *snapshot;
};

static SsStatus coupled_target_count(void *opaque, size_t *count) {
  const CoupledFixture *fixture = (const CoupledFixture *)opaque;
  if (fixture == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *count = fixture->point_count;
  return SS_OK;
}

static SsStatus coupled_target_at(void *opaque, size_t index,
                                  SsTransferTargetPoint *target) {
  const CoupledFixture *fixture = (const CoupledFixture *)opaque;
  if (fixture == NULL || target == NULL || index >= fixture->point_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *target = fixture->points[index];
  return SS_OK;
}

static SsStatus
coupled_locate(void *opaque, const SsTransferTargetPoint *target,
               SsTransferSourceCandidate candidates[SS_TRANSFER_MAX_CANDIDATES],
               size_t *candidate_count) {
  const CoupledFixture *fixture = (const CoupledFixture *)opaque;
  uint32_t coordinate;
  if (fixture == NULL || target == NULL || candidates == NULL ||
      candidate_count == NULL || target->dimension != fixture->dimension) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(candidates, 0, SS_TRANSFER_MAX_CANDIDATES * sizeof(SsTransferSourceCandidate));
  candidates[0].stable_source_cell_id = target->stable_target_cell_id;
  candidates[0].interior_margin = 0.25;
  candidates[0].owned = 1;
  for (coordinate = 0U; coordinate < fixture->dimension; ++coordinate) {
    candidates[0].reference[coordinate] = target->physical[coordinate];
  }
  *candidate_count = 1U;
  return SS_OK;
}

static SsStatus coupled_evaluate(void *opaque, SsEntityId stable_source_cell_id,
                                 const double reference[SS_TRANSFER_MAX_DIMENSION],
                                 double *pressure,
                                 double gradient[SS_TRANSFER_MAX_DIMENSION]) {
  const CoupledFixture *fixture = (const CoupledFixture *)opaque;
  (void)stable_source_cell_id;
  if (fixture == NULL || reference == NULL || pressure == NULL || gradient == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *pressure = 1.0 + 2.0 * reference[0] - 3.0 * reference[1] +
              (fixture->dimension == 3U ? 0.5 * reference[2] : 0.0);
  gradient[0] = 2.0;
  gradient[1] = -3.0;
  gradient[2] = fixture->dimension == 3U ? 0.5 : 0.0;
  return SS_OK;
}

static SsStatus coupled_source_state(void *opaque, SsTransferSourceState *state) {
  const CoupledFixture *fixture = (const CoupledFixture *)opaque;
  if (fixture == NULL || state == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  state->accepted_generation = fixture->generation;
  state->mesh_generation = 401U + fixture->dimension;
  state->coordinate_generation = 501U + fixture->dimension;
  strcpy(state->mesh_identity, fixture->mesh_identity);
  strcpy(state->field_space_identity, "coupled-pressure-p1-v1");
  strcpy(state->model_identity, "coupled-hydraulic-affine-v1");
  strcpy(state->pressure_unit, "kPa");
  return SS_OK;
}

static int coupled_pack_targets(CoupledFixture *fixture) {
  static const double square_coordinates[8] = {
      0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
  };
  static const size_t square_cells[6] = {
      0U, 1U, 2U, 1U, 3U, 2U,
  };
  static const SsEntityId square_cell_ids[2] = {
      UINT64_C(0x520000000001),
      UINT64_C(0x520000000002),
  };
  static const double tetra_coordinates[15] = {
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
  };
  static const size_t tetra_cells[8] = {
      0U, 1U, 2U, 3U, 1U, 2U, 3U, 4U,
  };
  static const SsEntityId tetra_cell_ids[2] = {
      UINT64_C(0x620000000001),
      UINT64_C(0x620000000002),
  };
  const double *coordinates =
      fixture->dimension == 2U ? square_coordinates : tetra_coordinates;
  const size_t *cells = fixture->dimension == 2U ? square_cells : tetra_cells;
  const SsEntityId *cell_ids =
      fixture->dimension == 2U ? square_cell_ids : tetra_cell_ids;
  const SsFeDescriptor *descriptor = NULL;
  SsFeQuadratureRequest request;
  SsFeQuadraturePlan plan;
  const SsFeGeometryPolicy geometry_policy = {1.0e-14, 1.0e14};
  size_t cell;
  memset(&plan, 0, sizeof(plan));
  if (ss_fe_descriptor_find(fixture->dimension, 1U, &descriptor) != SS_OK) {
    return 1;
  }
  memset(&request, 0, sizeof(request));
  request.parent_dimension = fixture->dimension;
  request.domain = SS_FE_QUADRATURE_VOLUME;
  request.operation = SS_FE_OPERATION_MECHANICS_STIFFNESS;
  request.field_order = 1U;
  request.coordinate_order = 1U;
  request.mode = SS_FE_QUADRATURE_MODE_DEFAULT;
  if (ss_fe_quadrature_plan(&request, &plan) != SS_OK) {
    return 1;
  }
  for (cell = 0U; cell < 2U; ++cell) {
    double local_coordinates[12] = {0.0};
    double minimum_edge = 0.0;
    double epsilon = 0.0;
    size_t local_node;
    size_t quadrature_index;
    for (local_node = 0U; local_node < descriptor->node_count; ++local_node) {
      const size_t global_node = cells[cell * descriptor->node_count + local_node];
      uint32_t coordinate;
      for (coordinate = 0U; coordinate < fixture->dimension; ++coordinate) {
        local_coordinates[local_node * fixture->dimension + coordinate] =
            coordinates[global_node * fixture->dimension + coordinate];
      }
    }
    if (ss_hydraulic_cell_epsilon(fixture->dimension, local_coordinates, 9.81,
                                  &minimum_edge, &epsilon) != SS_OK) {
      return 1;
    }
    for (quadrature_index = 0U; quadrature_index < plan.selected_rule->point_count;
         ++quadrature_index) {
      const double *reference =
          plan.selected_rule->points + quadrature_index * fixture->dimension;
      SsFeGeometry geometry;
      SsTransferTargetPoint *target;
      if (fixture->point_count >=
          sizeof(fixture->points) / sizeof(fixture->points[0])) {
        return 1;
      }
      memset(&geometry, 0, sizeof(geometry));
      if (ss_fe_geometry_eval(descriptor, local_coordinates, reference,
                              &geometry_policy, &geometry) != SS_OK) {
        return 1;
      }
      target = &fixture->points[fixture->point_count];
      target->stable_point_id =
          UINT64_C(0x710000000000) +
          (SsEntityId)(fixture->dimension * 100U + fixture->point_count + 1U);
      target->stable_target_cell_id = cell_ids[cell];
      target->quadrature_index = (uint32_t)quadrature_index;
      target->dimension = fixture->dimension;
      memcpy(target->physical, geometry.physical,
             fixture->dimension * sizeof(*target->physical));
      target->physical_weight =
          plan.selected_rule->weights[quadrature_index] * geometry.determinant;
      target->cell_epsilon = epsilon;
      target->saturated_unit_weight = 20.0;
      target->unsaturated_unit_weight = 18.0;
      ++fixture->point_count;
    }
  }
  return 0;
}

int coupled_fixture_create(MPI_Comm communicator, uint32_t dimension,
                           int requested_threads, CoupledFixture **fixture) {
  CoupledFixture *created;
  SsCouplingPlanSpec spec;
  if (fixture != NULL) {
    *fixture = NULL;
  }
  if (fixture == NULL || (dimension != 2U && dimension != 3U) ||
      requested_threads <= 0 || communicator == MPI_COMM_NULL) {
    return 1;
  }
  created = calloc(1U, sizeof(*created));
  if (created == NULL) {
    return 1;
  }
  created->dimension = dimension;
  created->generation = 61U + dimension;
  strcpy(created->mesh_identity,
         dimension == 2U ? "mechanics-square-p1-v1" : "mechanics-tetra-p1-v1");
  if (coupled_pack_targets(created) != 0) {
    free(created);
    return 1;
  }
  memset(&spec, 0, sizeof(spec));
  spec.dimension = dimension;
  spec.upward_axis = 1U;
  spec.transfer_type = SS_COUPLING_TRANSFER_SAME_MESH;
  spec.requested_threads = requested_threads;
  spec.openmp_work_threshold = 1U;
  strcpy(spec.target_mesh_identity, created->mesh_identity);
  strcpy(spec.target_space_identity, "coupled-mechanics-p1-v1");
  strcpy(spec.target_quadrature_identity, "coupled-mechanics-quadrature-v1");
  spec.geometry.context = created;
  spec.geometry.target_count = coupled_target_count;
  spec.geometry.target_at = coupled_target_at;
  spec.geometry.locate = coupled_locate;
  spec.geometry.evaluate_source = coupled_evaluate;
  spec.geometry.source_state = coupled_source_state;
  if (ss_coupling_plan_create(communicator, &spec, &created->plan) != SS_OK ||
      ss_coupling_field_snapshot_build(created->plan,
                                       SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT,
                                       &created->snapshot) != SS_OK) {
    (void)ss_coupling_field_snapshot_destroy(&created->snapshot);
    (void)ss_coupling_plan_destroy(&created->plan);
    free(created);
    return 1;
  }
  *fixture = created;
  return 0;
}

const SsCouplingFieldSnapshot *coupled_fixture_snapshot(const CoupledFixture *fixture) {
  return fixture == NULL ? NULL : fixture->snapshot;
}

int coupled_fixture_destroy(CoupledFixture **fixture) {
  int result = 0;
  if (fixture == NULL) {
    return 1;
  }
  if (*fixture != NULL) {
    if (ss_coupling_field_snapshot_destroy(&(*fixture)->snapshot) != SS_OK) {
      result = 1;
    }
    if (ss_coupling_plan_destroy(&(*fixture)->plan) != SS_OK) {
      result = 1;
    }
    free(*fixture);
  }
  *fixture = NULL;
  return result;
}
