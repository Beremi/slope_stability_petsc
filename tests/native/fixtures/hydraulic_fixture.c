/* Compact and case-slice seepage problems of the hydraulic tests.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "hydraulic_fixture.h"

#include "slope_internal/assembly/assembly.h"
#include "slope_internal/hydraulic/hydraulic.h"
#include "slope_internal/mesh/mesh.h"

#include <petscmat.h>
#include <petscvec.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HYDRAULIC_FIXTURE_CHECK(expression)                                            \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: hydraulic fixture failed: %s\n", __FILE__, __LINE__,     \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static const SsEntityId fan2_vertex_ids[] = {
    UINT64_C(0x910000000001), UINT64_C(0x910000000002), UINT64_C(0x910000000003),
    UINT64_C(0x910000000004), UINT64_C(0x910000000005),
};
static const double fan2_coordinates[] = {
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5,
};
static const uint64_t fan2_cells[] = {
    0U, 1U, 4U, 1U, 2U, 4U, 2U, 3U, 4U, 3U, 0U, 4U,
};
static const SsEntityId fan2_cell_ids[] = {
    UINT64_C(0x920000000001),
    UINT64_C(0x920000000002),
    UINT64_C(0x920000000003),
    UINT64_C(0x920000000004),
};
static const uint64_t fan2_facets[] = {
    0U, 1U, 1U, 2U, 2U, 3U, 3U, 0U,
};
static const SsEntityId fan2_facet_ids[] = {
    UINT64_C(0x930000000001),
    UINT64_C(0x930000000002),
    UINT64_C(0x930000000003),
    UINT64_C(0x930000000004),
};
static const SsMeshIntermediateEntity fan2_intermediate[] = {
    {1U,
     UINT64_C(0x930000000001),
     {UINT64_C(0x910000000001), UINT64_C(0x910000000002), 0U}},
    {1U,
     UINT64_C(0x930000000002),
     {UINT64_C(0x910000000002), UINT64_C(0x910000000003), 0U}},
    {1U,
     UINT64_C(0x930000000003),
     {UINT64_C(0x910000000003), UINT64_C(0x910000000004), 0U}},
    {1U,
     UINT64_C(0x930000000004),
     {UINT64_C(0x910000000001), UINT64_C(0x910000000004), 0U}},
    {1U,
     UINT64_C(0x940000000001),
     {UINT64_C(0x910000000001), UINT64_C(0x910000000005), 0U}},
    {1U,
     UINT64_C(0x940000000002),
     {UINT64_C(0x910000000002), UINT64_C(0x910000000005), 0U}},
    {1U,
     UINT64_C(0x940000000003),
     {UINT64_C(0x910000000003), UINT64_C(0x910000000005), 0U}},
    {1U,
     UINT64_C(0x940000000004),
     {UINT64_C(0x910000000004), UINT64_C(0x910000000005), 0U}},
};
static const SsMeshLabelAssignment fan2_labels[] = {
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x920000000001), 1U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x920000000002), 2U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x920000000003), 3U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x920000000004), 4U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x930000000001), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x930000000002), 2U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x930000000003), 3U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x930000000004), 4U},
};

static const SsEntityId fan3_vertex_ids[] = {
    UINT64_C(0xa10000000001), UINT64_C(0xa10000000002), UINT64_C(0xa10000000003),
    UINT64_C(0xa10000000004), UINT64_C(0xa10000000005),
};
static const double fan3_coordinates[] = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25, 0.25, 0.25,
};
static const uint64_t fan3_cells[] = {
    0U, 1U, 2U, 4U, 0U, 3U, 1U, 4U, 0U, 2U, 3U, 4U, 1U, 3U, 2U, 4U,
};
static const SsEntityId fan3_cell_ids[] = {
    UINT64_C(0xa20000000001),
    UINT64_C(0xa20000000002),
    UINT64_C(0xa20000000003),
    UINT64_C(0xa20000000004),
};
static const uint64_t fan3_facets[] = {
    0U, 1U, 2U, 0U, 1U, 3U, 0U, 2U, 3U, 1U, 2U, 3U,
};
static const SsEntityId fan3_facet_ids[] = {
    UINT64_C(0xa30000000001),
    UINT64_C(0xa30000000002),
    UINT64_C(0xa30000000003),
    UINT64_C(0xa30000000004),
};
static const SsMeshIntermediateEntity fan3_intermediate[] = {
    {1U,
     UINT64_C(0xa40000000001),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000002), 0U}},
    {1U,
     UINT64_C(0xa40000000002),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000003), 0U}},
    {1U,
     UINT64_C(0xa40000000003),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000004), 0U}},
    {1U,
     UINT64_C(0xa40000000004),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000005), 0U}},
    {1U,
     UINT64_C(0xa40000000005),
     {UINT64_C(0xa10000000002), UINT64_C(0xa10000000003), 0U}},
    {1U,
     UINT64_C(0xa40000000006),
     {UINT64_C(0xa10000000002), UINT64_C(0xa10000000004), 0U}},
    {1U,
     UINT64_C(0xa40000000007),
     {UINT64_C(0xa10000000002), UINT64_C(0xa10000000005), 0U}},
    {1U,
     UINT64_C(0xa40000000008),
     {UINT64_C(0xa10000000003), UINT64_C(0xa10000000004), 0U}},
    {1U,
     UINT64_C(0xa40000000009),
     {UINT64_C(0xa10000000003), UINT64_C(0xa10000000005), 0U}},
    {1U,
     UINT64_C(0xa4000000000a),
     {UINT64_C(0xa10000000004), UINT64_C(0xa10000000005), 0U}},
    {2U,
     UINT64_C(0xa30000000001),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000002), UINT64_C(0xa10000000003)}},
    {2U,
     UINT64_C(0xa30000000002),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000002), UINT64_C(0xa10000000004)}},
    {2U,
     UINT64_C(0xa30000000003),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000003), UINT64_C(0xa10000000004)}},
    {2U,
     UINT64_C(0xa30000000004),
     {UINT64_C(0xa10000000002), UINT64_C(0xa10000000003), UINT64_C(0xa10000000004)}},
    {2U,
     UINT64_C(0xa50000000001),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000002), UINT64_C(0xa10000000005)}},
    {2U,
     UINT64_C(0xa50000000002),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000003), UINT64_C(0xa10000000005)}},
    {2U,
     UINT64_C(0xa50000000003),
     {UINT64_C(0xa10000000001), UINT64_C(0xa10000000004), UINT64_C(0xa10000000005)}},
    {2U,
     UINT64_C(0xa50000000004),
     {UINT64_C(0xa10000000002), UINT64_C(0xa10000000003), UINT64_C(0xa10000000005)}},
    {2U,
     UINT64_C(0xa50000000005),
     {UINT64_C(0xa10000000002), UINT64_C(0xa10000000004), UINT64_C(0xa10000000005)}},
    {2U,
     UINT64_C(0xa50000000006),
     {UINT64_C(0xa10000000003), UINT64_C(0xa10000000004), UINT64_C(0xa10000000005)}},
};
static const SsMeshLabelAssignment fan3_labels[] = {
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0xa20000000001), 1U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0xa20000000002), 2U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0xa20000000003), 3U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0xa20000000004), 4U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0xa30000000001), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0xa30000000002), 2U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0xa30000000003), 3U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0xa30000000004), 4U},
};

typedef struct {
  uint32_t dimension;
  SsMesh *mesh;
  SsFieldSpace *space;
  SsAssemblyPlan *plan;
  SsOperator *operator_handle;
  SsHydraulicOperatorModel *model;
  SsHydraulicCellCoefficients cell_coefficients[4];
  double conductivity[4];
  double epsilon[4];
  SsScalarOperatorSpec scalar_spec;
  SsOperatorSpec operator_spec;
  Vec reference_load;
  Vec parent;
  Vec accepted;
  Vec residual;
  Mat elastic;
  Mat final_jacobian;
  SsNonlinearProblem nonlinear;
  SsNonlinearPolicy nonlinear_policy;
} HydraulicFixture;

typedef enum {
  HYDRAULIC_FIXTURE_COMPACT = 0,
  HYDRAULIC_FIXTURE_CASE_007 = 1,
  HYDRAULIC_FIXTURE_CASE_009 = 2
} HydraulicFixtureKind;

static SsMeshInput hydraulic_mesh_input(uint32_t dimension) {
  if (dimension == 2U) {
    const SsMeshInput input = {
        .dimension = 2U,
        .global_vertex_count = 5U,
        .global_cell_count = 4U,
        .global_boundary_facet_count = 4U,
        .global_intermediate_entity_count =
            sizeof(fan2_intermediate) / sizeof(fan2_intermediate[0]),
        .global_coordinate_node_count = 5U,
        .vertex_count = 5U,
        .cell_count = 4U,
        .boundary_facet_count = 4U,
        .vertex_ids = fan2_vertex_ids,
        .vertex_coordinates = fan2_coordinates,
        .cell_ids = fan2_cell_ids,
        .cells = fan2_cells,
        .boundary_facet_ids = fan2_facet_ids,
        .boundary_facets = fan2_facets,
        .intermediate_entity_count =
            sizeof(fan2_intermediate) / sizeof(fan2_intermediate[0]),
        .intermediate_entities = fan2_intermediate,
        .label_count = sizeof(fan2_labels) / sizeof(fan2_labels[0]),
        .labels = fan2_labels,
        .coordinate_order = 1U,
        .coordinate_node_count = 5U,
        .coordinate_node_ids = fan2_vertex_ids,
        .coordinate_coordinates = fan2_coordinates,
        .cell_coordinate_nodes = fan2_cells,
        .absolute_determinant_tolerance = 1.0e-12,
        .relative_determinant_tolerance = 1.0e-12,
    };
    return input;
  }
  {
    const SsMeshInput input = {
        .dimension = 3U,
        .global_vertex_count = 5U,
        .global_cell_count = 4U,
        .global_boundary_facet_count = 4U,
        .global_intermediate_entity_count =
            sizeof(fan3_intermediate) / sizeof(fan3_intermediate[0]),
        .global_coordinate_node_count = 5U,
        .vertex_count = 5U,
        .cell_count = 4U,
        .boundary_facet_count = 4U,
        .vertex_ids = fan3_vertex_ids,
        .vertex_coordinates = fan3_coordinates,
        .cell_ids = fan3_cell_ids,
        .cells = fan3_cells,
        .boundary_facet_ids = fan3_facet_ids,
        .boundary_facets = fan3_facets,
        .intermediate_entity_count =
            sizeof(fan3_intermediate) / sizeof(fan3_intermediate[0]),
        .intermediate_entities = fan3_intermediate,
        .label_count = sizeof(fan3_labels) / sizeof(fan3_labels[0]),
        .labels = fan3_labels,
        .coordinate_order = 1U,
        .coordinate_node_count = 5U,
        .coordinate_node_ids = fan3_vertex_ids,
        .coordinate_coordinates = fan3_coordinates,
        .cell_coordinate_nodes = fan3_cells,
        .absolute_determinant_tolerance = 1.0e-12,
        .relative_determinant_tolerance = 1.0e-12,
    };
    return input;
  }
}

static void hydraulic_direct_policy(SsLinearPolicyChain *chain) {
  SsLinearPolicy *policy;
  memset(chain, 0, sizeof(*chain));
  chain->count = 1U;
  policy = &chain->policies[0];
  strcpy(policy->identity, "fixture-hydraulic-direct");
  strcpy(policy->options_prefix, "fixture_hyd_");
  policy->operator_class = SS_LINEAR_OPERATOR_DIRECT_VALIDATION;
  policy->method = SS_LINEAR_METHOD_PREONLY;
  policy->pc = SS_LINEAR_PC_LU;
  policy->side = SS_LINEAR_SIDE_LEFT;
  policy->rtol = 1.0e-12;
  policy->atol = 1.0e-14;
  policy->dtol = 1.0e8;
  policy->maximum_iterations = 1;
  policy->restart = 2;
  policy->vector_memory_cap_bytes = 8U * 1024U * 1024U;
}

static SsStatus hydraulic_evaluate(void *opaque, Vec state, PetscReal factor,
                                   Vec residual, Mat tangent, PetscReal *functional) {
  HydraulicFixture *fixture = (HydraulicFixture *)opaque;
  const SsOperatorState operator_state = {
      .field = state,
      .reduction_factor = 1.0,
      .external_load_factor = 1.0,
      .cancelled = 0U,
  };
  SsErrorRecord error = {0};
  (void)factor;
  (void)functional;
  if (fixture == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return tangent == NULL
             ? ss_operator_residual(fixture->operator_handle, &operator_state, residual,
                                    &error)
             : ss_operator_residual_jacobian(fixture->operator_handle, &operator_state,
                                             residual, tangent, &error);
}

static SsStatus hydraulic_admissibility(void *opaque, Vec state, PetscReal factor,
                                        int *admissible) {
  PetscReal norm = 0.0;
  (void)opaque;
  if (admissible == NULL || !isfinite((double)factor) ||
      VecNorm(state, NORM_INFINITY, &norm) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *admissible = isfinite((double)norm) != 0;
  return SS_OK;
}

static int hydraulic_case_007_prescribed(const double *coordinates,
                                         double *prescribed) {
  SsHydraulicHeadModel head;
  size_t vertex;
  memset(&head, 0, sizeof(head));
  head.kind = SS_HYDRAULIC_HEAD_PIECEWISE_LINEAR_LEVEL;
  head.dimension = 2U;
  head.upward_axis = 1U;
  head.piecewise_axis = 0U;
  head.gamma_water = 9.81;
  head.piecewise_point_count = 2U;
  head.piecewise_coordinates[0] = 0.0;
  head.piecewise_coordinates[1] =
      15.0 + (1.0 - 1.0 / 12.25) * (12.25 / tan(26.6 * acos(-1.0) / 180.0));
  head.piecewise_levels[0] = 18.25;
  head.piecewise_levels[1] = 9.0;
  strcpy(head.identity, "CASE-007-head_support-piecewise-v1");
  for (vertex = 0U; vertex < 4U; ++vertex) {
    if (ss_hydraulic_head_evaluate(&head, coordinates + vertex * 2U,
                                   &prescribed[vertex]) != SS_OK) {
      return 1;
    }
  }
  return 0;
}

static int hydraulic_case_009_prescribed(const double *coordinates,
                                         double *prescribed) {
  static const SsHydraulicHeadKind kinds[4] = {
      SS_HYDRAULIC_HEAD_DRY,
      SS_HYDRAULIC_HEAD_CONSTANT_LEVEL,
      SS_HYDRAULIC_HEAD_CONSTANT_LEVEL,
      SS_HYDRAULIC_HEAD_DRY,
  };
  static const double levels[4] = {0.0, 55.0, 35.0, 0.0};
  static const char *const identities[4] = {
      "CASE-009-head_dry-a-v1",
      "CASE-009-head_porous-55-v1",
      "CASE-009-head_free-35-v1",
      "CASE-009-head_dry-b-v1",
  };
  size_t vertex;
  for (vertex = 0U; vertex < 4U; ++vertex) {
    SsHydraulicHeadModel head;
    memset(&head, 0, sizeof(head));
    head.kind = kinds[vertex];
    head.dimension = 3U;
    head.upward_axis = 1U;
    head.gamma_water = 9.81;
    head.level = levels[vertex];
    strcpy(head.identity, identities[vertex]);
    if (ss_hydraulic_head_evaluate(&head, coordinates + vertex * 3U,
                                   &prescribed[vertex]) != SS_OK) {
      return 1;
    }
  }
  return 0;
}

static int hydraulic_fixture_create(uint32_t dimension, uint32_t field_order,
                                    HydraulicFixtureKind fixture_kind,
                                    int requested_threads, HydraulicFixture *fixture) {
  static const double compact_pressure_2d[4] = {1.0, 1.0, 0.0, 0.0};
  static const double compact_pressure_3d[4] = {0.3, 0.3, 0.0, 0.3};
  static const size_t outer_edge_indices[6] = {0U, 1U, 2U, 4U, 5U, 7U};
  static const size_t outer_edge_vertices[6][2] = {
      {0U, 1U}, {0U, 2U}, {0U, 3U}, {1U, 2U}, {1U, 3U}, {2U, 3U},
  };
  const SsEntityId *vertex_ids = dimension == 2U ? fan2_vertex_ids : fan3_vertex_ids;
  const SsEntityId *cell_ids = dimension == 2U ? fan2_cell_ids : fan3_cell_ids;
  const double *coordinates = dimension == 2U ? fan2_coordinates : fan3_coordinates;
  const uint64_t *cells = dimension == 2U ? fan2_cells : fan3_cells;
  double prescribed[4] = {0.0, 0.0, 0.0, 0.0};
  const size_t cell_vertex_count = (size_t)dimension + 1U;
  SsStrongConstraint constraints[10];
  size_t constraint_count = 4U;
  SsFieldSpec field_spec;
  SsHydraulicOperatorModelSpec model_spec;
  SsHydraulicOperatorModelSummary model_summary;
  SsMeshInput mesh_input = hydraulic_mesh_input(dimension);
  SsMeshError mesh_error = {0};
  SsErrorRecord error = {0};
  SsOperatorState operator_state;
  size_t cell;
  memset(fixture, 0, sizeof(*fixture));
  fixture->dimension = dimension;
  HYDRAULIC_FIXTURE_CHECK(field_order == 1U || (dimension == 3U && field_order == 2U));
  if (fixture_kind == HYDRAULIC_FIXTURE_COMPACT) {
    memcpy(prescribed, dimension == 2U ? compact_pressure_2d : compact_pressure_3d,
           sizeof(prescribed));
  } else if (fixture_kind == HYDRAULIC_FIXTURE_CASE_007 && dimension == 2U &&
             field_order == 1U) {
    HYDRAULIC_FIXTURE_CHECK(hydraulic_case_007_prescribed(coordinates, prescribed) ==
                            0);
  } else if (fixture_kind == HYDRAULIC_FIXTURE_CASE_009 && dimension == 3U &&
             field_order == 2U) {
    HYDRAULIC_FIXTURE_CHECK(hydraulic_case_009_prescribed(coordinates, prescribed) ==
                            0);
  } else {
    return 1;
  }
  for (cell = 0U; cell < 4U; ++cell) {
    double vertex_coordinates[12] = {0.0};
    size_t local_vertex;
    fixture->conductivity[cell] =
        fixture_kind == HYDRAULIC_FIXTURE_COMPACT ? (double)(cell + 1U) : 1.0;
    for (local_vertex = 0U; local_vertex < cell_vertex_count; ++local_vertex) {
      const size_t global_vertex =
          (size_t)cells[cell * cell_vertex_count + local_vertex];
      uint32_t coordinate;
      for (coordinate = 0U; coordinate < dimension; ++coordinate) {
        vertex_coordinates[local_vertex * dimension + coordinate] =
            coordinates[global_vertex * dimension + coordinate];
      }
    }
    HYDRAULIC_FIXTURE_CHECK(
        ss_hydraulic_cell_epsilon(dimension, vertex_coordinates, 9.81, &(double){0.0},
                                  &fixture->epsilon[cell]) == SS_OK);
    fixture->cell_coefficients[cell] = (SsHydraulicCellCoefficients){
        cell_ids[cell],
        fixture->epsilon[cell],
        1U,
        &fixture->conductivity[cell],
    };
  }
  memset(&model_spec, 0, sizeof(model_spec));
  model_spec.dimension = dimension;
  model_spec.upward_axis = 1U;
  model_spec.gamma_water = 9.81;
  model_spec.cell_count = 4U;
  model_spec.cells = fixture->cell_coefficients;
  if (fixture_kind == HYDRAULIC_FIXTURE_CASE_007) {
    strcpy(model_spec.model_identity, "CASE-007-reduced-seepage-slice-v1");
  } else if (fixture_kind == HYDRAULIC_FIXTURE_CASE_009) {
    strcpy(model_spec.model_identity, "CASE-009-reduced-seepage-slice-v1");
  } else {
    strcpy(model_spec.model_identity,
           dimension == 2U ? "HYD-C2-REFERENCE-v1" : "HYD-C3-CORRECTED-v1");
  }
  strcpy(model_spec.pressure_unit, "kPa");
  strcpy(model_spec.coordinate_unit, "m");
  strcpy(model_spec.conductivity_unit,
         fixture_kind == HYDRAULIC_FIXTURE_COMPACT ? "relative-steady" : "m/s");
  HYDRAULIC_FIXTURE_CHECK(
      ss_hydraulic_operator_model_create(&model_spec, &fixture->model) == SS_OK);
  HYDRAULIC_FIXTURE_CHECK(
      ss_hydraulic_operator_model_summary(fixture->model, &model_summary) == SS_OK);
  HYDRAULIC_FIXTURE_CHECK(model_summary.cell_count == 4U);
  for (cell = 0U; cell < 4U; ++cell) {
    constraints[cell] = (SsStrongConstraint){
        vertex_ids[cell],
        0U,
        0U,
        prescribed[cell],
    };
  }
  if (field_order == 2U) {
    size_t edge;
    for (edge = 0U; edge < 6U; ++edge) {
      const size_t first = outer_edge_vertices[edge][0];
      const size_t second = outer_edge_vertices[edge][1];
      constraints[constraint_count++] = (SsStrongConstraint){
          fan3_intermediate[outer_edge_indices[edge]].stable_entity_id,
          0U,
          0U,
          0.5 * (prescribed[first] + prescribed[second]),
      };
    }
  }
  field_spec = (SsFieldSpec){
      field_order,
      1U,
      constraint_count,
      constraints,
  };
  HYDRAULIC_FIXTURE_CHECK(ss_mesh_create(PETSC_COMM_SELF, &mesh_input, &fixture->mesh,
                                         &mesh_error) == SS_OK);
  HYDRAULIC_FIXTURE_CHECK(
      ss_mesh_field_space_create(fixture->mesh, &field_spec, &fixture->space) == SS_OK);
  HYDRAULIC_FIXTURE_CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, fixture->space,
                                                  &fixture->plan, &error) == SS_OK);
  fixture->scalar_spec = (SsScalarOperatorSpec){
      ss_hydraulic_operator_point_kernel,
      fixture->model,
  };
  fixture->operator_spec = (SsOperatorSpec){
      .physics = SS_ASSEMBLY_SCALAR_FLUX,
      .determinism = SS_ASSEMBLY_QUALIFICATION,
      .openmp_work_threshold = 1U,
      .requested_threads = requested_threads,
      .scalar = &fixture->scalar_spec,
  };
  HYDRAULIC_FIXTURE_CHECK(ss_operator_create(fixture->plan, &fixture->operator_spec,
                                             &fixture->operator_handle,
                                             &error) == SS_OK);
  HYDRAULIC_FIXTURE_CHECK(
      ss_assembly_plan_create_vector(fixture->plan, &fixture->reference_load) ==
          SS_OK &&
      ss_assembly_plan_create_vector(fixture->plan, &fixture->parent) == SS_OK &&
      ss_assembly_plan_create_vector(fixture->plan, &fixture->accepted) == SS_OK &&
      ss_assembly_plan_create_vector(fixture->plan, &fixture->residual) == SS_OK &&
      ss_assembly_plan_create_matrix(fixture->plan, &fixture->elastic) == SS_OK &&
      ss_assembly_plan_create_matrix(fixture->plan, &fixture->final_jacobian) == SS_OK);
  HYDRAULIC_FIXTURE_CHECK(VecSet(fixture->reference_load, 1.0) == PETSC_SUCCESS);
  HYDRAULIC_FIXTURE_CHECK(VecSet(fixture->parent, dimension == 2U ? 0.0 : 0.3) ==
                          PETSC_SUCCESS);
  HYDRAULIC_FIXTURE_CHECK(VecSet(fixture->accepted, 77.0) == PETSC_SUCCESS);
  operator_state = (SsOperatorState){
      .field = fixture->parent,
      .reduction_factor = 1.0,
      .external_load_factor = 1.0,
      .cancelled = 0U,
  };
  HYDRAULIC_FIXTURE_CHECK(ss_operator_jacobian(fixture->operator_handle,
                                               &operator_state, fixture->elastic,
                                               &error) == SS_OK);
  fixture->nonlinear.communicator = PETSC_COMM_SELF;
  fixture->nonlinear.reference_load = fixture->reference_load;
  fixture->nonlinear.elastic_reference = fixture->elastic;
  fixture->nonlinear.operations.evaluate = hydraulic_evaluate;
  fixture->nonlinear.operations.admissibility = hydraulic_admissibility;
  fixture->nonlinear.context = fixture;
  fixture->nonlinear.linear_properties.finite_action = 1;
  fixture->nonlinear.linear_properties.nonlinear_hydraulic = 1;
  hydraulic_direct_policy(&fixture->nonlinear.linear_policies);
  fixture->nonlinear.structure_generation = 301U + dimension;
  fixture->nonlinear.first_value_generation = 311U + dimension;
  fixture->nonlinear.transaction_generation = 321U + dimension;
  fixture->nonlinear.accepted_generation = 331U + dimension;
  fixture->nonlinear_policy = (SsNonlinearPolicy){
      .tolerance = 1.0e-11,
      .stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_RESIDUAL,
      .stopping_tolerance = 1.0e-11,
      .maximum_iterations = 16,
      .maximum_line_search_trials = 16,
      .minimum_alpha = 1.0 / 65536.0,
      .regularization_minimum = 1.0e-8,
      .singular_relative_tolerance = 1.0e-12,
      .relaxed_factor = 10.0,
      .allow_relaxed = 0,
      .line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE,
  };
  return 0;
}

static int hydraulic_fixture_destroy(HydraulicFixture *fixture) {
  int result = 0;
  if (MatDestroy(&fixture->final_jacobian) != PETSC_SUCCESS ||
      MatDestroy(&fixture->elastic) != PETSC_SUCCESS ||
      VecDestroy(&fixture->residual) != PETSC_SUCCESS ||
      VecDestroy(&fixture->accepted) != PETSC_SUCCESS ||
      VecDestroy(&fixture->parent) != PETSC_SUCCESS ||
      VecDestroy(&fixture->reference_load) != PETSC_SUCCESS ||
      ss_operator_destroy(&fixture->operator_handle) != SS_OK ||
      ss_assembly_plan_destroy(&fixture->plan) != SS_OK ||
      ss_field_space_destroy(&fixture->space) != SS_OK ||
      ss_mesh_destroy(&fixture->mesh) != SS_OK ||
      ss_hydraulic_operator_model_destroy(&fixture->model) != SS_OK) {
    result = 1;
  }
  return result;
}

static int hydraulic_run_profile(uint32_t dimension, uint32_t field_order,
                                 HydraulicFixtureKind fixture_kind,
                                 int requested_threads,
                                 HydraulicFixtureProfile *profile) {
  HydraulicFixture fixture;
  SsHydraulicOperatorModelSummary summary;
  SsNonlinearSolveRecord failure_record;
  SsNonlinearSolveRecord accepted_record;
  SsAssemblyDiagnostics diagnostics;
  SsOperatorState operator_state;
  const PetscInt index = 0;
  PetscScalar accepted_value = 0.0;
  PetscScalar failure_value = 0.0;
  PetscScalar jacobian_value = 0.0;
  PetscReal accepted_norm = 0.0;
  PetscReal residual_norm = 0.0;
  PetscInt free_dof_count = 0;
  if ((dimension != 2U && dimension != 3U) ||
      (field_order != 1U && !(dimension == 3U && field_order == 2U)) ||
      requested_threads <= 0 || profile == NULL) {
    return 1;
  }
  memset(profile, 0, sizeof(*profile));
  HYDRAULIC_FIXTURE_CHECK(hydraulic_fixture_create(dimension, field_order, fixture_kind,
                                                   requested_threads, &fixture) == 0);
  HYDRAULIC_FIXTURE_CHECK(ss_operator_inject_failure(fixture.operator_handle,
                                                     SS_ASSEMBLY_FAIL_PRECOMMIT, 0,
                                                     0U) == SS_OK);
  memset(&failure_record, 0, sizeof(failure_record));
  HYDRAULIC_FIXTURE_CHECK(ss_nonlinear_fixed_solve(&fixture.nonlinear,
                                                   &fixture.nonlinear_policy, 1.0,
                                                   fixture.parent, fixture.accepted,
                                                   &failure_record) == SS_OK);
  HYDRAULIC_FIXTURE_CHECK(VecGetValues(fixture.accepted, 1, &index, &failure_value) ==
                          PETSC_SUCCESS);
  HYDRAULIC_FIXTURE_CHECK(failure_record.transaction_committed == 0);
  HYDRAULIC_FIXTURE_CHECK(failure_record.transaction_rolled_back == 1);
  HYDRAULIC_FIXTURE_CHECK(failure_record.parent_unchanged_on_failure == 1);
  HYDRAULIC_FIXTURE_CHECK((double)failure_value == 77.0);
  memset(&accepted_record, 0, sizeof(accepted_record));
  HYDRAULIC_FIXTURE_CHECK(ss_nonlinear_fixed_solve(&fixture.nonlinear,
                                                   &fixture.nonlinear_policy, 1.0,
                                                   fixture.parent, fixture.accepted,
                                                   &accepted_record) == SS_OK);
  HYDRAULIC_FIXTURE_CHECK(accepted_record.outcome == SS_NONLINEAR_CONVERGED &&
                          accepted_record.transaction_committed == 1 &&
                          accepted_record.transaction_rolled_back == 0);
  operator_state = (SsOperatorState){
      .field = fixture.accepted,
      .reduction_factor = 1.0,
      .external_load_factor = 1.0,
      .cancelled = 0U,
  };
  HYDRAULIC_FIXTURE_CHECK(
      ss_operator_residual_jacobian(fixture.operator_handle, &operator_state,
                                    fixture.residual, fixture.final_jacobian,
                                    &(SsErrorRecord){0}) == SS_OK);
  HYDRAULIC_FIXTURE_CHECK(VecNorm(fixture.residual, NORM_2, &residual_norm) ==
                          PETSC_SUCCESS);
  HYDRAULIC_FIXTURE_CHECK(VecNorm(fixture.accepted, NORM_2, &accepted_norm) ==
                          PETSC_SUCCESS);
  HYDRAULIC_FIXTURE_CHECK(VecGetSize(fixture.accepted, &free_dof_count) ==
                          PETSC_SUCCESS);
  HYDRAULIC_FIXTURE_CHECK(VecGetValues(fixture.accepted, 1, &index, &accepted_value) ==
                          PETSC_SUCCESS);
  HYDRAULIC_FIXTURE_CHECK(MatGetValues(fixture.final_jacobian, 1, &index, 1, &index,
                                       &jacobian_value) == PETSC_SUCCESS);
  HYDRAULIC_FIXTURE_CHECK(
      ss_hydraulic_operator_model_summary(fixture.model, &summary) == SS_OK);
  memset(&diagnostics, 0, sizeof(diagnostics));
  HYDRAULIC_FIXTURE_CHECK(
      ss_operator_diagnostics(fixture.operator_handle, &diagnostics) == SS_OK);
  profile->dimension = dimension;
  profile->field_order = field_order;
  profile->free_dof_count = free_dof_count;
  profile->accepted_free_pressure = (double)PetscRealPart(accepted_value);
  profile->accepted_pressure_norm = (double)accepted_norm;
  profile->accepted_residual_norm = (double)residual_norm;
  profile->accepted_jacobian_diagonal = (double)PetscRealPart(jacobian_value);
  profile->conductivity_minimum = summary.conductivity_minimum;
  profile->conductivity_maximum = summary.conductivity_maximum;
  profile->epsilon_minimum = summary.epsilon_minimum;
  profile->epsilon_maximum = summary.epsilon_maximum;
  profile->nonlinear_iterations = accepted_record.iterations;
  profile->nonlinear_outcome = accepted_record.outcome;
  profile->injected_failure_outcome = failure_record.outcome;
  profile->injected_failure_rolled_back = failure_record.transaction_rolled_back;
  profile->injected_failure_parent_unchanged =
      failure_record.parent_unchanged_on_failure;
  profile->accepted_transaction_committed = accepted_record.transaction_committed;
  profile->observed_threads = diagnostics.observed_threads;
  HYDRAULIC_FIXTURE_CHECK(hydraulic_fixture_destroy(&fixture) == 0);
  return 0;
}

int hydraulic_fixture_compact_profile(uint32_t dimension, int requested_threads,
                                      HydraulicFixtureProfile *profile) {
  return hydraulic_run_profile(dimension, 1U, HYDRAULIC_FIXTURE_COMPACT,
                               requested_threads, profile);
}

int hydraulic_fixture_case_slice_profile(const char *case_id, int requested_threads,
                                         HydraulicFixtureProfile *profile) {
  if (case_id == NULL) {
    return 1;
  }
  if (strcmp(case_id, "CASE-007") == 0) {
    return hydraulic_run_profile(2U, 1U, HYDRAULIC_FIXTURE_CASE_007, requested_threads,
                                 profile);
  }
  if (strcmp(case_id, "CASE-009") == 0) {
    return hydraulic_run_profile(3U, 2U, HYDRAULIC_FIXTURE_CASE_009, requested_threads,
                                 profile);
  }
  return 1;
}
