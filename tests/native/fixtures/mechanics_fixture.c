/* Compact mechanics problems (direct and PMG solver profiles, region body
 * force, coupled SSR) shared by the nonlinear and hydraulic tests.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mechanics_fixture.h"
#include "../../../src/app/solver_internal.h"

#include "slope_internal/analysis/analysis.h"
#include "slope_internal/app/solver.h"
#include "slope_internal/assembly/assembly.h"
#include "slope_internal/material/material.h"
#include "slope_internal/mesh/mesh.h"
#include "slope_internal/nonlinear/nonlinear.h"
#include "slope_internal/pmg/pmg.h"

#include <petscsys.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define FIXTURE_CHECK(expression)                                                      \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: mechanics fixture failed: %s\n", __FILE__, __LINE__,     \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static const SsEntityId square_vertex_ids[] = {
    UINT64_C(0x510000000001),
    UINT64_C(0x510000000002),
    UINT64_C(0x510000000003),
    UINT64_C(0x510000000004),
};
static const double square_coordinates[] = {
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
};
static const uint64_t square_cells[] = {
    0U, 1U, 2U, 1U, 3U, 2U,
};
static const SsEntityId square_cell_ids[] = {
    UINT64_C(0x520000000001),
    UINT64_C(0x520000000002),
};
static const uint64_t square_facets[] = {
    0U, 1U, 1U, 3U, 3U, 2U, 2U, 0U,
};
static const SsEntityId square_facet_ids[] = {
    UINT64_C(0x530000000001),
    UINT64_C(0x530000000002),
    UINT64_C(0x530000000003),
    UINT64_C(0x530000000004),
};
static const SsMeshIntermediateEntity square_intermediate_entities[] = {
    {1U,
     UINT64_C(0x530000000001),
     {UINT64_C(0x510000000001), UINT64_C(0x510000000002), 0U}},
    {1U,
     UINT64_C(0x530000000004),
     {UINT64_C(0x510000000001), UINT64_C(0x510000000003), 0U}},
    {1U,
     UINT64_C(0x540000000001),
     {UINT64_C(0x510000000002), UINT64_C(0x510000000003), 0U}},
    {1U,
     UINT64_C(0x530000000002),
     {UINT64_C(0x510000000002), UINT64_C(0x510000000004), 0U}},
    {1U,
     UINT64_C(0x530000000003),
     {UINT64_C(0x510000000003), UINT64_C(0x510000000004), 0U}},
};
static const SsMeshLabelAssignment square_labels[] = {
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x520000000001), 1U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x520000000002), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x530000000001), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x530000000002), 2U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x530000000003), 3U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x530000000004), 4U},
};
static const SsStrongConstraint square_constraints[] = {
    {UINT64_C(0x510000000001), 0U, 0U, 0.0},
    {UINT64_C(0x510000000001), 0U, 1U, 0.0},
    {UINT64_C(0x510000000002), 0U, 1U, 0.0},
};
static const SsStrongConstraint square_p2_constraints[] = {
    {UINT64_C(0x510000000001), 0U, 0U, 0.0},
    {UINT64_C(0x510000000001), 0U, 1U, 0.0},
    {UINT64_C(0x510000000002), 0U, 1U, 0.0},
    {UINT64_C(0x530000000001), 0U, 1U, 0.0},
};

static const SsEntityId tetra_vertex_ids[] = {
    UINT64_C(0x610000000001), UINT64_C(0x610000000002), UINT64_C(0x610000000003),
    UINT64_C(0x610000000004), UINT64_C(0x610000000005),
};
static const double tetra_coordinates[] = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
};
static const uint64_t tetra_cells[] = {
    0U, 1U, 2U, 3U, 1U, 2U, 3U, 4U,
};
static const SsEntityId tetra_cell_ids[] = {
    UINT64_C(0x620000000001),
    UINT64_C(0x620000000002),
};
static const uint64_t tetra_facets[] = {
    0U, 1U, 2U, 0U, 1U, 3U, 0U, 2U, 3U, 1U, 2U, 4U, 1U, 3U, 4U, 2U, 3U, 4U,
};
static const SsEntityId tetra_facet_ids[] = {
    UINT64_C(0x630000000001), UINT64_C(0x630000000002), UINT64_C(0x630000000003),
    UINT64_C(0x630000000004), UINT64_C(0x630000000005), UINT64_C(0x630000000006),
};
static const SsMeshIntermediateEntity tetra_intermediate_entities[] = {
    {1U,
     UINT64_C(0x640000000001),
     {UINT64_C(0x610000000001), UINT64_C(0x610000000002), 0U}},
    {1U,
     UINT64_C(0x640000000002),
     {UINT64_C(0x610000000001), UINT64_C(0x610000000003), 0U}},
    {1U,
     UINT64_C(0x640000000003),
     {UINT64_C(0x610000000001), UINT64_C(0x610000000004), 0U}},
    {1U,
     UINT64_C(0x640000000004),
     {UINT64_C(0x610000000002), UINT64_C(0x610000000003), 0U}},
    {1U,
     UINT64_C(0x640000000005),
     {UINT64_C(0x610000000002), UINT64_C(0x610000000004), 0U}},
    {1U,
     UINT64_C(0x640000000006),
     {UINT64_C(0x610000000003), UINT64_C(0x610000000004), 0U}},
    {1U,
     UINT64_C(0x640000000007),
     {UINT64_C(0x610000000002), UINT64_C(0x610000000005), 0U}},
    {1U,
     UINT64_C(0x640000000008),
     {UINT64_C(0x610000000003), UINT64_C(0x610000000005), 0U}},
    {1U,
     UINT64_C(0x640000000009),
     {UINT64_C(0x610000000004), UINT64_C(0x610000000005), 0U}},
    {2U,
     UINT64_C(0x630000000001),
     {UINT64_C(0x610000000001), UINT64_C(0x610000000002), UINT64_C(0x610000000003)}},
    {2U,
     UINT64_C(0x630000000002),
     {UINT64_C(0x610000000001), UINT64_C(0x610000000002), UINT64_C(0x610000000004)}},
    {2U,
     UINT64_C(0x630000000003),
     {UINT64_C(0x610000000001), UINT64_C(0x610000000003), UINT64_C(0x610000000004)}},
    {2U,
     UINT64_C(0x650000000001),
     {UINT64_C(0x610000000002), UINT64_C(0x610000000003), UINT64_C(0x610000000004)}},
    {2U,
     UINT64_C(0x630000000004),
     {UINT64_C(0x610000000002), UINT64_C(0x610000000003), UINT64_C(0x610000000005)}},
    {2U,
     UINT64_C(0x630000000005),
     {UINT64_C(0x610000000002), UINT64_C(0x610000000004), UINT64_C(0x610000000005)}},
    {2U,
     UINT64_C(0x630000000006),
     {UINT64_C(0x610000000003), UINT64_C(0x610000000004), UINT64_C(0x610000000005)}},
};
static const SsMeshLabelAssignment tetra_labels[] = {
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x620000000001), 1U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x620000000002), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x630000000001), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x630000000002), 2U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x630000000003), 3U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x630000000004), 4U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x630000000005), 5U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x630000000006), 6U},
};
static const SsStrongConstraint tetra_constraints[] = {
    {UINT64_C(0x610000000001), 0U, 0U, 0.0}, {UINT64_C(0x610000000001), 0U, 1U, 0.0},
    {UINT64_C(0x610000000001), 0U, 2U, 0.0}, {UINT64_C(0x610000000002), 0U, 1U, 0.0},
    {UINT64_C(0x610000000002), 0U, 2U, 0.0}, {UINT64_C(0x610000000003), 0U, 2U, 0.0},
};
static const SsStrongConstraint tetra_p2_constraints[] = {
    {UINT64_C(0x610000000001), 0U, 0U, 0.0}, {UINT64_C(0x610000000001), 0U, 1U, 0.0},
    {UINT64_C(0x610000000001), 0U, 2U, 0.0}, {UINT64_C(0x610000000002), 0U, 1U, 0.0},
    {UINT64_C(0x610000000002), 0U, 2U, 0.0}, {UINT64_C(0x610000000003), 0U, 2U, 0.0},
    {UINT64_C(0x640000000001), 0U, 1U, 0.0}, {UINT64_C(0x640000000001), 0U, 2U, 0.0},
    {UINT64_C(0x640000000002), 0U, 2U, 0.0}, {UINT64_C(0x640000000004), 0U, 2U, 0.0},
};

typedef struct {
  uint32_t dimension;
  uint32_t field_order;
  SsMesh *mesh;
  SsFieldSpace *space;
  SsFieldSpace *coarse_space;
  SsAssemblyPlan *plan;
  SsOperator *operator_handle;
  SsMcParameters *material;
  SsMechanicsMaterialRegion region;
  SsMechanicsOperatorSpec mechanics_spec;
  SsOperatorSpec operator_spec;
  Vec load;
  Mat elastic;
  Mat prolongation;
  SsPmg *pmg;
  SsNonlinearProblem nonlinear;
  SsNonlinearPolicy nonlinear_policy;
} MechanicsFixture;

static int mechanics_close(PetscReal observed, PetscReal expected,
                           PetscReal absolute_tolerance, PetscReal relative_tolerance) {
  const PetscReal scale = fmax(fabs(observed), fabs(expected));
  return isfinite((double)observed) && isfinite((double)expected) &&
         fabs(observed - expected) <=
             fmax(absolute_tolerance, relative_tolerance * scale);
}

static int mechanics_has_augmented_attempt(const SsAnalysisRecord *record,
                                           int require_sensitivity) {
  size_t index;
  for (index = 0U; index < record->attempt_count; ++index) {
    const SsAnalysisAttemptRecord *attempt = &record->attempts[index];
    if (attempt->successful != 0 && attempt->committed != 0 &&
        isfinite((double)attempt->sensitivity_numerator) &&
        isfinite((double)attempt->sensitivity_denominator) &&
        fabs(attempt->sensitivity_denominator) > DBL_MIN &&
        (require_sensitivity == 0 || (isfinite((double)attempt->sensitivity_epsilon) &&
                                      attempt->sensitivity_epsilon > 0.0))) {
      return 1;
    }
  }
  return 0;
}

static SsMeshInput mechanics_mesh_input(uint32_t dimension) {
  if (dimension == 2U) {
    const SsMeshInput input = {
        .dimension = 2U,
        .global_vertex_count = 4U,
        .global_cell_count = 2U,
        .global_boundary_facet_count = 4U,
        .global_intermediate_entity_count = sizeof(square_intermediate_entities) /
                                            sizeof(square_intermediate_entities[0]),
        .global_coordinate_node_count = 4U,
        .vertex_count = 4U,
        .cell_count = 2U,
        .boundary_facet_count = 4U,
        .vertex_ids = square_vertex_ids,
        .vertex_coordinates = square_coordinates,
        .cell_ids = square_cell_ids,
        .cells = square_cells,
        .boundary_facet_ids = square_facet_ids,
        .boundary_facets = square_facets,
        .intermediate_entity_count = sizeof(square_intermediate_entities) /
                                     sizeof(square_intermediate_entities[0]),
        .intermediate_entities = square_intermediate_entities,
        .label_count = sizeof(square_labels) / sizeof(square_labels[0]),
        .labels = square_labels,
        .coordinate_order = 1U,
        .coordinate_node_count = 4U,
        .coordinate_node_ids = square_vertex_ids,
        .coordinate_coordinates = square_coordinates,
        .cell_coordinate_nodes = square_cells,
        .absolute_determinant_tolerance = 1.0e-12,
        .relative_determinant_tolerance = 1.0e-12,
    };
    return input;
  }
  {
    const SsMeshInput input = {
        .dimension = 3U,
        .global_vertex_count = 5U,
        .global_cell_count = 2U,
        .global_boundary_facet_count = 6U,
        .global_intermediate_entity_count = sizeof(tetra_intermediate_entities) /
                                            sizeof(tetra_intermediate_entities[0]),
        .global_coordinate_node_count = 5U,
        .vertex_count = 5U,
        .cell_count = 2U,
        .boundary_facet_count = 6U,
        .vertex_ids = tetra_vertex_ids,
        .vertex_coordinates = tetra_coordinates,
        .cell_ids = tetra_cell_ids,
        .cells = tetra_cells,
        .boundary_facet_ids = tetra_facet_ids,
        .boundary_facets = tetra_facets,
        .intermediate_entity_count = sizeof(tetra_intermediate_entities) /
                                     sizeof(tetra_intermediate_entities[0]),
        .intermediate_entities = tetra_intermediate_entities,
        .label_count = sizeof(tetra_labels) / sizeof(tetra_labels[0]),
        .labels = tetra_labels,
        .coordinate_order = 1U,
        .coordinate_node_count = 5U,
        .coordinate_node_ids = tetra_vertex_ids,
        .coordinate_coordinates = tetra_coordinates,
        .cell_coordinate_nodes = tetra_cells,
        .absolute_determinant_tolerance = 1.0e-12,
        .relative_determinant_tolerance = 1.0e-12,
    };
    return input;
  }
}

static void mechanics_direct_policy(SsLinearPolicyChain *chain) {
  SsLinearPolicy *policy;
  memset(chain, 0, sizeof(*chain));
  chain->count = 1U;
  policy = &chain->policies[0];
  (void)snprintf(policy->identity, sizeof(policy->identity), "%s",
                 "fixture-mechanics-direct");
  (void)snprintf(policy->options_prefix, sizeof(policy->options_prefix), "%s",
                 "fixture_mech_direct_");
  policy->operator_class = SS_LINEAR_OPERATOR_DIRECT_VALIDATION;
  policy->method = SS_LINEAR_METHOD_PREONLY;
  policy->pc = SS_LINEAR_PC_LU;
  policy->side = SS_LINEAR_SIDE_LEFT;
  policy->rtol = 1.0e-11;
  policy->atol = 1.0e-13;
  policy->dtol = 1.0e8;
  policy->maximum_iterations = 1;
  policy->restart = 2;
  policy->vector_memory_cap_bytes = 16U * 1024U * 1024U;
}

static void mechanics_pmg_policy(SsLinearPolicyChain *chain) {
  SsLinearPolicy *policy;
  memset(chain, 0, sizeof(*chain));
  chain->count = 1U;
  policy = &chain->policies[0];
  (void)snprintf(policy->identity, sizeof(policy->identity), "%s",
                 "fixture-mechanics-pmg");
  (void)snprintf(policy->options_prefix, sizeof(policy->options_prefix), "%s",
                 "fixture_mech_pmg_");
  policy->operator_class = SS_LINEAR_OPERATOR_GENERAL;
  policy->method = SS_LINEAR_METHOD_FGMRES;
  policy->pc = SS_LINEAR_PC_PMG;
  policy->side = SS_LINEAR_SIDE_RIGHT;
  policy->rtol = 1.0e-10;
  policy->atol = 1.0e-12;
  policy->dtol = 1.0e8;
  policy->maximum_iterations = 80;
  policy->restart = 20;
  policy->vector_memory_cap_bytes = 32U * 1024U * 1024U;
}

static SsStatus mechanics_evaluate_common(MechanicsFixture *fixture, Vec state,
                                          PetscReal reduction_factor,
                                          PetscReal external_load_factor, Vec residual,
                                          Mat tangent, PetscReal *functional) {
  const SsOperatorState operator_state = {
      .field = state,
      .reduction_factor = reduction_factor,
      .external_load_factor = external_load_factor,
      .cancelled = 0U,
  };
  SsErrorRecord error = {0};
  SsStatus status;
  if (tangent != NULL) {
    status = ss_operator_residual_jacobian(fixture->operator_handle, &operator_state,
                                           residual, tangent, &error);
  } else {
    status = ss_operator_residual(fixture->operator_handle, &operator_state, residual,
                                  &error);
  }
  if (status == SS_OK && functional != NULL) {
    double value = NAN;
    status = ss_operator_functional(fixture->operator_handle, &operator_state, &value,
                                    &error);
    *functional = (PetscReal)value;
  }
  return status;
}

static SsStatus mechanics_evaluate(void *opaque, Vec state, PetscReal factor,
                                   Vec residual, Mat tangent, PetscReal *functional) {
  return mechanics_evaluate_common((MechanicsFixture *)opaque, state, factor, 1.0,
                                   residual, tangent, functional);
}

static SsStatus mechanics_controlled_evaluate(void *opaque, SsNonlinearControlMode mode,
                                              Vec state, PetscReal control,
                                              Vec residual, Mat tangent,
                                              PetscReal *functional) {
  MechanicsFixture *fixture = (MechanicsFixture *)opaque;
  if (mode == SS_NONLINEAR_CONTROL_LIMIT_LOAD) {
    return mechanics_evaluate_common(fixture, state, 1.0, control, residual, tangent,
                                     functional);
  }
  if (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT) {
    return mechanics_evaluate_common(fixture, state, control, 1.0, residual, tangent,
                                     functional);
  }
  return SS_ERROR_INVALID_ARGUMENT;
}

static SsStatus mechanics_admissibility(void *opaque, Vec state, PetscReal factor,
                                        int *admissible) {
  PetscReal norm = 0.0;
  (void)opaque;
  if (admissible == NULL || VecNorm(state, NORM_INFINITY, &norm) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *admissible = isfinite((double)norm) && isfinite((double)factor);
  return SS_OK;
}

static int mechanics_create_pmg(MechanicsFixture *fixture) {
  const PetscMPIInt active_rank = 0;
  const uint32_t orders[] = {1U, 2U};
  Mat transfers[1] = {NULL};
  SsPmgLevelPolicy levels[2];
  SsPmgSpecification specification;
  size_t level_count = 1U;
  memset(levels, 0, sizeof(levels));
  (void)snprintf(levels[0].options_prefix, sizeof(levels[0].options_prefix), "%s",
                 "fixture_mech_level_");
  levels[0].method =
      fixture->field_order == 1U ? SS_PMG_LEVEL_FGMRES_JACOBI : SS_PMG_LEVEL_PREONLY_LU;
  levels[0].active_rank_count = 1U;
  levels[0].active_ranks = &active_rank;
  levels[0].maximum_iterations = fixture->field_order == 1U ? 80 : 1;
  levels[0].rtol = 1.0e-11;
  levels[0].atol = 1.0e-13;
  if (fixture->field_order == 2U) {
    SsFieldSpec coarse_spec;
    memset(&coarse_spec, 0, sizeof(coarse_spec));
    coarse_spec.order = 1U;
    coarse_spec.components = fixture->dimension;
    coarse_spec.constraint_count =
        fixture->dimension == 2U
            ? sizeof(square_constraints) / sizeof(square_constraints[0])
            : sizeof(tetra_constraints) / sizeof(tetra_constraints[0]);
    coarse_spec.constraints =
        fixture->dimension == 2U ? square_constraints : tetra_constraints;
    if (ss_mesh_field_space_create(fixture->mesh, &coarse_spec,
                                   &fixture->coarse_space) != SS_OK ||
        ss_field_space_transfer_create(fixture->coarse_space, fixture->space,
                                       &fixture->prolongation) != SS_OK) {
      return 1;
    }
    transfers[0] = fixture->prolongation;
    level_count = 2U;
    (void)snprintf(levels[1].options_prefix, sizeof(levels[1].options_prefix), "%s",
                   "fixture_mech_fine_");
    levels[1].method = SS_PMG_LEVEL_RICHARDSON_JACOBI;
    levels[1].active_rank_count = 1U;
    levels[1].active_ranks = &active_rank;
    levels[1].pre_smooth_steps = 2;
    levels[1].post_smooth_steps = 2;
    levels[1].maximum_iterations = 8;
    levels[1].richardson_scale = 0.2;
    levels[1].rtol = 1.0e-11;
    levels[1].atol = 1.0e-13;
  }
  memset(&specification, 0, sizeof(specification));
  specification.fine_order = fixture->field_order;
  specification.level_count = level_count;
  specification.orders = orders;
  specification.prolongations = fixture->field_order == 2U ? transfers : NULL;
  specification.level_policies = levels;
  specification.transfer_generation = 81U + fixture->dimension;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 64U * 1024U * 1024U;
  return ss_pmg_create(PETSC_COMM_SELF, &specification, &fixture->pmg) == SS_OK ? 0 : 1;
}

static int mechanics_fixture_create_with_coupling(
    uint32_t dimension, uint32_t field_order, SsMcDavis davis,
    SsMechanicsCouplingPointKernel coupling_point_kernel,
    const void *coupling_point_context, int requested_threads,
    MechanicsFixture *fixture) {
  const SsMcParameterInput material_input = {
      1.0e4,
      0.27,
      coupling_point_kernel == NULL ? 4.0 : (dimension == 2U ? 4.0 : 12.0),
      0.50,
      0.15,
      davis,
      UINT64_C(1),
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  };
  SsMeshInput mesh_input = mechanics_mesh_input(dimension);
  SsFieldSpec field_spec;
  SsMeshError mesh_error = {0};
  SsMcError material_error = {0};
  SsErrorRecord error = {0};
  Vec zero = NULL;
  Vec residual = NULL;
  SsOperatorState state;
  PetscReal load_norm = 0.0;
  memset(fixture, 0, sizeof(*fixture));
  fixture->dimension = dimension;
  fixture->field_order = field_order;
  FIXTURE_CHECK(field_order == 1U || field_order == 2U);
  field_spec.order = field_order;
  field_spec.components = dimension;
  if (dimension == 2U) {
    field_spec.constraint_count =
        field_order == 1U
            ? sizeof(square_constraints) / sizeof(square_constraints[0])
            : sizeof(square_p2_constraints) / sizeof(square_p2_constraints[0]);
    field_spec.constraints =
        field_order == 1U ? square_constraints : square_p2_constraints;
  } else {
    field_spec.constraint_count =
        field_order == 1U
            ? sizeof(tetra_constraints) / sizeof(tetra_constraints[0])
            : sizeof(tetra_p2_constraints) / sizeof(tetra_p2_constraints[0]);
    field_spec.constraints =
        field_order == 1U ? tetra_constraints : tetra_p2_constraints;
  }
  FIXTURE_CHECK(ss_mc_parameters_create(&material_input, &fixture->material,
                                        &material_error) == SS_OK);
  FIXTURE_CHECK(ss_mesh_create(PETSC_COMM_SELF, &mesh_input, &fixture->mesh,
                               &mesh_error) == SS_OK);
  FIXTURE_CHECK(
      ss_mesh_field_space_create(fixture->mesh, &field_spec, &fixture->space) == SS_OK);
  FIXTURE_CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, fixture->space, &fixture->plan,
                                        &error) == SS_OK);
  fixture->region.material_value = 1U;
  fixture->region.parameters = fixture->material;
  fixture->mechanics_spec.regions = &fixture->region;
  fixture->mechanics_spec.region_count = 1U;
  fixture->mechanics_spec.body_force[dimension - 1U] =
      coupling_point_kernel == NULL ? (dimension == 2U ? -15.0 : -18.0) : 0.0;
  fixture->mechanics_spec.coupling_point_kernel = coupling_point_kernel;
  fixture->mechanics_spec.coupling_point_context = coupling_point_context;
  fixture->operator_spec.physics = SS_ASSEMBLY_MECHANICS;
  fixture->operator_spec.determinism = SS_ASSEMBLY_QUALIFICATION;
  fixture->operator_spec.openmp_work_threshold = 1U;
  fixture->operator_spec.requested_threads = requested_threads;
  fixture->operator_spec.mechanics = &fixture->mechanics_spec;
  FIXTURE_CHECK(ss_operator_create(fixture->plan, &fixture->operator_spec,
                                   &fixture->operator_handle, &error) == SS_OK);
  FIXTURE_CHECK(ss_assembly_plan_create_vector(fixture->plan, &fixture->load) == SS_OK);
  FIXTURE_CHECK(ss_assembly_plan_create_matrix(fixture->plan, &fixture->elastic) ==
                SS_OK);
  FIXTURE_CHECK(ss_assembly_plan_create_vector(fixture->plan, &zero) == SS_OK);
  FIXTURE_CHECK(VecDuplicate(zero, &residual) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecSet(zero, 0.0) == PETSC_SUCCESS);
  memset(&state, 0, sizeof(state));
  state.field = zero;
  state.reduction_factor = 1.0;
  state.external_load_factor = 1.0;
  FIXTURE_CHECK(ss_operator_residual(fixture->operator_handle, &state, residual,
                                     &error) == SS_OK);
  FIXTURE_CHECK(VecCopy(residual, fixture->load) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecScale(fixture->load, -1.0) == PETSC_SUCCESS);
  state.external_load_factor = 0.0;
  FIXTURE_CHECK(ss_operator_jacobian(fixture->operator_handle, &state, fixture->elastic,
                                     &error) == SS_OK);
  FIXTURE_CHECK(VecNorm(fixture->load, NORM_2, &load_norm) == PETSC_SUCCESS);
  FIXTURE_CHECK(isfinite((double)load_norm) && load_norm > 0.0);
  FIXTURE_CHECK(VecDestroy(&residual) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDestroy(&zero) == PETSC_SUCCESS);

  fixture->nonlinear.communicator = PETSC_COMM_SELF;
  fixture->nonlinear.reference_load = fixture->load;
  fixture->nonlinear.elastic_reference = fixture->elastic;
  fixture->nonlinear.operations.evaluate = mechanics_evaluate;
  fixture->nonlinear.operations.evaluate_controlled = mechanics_controlled_evaluate;
  fixture->nonlinear.operations.admissibility = mechanics_admissibility;
  fixture->nonlinear.context = fixture;
  fixture->nonlinear.linear_properties.finite_action = 1;
  fixture->nonlinear.linear_properties.measured_symmetric = 1;
  fixture->nonlinear.linear_properties.pc_symmetric = 1;
  fixture->nonlinear.linear_properties.pc_positive = 1;
  mechanics_direct_policy(&fixture->nonlinear.linear_policies);
  fixture->nonlinear.structure_generation = 91U + dimension;
  fixture->nonlinear.first_value_generation = 101U;
  fixture->nonlinear.transaction_generation = 111U;
  fixture->nonlinear.accepted_generation = 121U;
  memset(&fixture->nonlinear_policy, 0, sizeof(fixture->nonlinear_policy));
  fixture->nonlinear_policy.tolerance = 1.0e-9;
  fixture->nonlinear_policy.stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_RESIDUAL;
  fixture->nonlinear_policy.stopping_tolerance = fixture->nonlinear_policy.tolerance;
  fixture->nonlinear_policy.maximum_iterations = 48;
  fixture->nonlinear_policy.maximum_line_search_trials = 16;
  fixture->nonlinear_policy.minimum_alpha = 1.0 / 65536.0;
  fixture->nonlinear_policy.regularization_minimum = dimension == 2U ? 1.0e-6 : 1.0e-3;
  fixture->nonlinear_policy.singular_relative_tolerance = 1.0e-11;
  fixture->nonlinear_policy.relaxed_factor = 10.0;
  fixture->nonlinear_policy.line_method = SS_NONLINEAR_LINE_DIRECTIONAL;
  return 0;
}

static int mechanics_fixture_create(uint32_t dimension, uint32_t field_order,
                                    SsMcDavis davis, MechanicsFixture *fixture) {
  return mechanics_fixture_create_with_coupling(dimension, field_order, davis, NULL,
                                                NULL, 1, fixture);
}

static int mechanics_fixture_destroy(MechanicsFixture *fixture) {
  int result = 0;
  if (ss_pmg_destroy(&fixture->pmg) != SS_OK) {
    result = 1;
  }
  if (MatDestroy(&fixture->prolongation) != PETSC_SUCCESS) {
    result = 1;
  }
  if (MatDestroy(&fixture->elastic) != PETSC_SUCCESS) {
    result = 1;
  }
  if (VecDestroy(&fixture->load) != PETSC_SUCCESS) {
    result = 1;
  }
  if (ss_operator_destroy(&fixture->operator_handle) != SS_OK) {
    result = 1;
  }
  if (ss_assembly_plan_destroy(&fixture->plan) != SS_OK) {
    result = 1;
  }
  if (ss_field_space_destroy(&fixture->space) != SS_OK) {
    result = 1;
  }
  if (ss_field_space_destroy(&fixture->coarse_space) != SS_OK) {
    result = 1;
  }
  if (ss_mesh_destroy(&fixture->mesh) != SS_OK) {
    result = 1;
  }
  if (ss_mc_parameters_destroy(&fixture->material) != SS_OK) {
    result = 1;
  }
  return result;
}

static SsStatus mechanics_analysis_fixed(void *opaque, PetscReal factor, Vec incoming,
                                         Vec solution, SsNonlinearSolveRecord *record) {
  MechanicsFixture *fixture = (MechanicsFixture *)opaque;
  return ss_nonlinear_fixed_solve(&fixture->nonlinear, &fixture->nonlinear_policy,
                                  factor, incoming, solution, record);
}

static SsStatus mechanics_analysis_work(void *opaque, SsAnalysisMethod method,
                                        PetscReal target_work,
                                        PetscReal incoming_factor, Vec incoming,
                                        Vec solution, PetscReal *accepted_factor,
                                        SsNonlinearSolveRecord *record) {
  MechanicsFixture *fixture = (MechanicsFixture *)opaque;
  SsNonlinearPolicy policy = fixture->nonlinear_policy;
  SsNonlinearControlMode mode;
  if (method == SS_ANALYSIS_LIMIT_LOAD) {
    mode = SS_NONLINEAR_CONTROL_LIMIT_LOAD;
  } else if (method == SS_ANALYSIS_SSR_INDIRECT) {
    mode = SS_NONLINEAR_CONTROL_SSR_INDIRECT;
    policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
  } else {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_nonlinear_work_controlled_solve(&fixture->nonlinear, &policy, mode,
                                            target_work, incoming_factor, incoming,
                                            solution, accepted_factor, record);
}

static SsStatus mechanics_analysis_potential(void *opaque, PetscReal factor, Vec state,
                                             PetscReal *total_potential) {
  MechanicsFixture *fixture = (MechanicsFixture *)opaque;
  Vec residual = NULL;
  SsStatus status;
  if (VecDuplicate(state, &residual) != PETSC_SUCCESS) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  status = mechanics_evaluate_common(fixture, state, factor, 1.0, residual, NULL,
                                     total_potential);
  if (VecDestroy(&residual) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}

static void mechanics_analysis_problem(MechanicsFixture *fixture,
                                       SsAnalysisProblem *problem) {
  memset(problem, 0, sizeof(*problem));
  problem->communicator = PETSC_COMM_SELF;
  problem->reference_load = fixture->load;
  problem->operations.solve_fixed = mechanics_analysis_fixed;
  problem->operations.solve_work_controlled = mechanics_analysis_work;
  problem->operations.total_potential = mechanics_analysis_potential;
  problem->context = fixture;
}

static void mechanics_analysis_policy(SsAnalysisMethod method, PetscReal fixed_work,
                                      SsAnalysisPolicy *policy) {
  memset(policy, 0, sizeof(*policy));
  policy->method = method;
  policy->nonlinear_tolerance = 1.0e-9;
  policy->factor_initial = 1.0;
  policy->factor_step_initial = 0.1;
  policy->factor_step_minimum = 0.0125;
  policy->factor_maximum = 1.4;
  policy->work_step_initial = 0.2 * fixed_work;
  policy->work_step_minimum = 0.0125 * fixed_work;
  policy->work_maximum = 1.4 * fixed_work;
  policy->accepted_step_limit = 4;
  policy->consecutive_retry_limit = 5;
}

static int mechanics_run_dimension(uint32_t dimension) {
  static const PetscReal expected_fixed_work[] = {0.0, 0.0, 0.0057368553082191775,
                                                  0.0029595068486260043};
  static const PetscReal expected_direct_factor[] = {0.0, 0.0, 1.2500000000000002,
                                                     1.2625000000000002};
  static const PetscReal expected_direct_control[] = {0.0, 0.0, 0.0064779977129031377,
                                                      0.001336781893946509};
  static const PetscReal expected_direct_work[] = {0.0, 0.0, 0.012063106429870629,
                                                   0.0043302935858201682};
  static const PetscReal expected_indirect_factor[] = {0.0, 0.0, 1.201421804357425,
                                                       1.2658474242735884};
  static const PetscReal expected_indirect_work[] = {0.0, 0.0, 0.0080315974315068473,
                                                     0.0059190136972520086};
  MechanicsFixture fixture;
  SsAnalysisProblem analysis_problem;
  SsAnalysisPolicy analysis_policy;
  SsAnalysisRecord analysis_record;
  SsNonlinearSolveRecord solve_record;
  Vec zero = NULL;
  Vec direct_solution = NULL;
  Vec pmg_solution = NULL;
  Vec controlled_solution = NULL;
  PetscScalar work_scalar = 0.0;
  PetscReal profile_difference = NAN;
  PetscReal fixed_work;
  PetscReal accepted_control = NAN;
  FIXTURE_CHECK(mechanics_fixture_create(dimension, 1U, SS_MC_DAVIS_B, &fixture) == 0);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &zero) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &direct_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &pmg_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &controlled_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecSet(zero, 0.0) == PETSC_SUCCESS);
  FIXTURE_CHECK(ss_nonlinear_fixed_solve(&fixture.nonlinear, &fixture.nonlinear_policy,
                                         1.0, zero, direct_solution,
                                         &solve_record) == SS_OK);
  FIXTURE_CHECK(solve_record.outcome == SS_NONLINEAR_CONVERGED);
  FIXTURE_CHECK(solve_record.history_count > 0U);
  FIXTURE_CHECK(strcmp(solve_record.history[0].linear_setup.selected_identity,
                       "fixture-mechanics-direct") == 0);
  FIXTURE_CHECK(VecDot(fixture.load, direct_solution, &work_scalar) == PETSC_SUCCESS);
  fixed_work = PetscRealPart(work_scalar);
  FIXTURE_CHECK(isfinite((double)fixed_work) && fixed_work > 0.0);
  FIXTURE_CHECK(
      mechanics_close(fixed_work, expected_fixed_work[dimension], 1.0e-12, 2.0e-8));

  FIXTURE_CHECK(mechanics_create_pmg(&fixture) == 0);
  fixture.nonlinear.pmg = fixture.pmg;
  fixture.nonlinear.linear_properties.pc_variable = 1;
  mechanics_pmg_policy(&fixture.nonlinear.linear_policies);
  FIXTURE_CHECK(ss_nonlinear_fixed_solve(&fixture.nonlinear, &fixture.nonlinear_policy,
                                         1.0, zero, pmg_solution,
                                         &solve_record) == SS_OK);
  FIXTURE_CHECK(solve_record.outcome == SS_NONLINEAR_CONVERGED);
  FIXTURE_CHECK(solve_record.history_count > 0U &&
                solve_record.history[0].linear_setup.pmg_selected != 0);
  FIXTURE_CHECK(VecAXPY(pmg_solution, -1.0, direct_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecNorm(pmg_solution, NORM_2, &profile_difference) == PETSC_SUCCESS);
  FIXTURE_CHECK(isfinite((double)profile_difference) && profile_difference <= 1.0e-8);

  fixture.nonlinear.pmg = NULL;
  fixture.nonlinear.linear_properties.pc_variable = 0;
  mechanics_direct_policy(&fixture.nonlinear.linear_policies);
  FIXTURE_CHECK(ss_nonlinear_work_controlled_solve(
                    &fixture.nonlinear, &fixture.nonlinear_policy,
                    SS_NONLINEAR_CONTROL_LIMIT_LOAD, 0.6 * fixed_work, 0.6,
                    direct_solution, controlled_solution, &accepted_control,
                    &solve_record) == SS_OK);
  FIXTURE_CHECK(solve_record.outcome == SS_NONLINEAR_CONVERGED);
  FIXTURE_CHECK(isfinite((double)accepted_control));
  FIXTURE_CHECK(VecDot(fixture.load, controlled_solution, &work_scalar) ==
                PETSC_SUCCESS);
  FIXTURE_CHECK(
      mechanics_close(PetscRealPart(work_scalar), 0.6 * fixed_work, 1.0e-12, 1.0e-10));

  mechanics_analysis_problem(&fixture, &analysis_problem);
  mechanics_analysis_policy(SS_ANALYSIS_SSR_DIRECT, fixed_work, &analysis_policy);
  if (dimension == 3U) {
    analysis_policy.factor_maximum = 2.0;
    analysis_policy.direct_control_maximum = 1.0e-6;
    analysis_policy.accepted_step_limit = 8;
  }
  FIXTURE_CHECK(VecSet(zero, 0.0) == PETSC_SUCCESS);
  FIXTURE_CHECK(ss_analysis_run(&analysis_problem, &analysis_policy, zero,
                                controlled_solution, &analysis_record) == SS_OK);
  FIXTURE_CHECK(analysis_record.point_count == (dimension == 2U ? 4U : 5U));
  FIXTURE_CHECK(analysis_record.commit_count == analysis_record.point_count);
  FIXTURE_CHECK(analysis_record.terminal_success != 0);
  FIXTURE_CHECK(analysis_record.stop_reason ==
                (dimension == 2U ? SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT
                                 : SS_ANALYSIS_STOP_DIRECT_CONTROL_HORIZON_REACHED));
  FIXTURE_CHECK(mechanics_close(analysis_record.final_factor,
                                expected_direct_factor[dimension], 1.0e-10, 2.0e-8));
  FIXTURE_CHECK(mechanics_close(analysis_record.final_control,
                                expected_direct_control[dimension], 1.0e-10, 2.0e-7));
  FIXTURE_CHECK(mechanics_close(analysis_record.final_work,
                                expected_direct_work[dimension], 1.0e-10, 2.0e-8));
  FIXTURE_CHECK(fabs(analysis_record.final_control - analysis_record.final_work) >
                1.0e-4);
  FIXTURE_CHECK(mechanics_has_augmented_attempt(&analysis_record, 1) != 0);

  mechanics_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, fixed_work, &analysis_policy);
  if (dimension == 3U) {
    analysis_policy.factor_maximum = 2.0;
    analysis_policy.work_maximum = 2.0 * fixed_work;
    analysis_policy.accepted_step_limit = 8;
  }
  FIXTURE_CHECK(VecSet(zero, 0.0) == PETSC_SUCCESS);
  FIXTURE_CHECK(ss_analysis_run(&analysis_problem, &analysis_policy, zero,
                                controlled_solution, &analysis_record) == SS_OK);
  FIXTURE_CHECK(analysis_record.point_count == (dimension == 2U ? 4U : 7U));
  FIXTURE_CHECK(analysis_record.commit_count == analysis_record.point_count);
  FIXTURE_CHECK(analysis_record.terminal_success != 0);
  FIXTURE_CHECK(analysis_record.stop_reason == SS_ANALYSIS_STOP_WORK_HORIZON_REACHED);
  FIXTURE_CHECK(mechanics_close(analysis_record.final_factor,
                                expected_indirect_factor[dimension], 1.0e-10, 2.0e-7));
  FIXTURE_CHECK(mechanics_close(analysis_record.final_work,
                                expected_indirect_work[dimension], 1.0e-10, 2.0e-8));
  FIXTURE_CHECK(mechanics_close(analysis_record.final_control,
                                analysis_record.final_work, 1.0e-12, 1.0e-12));
  FIXTURE_CHECK(analysis_record.final_work > 1.35 * fixed_work);
  FIXTURE_CHECK(mechanics_has_augmented_attempt(&analysis_record, 1) != 0);

  mechanics_analysis_policy(SS_ANALYSIS_LIMIT_LOAD, fixed_work, &analysis_policy);
  analysis_policy.work_maximum = 0.8 * fixed_work;
  FIXTURE_CHECK(VecSet(zero, 0.0) == PETSC_SUCCESS);
  FIXTURE_CHECK(ss_analysis_run(&analysis_problem, &analysis_policy, zero,
                                controlled_solution, &analysis_record) == SS_OK);
  FIXTURE_CHECK(analysis_record.point_count == 5U);
  FIXTURE_CHECK(analysis_record.rollback_count == 0U);
  FIXTURE_CHECK(analysis_record.commit_count == analysis_record.accepted_step_count);
  FIXTURE_CHECK(analysis_record.terminal_success != 0);
  FIXTURE_CHECK(analysis_record.stop_reason == SS_ANALYSIS_STOP_WORK_HORIZON_REACHED);
  FIXTURE_CHECK(mechanics_close(analysis_record.final_control, 0.8 * fixed_work,
                                1.0e-12, 2.0e-8));
  FIXTURE_CHECK(
      mechanics_close(analysis_record.final_work, 0.8 * fixed_work, 1.0e-12, 2.0e-8));
  FIXTURE_CHECK(mechanics_close(analysis_record.final_factor, 0.8, 1.0e-9, 2.0e-8));
  FIXTURE_CHECK(mechanics_has_augmented_attempt(&analysis_record, 0) != 0);
  PetscReal terminal_residual = NAN;
  PetscReal load_norm = NAN;
  FIXTURE_CHECK(mechanics_controlled_evaluate(
                    &fixture, SS_NONLINEAR_CONTROL_LIMIT_LOAD, controlled_solution,
                    analysis_record.final_factor, pmg_solution, NULL, NULL) == SS_OK);
  FIXTURE_CHECK(VecNorm(pmg_solution, NORM_2, &terminal_residual) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecNorm(fixture.load, NORM_2, &load_norm) == PETSC_SUCCESS);
  FIXTURE_CHECK(terminal_residual <=
                fixture.nonlinear_policy.stopping_tolerance * load_norm);

  FIXTURE_CHECK(VecDestroy(&controlled_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDestroy(&pmg_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDestroy(&direct_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDestroy(&zero) == PETSC_SUCCESS);
  FIXTURE_CHECK(mechanics_fixture_destroy(&fixture) == 0);
  return 0;
}

static int mechanics_run_high_order_profile(uint32_t dimension) {
  MechanicsFixture fixture;
  SsNonlinearSolveRecord direct_record;
  SsNonlinearSolveRecord pmg_record;
  Vec zero = NULL;
  Vec direct_solution = NULL;
  Vec pmg_solution = NULL;
  PetscReal difference = NAN;
  PetscReal elastic_reference_norm = NAN;
  size_t iteration;
  FIXTURE_CHECK(mechanics_fixture_create(dimension, 2U, SS_MC_DAVIS_B, &fixture) == 0);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &zero) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &direct_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &pmg_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecSet(zero, 0.0) == PETSC_SUCCESS);
  FIXTURE_CHECK(ss_nonlinear_fixed_solve(&fixture.nonlinear, &fixture.nonlinear_policy,
                                         1.0, zero, direct_solution,
                                         &direct_record) == SS_OK);
  FIXTURE_CHECK(direct_record.outcome == SS_NONLINEAR_CONVERGED);
  FIXTURE_CHECK(direct_record.history_count > 0U);
  FIXTURE_CHECK(strcmp(direct_record.history[0].linear_setup.selected_identity,
                       "fixture-mechanics-direct") == 0);

  FIXTURE_CHECK(mechanics_create_pmg(&fixture) == 0);
  /*
   * The elastic reference supplies only the fixed sparsity pattern once a
   * nonlinear tangent has been assembled.  Poison its values so this profile
   * fails if PMG is ever refreshed from the stale reference instead of the
   * current regularized tangent.
   */
  FIXTURE_CHECK(MatZeroEntries(fixture.elastic) == PETSC_SUCCESS);
  FIXTURE_CHECK(MatNorm(fixture.elastic, NORM_FROBENIUS, &elastic_reference_norm) ==
                PETSC_SUCCESS);
  FIXTURE_CHECK(elastic_reference_norm == 0.0);
  fixture.nonlinear.pmg = fixture.pmg;
  fixture.nonlinear.linear_properties.pc_variable = 1;
  mechanics_pmg_policy(&fixture.nonlinear.linear_policies);
  FIXTURE_CHECK(ss_nonlinear_fixed_solve(&fixture.nonlinear, &fixture.nonlinear_policy,
                                         1.0, zero, pmg_solution,
                                         &pmg_record) == SS_OK);
  FIXTURE_CHECK(pmg_record.outcome == SS_NONLINEAR_CONVERGED);
  FIXTURE_CHECK(pmg_record.history_count > 0U);
  FIXTURE_CHECK(pmg_record.history[0].linear_setup.pmg_selected != 0);
  FIXTURE_CHECK(pmg_record.history[0].linear_setup.pmg_setup.hierarchy.level_count ==
                2U);
  FIXTURE_CHECK(pmg_record.history[0].linear_setup.pmg_setup.hierarchy.orders[0] == 1U);
  FIXTURE_CHECK(pmg_record.history[0].linear_setup.pmg_setup.hierarchy.orders[1] == 2U);
  for (iteration = 0U; iteration < pmg_record.history_count; ++iteration) {
    const SsNonlinearIterationRecord *entry = &pmg_record.history[iteration];
    FIXTURE_CHECK(entry->linear_setup.pmg_selected != 0);
    FIXTURE_CHECK(entry->linear_setup.pmg_setup.requested_value_generation ==
                  entry->operator_value_generation);
    FIXTURE_CHECK(entry->linear_setup.pmg_setup.preconditioner_value_generation ==
                  entry->operator_value_generation);
    FIXTURE_CHECK(entry->linear_setup.pmg_setup.numeric_refreshed != 0);
  }
  FIXTURE_CHECK(VecAXPY(pmg_solution, -1.0, direct_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecNorm(pmg_solution, NORM_2, &difference) == PETSC_SUCCESS);
  FIXTURE_CHECK(isfinite((double)difference) && difference <= 1.0e-7);

  FIXTURE_CHECK(VecDestroy(&pmg_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDestroy(&direct_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDestroy(&zero) == PETSC_SUCCESS);
  FIXTURE_CHECK(mechanics_fixture_destroy(&fixture) == 0);
  return 0;
}

static int mechanics_run_davis_modes(void) {
  const SsMcDavis modes[] = {SS_MC_DAVIS_A, SS_MC_DAVIS_B, SS_MC_DAVIS_C};
  size_t index;
  for (index = 0U; index < sizeof(modes) / sizeof(modes[0]); ++index) {
    MechanicsFixture fixture;
    SsNonlinearSolveRecord record;
    Vec zero = NULL;
    Vec accepted = NULL;
    FIXTURE_CHECK(mechanics_fixture_create(2U, 1U, modes[index], &fixture) == 0);
    FIXTURE_CHECK(VecDuplicate(fixture.load, &zero) == PETSC_SUCCESS);
    FIXTURE_CHECK(VecDuplicate(fixture.load, &accepted) == PETSC_SUCCESS);
    FIXTURE_CHECK(VecSet(zero, 0.0) == PETSC_SUCCESS);
    FIXTURE_CHECK(ss_nonlinear_fixed_solve(&fixture.nonlinear,
                                           &fixture.nonlinear_policy, 1.0, zero,
                                           accepted, &record) == SS_OK);
    if (record.outcome != SS_NONLINEAR_CONVERGED) {
      fprintf(stderr,
              "Davis fixture mode=%d outcome=%d iterations=%d initial=%.17g "
              "final=%.17g r=%.17g\n",
              (int)modes[index], (int)record.outcome, (int)record.iterations,
              (double)record.initial_convergence, (double)record.final_convergence,
              (double)record.final_regularization);
    }
    FIXTURE_CHECK(record.outcome == SS_NONLINEAR_CONVERGED);
    FIXTURE_CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
    FIXTURE_CHECK(VecDestroy(&zero) == PETSC_SUCCESS);
    FIXTURE_CHECK(mechanics_fixture_destroy(&fixture) == 0);
  }
  return 0;
}

int mechanics_fixture_coupled_ssr_profile(
    uint32_t dimension, SsMechanicsCouplingPointKernel coupling_point_kernel,
    const void *coupling_point_context, int requested_threads,
    double *trajectory_checksum) {
  static const double uncoupled_checksum[4] = {
      0.0,
      0.0,
      4.0 + 1.201421804357425 + 2.0 * 0.0080315974315068473,
      7.0 + 1.2658474242735884 + 2.0 * 0.0059190136972520086,
  };
  MechanicsFixture fixture;
  SsNonlinearSolveRecord direct_record;
  SsNonlinearSolveRecord pmg_record;
  SsAnalysisProblem analysis_problem;
  SsAnalysisPolicy analysis_policy;
  SsAnalysisRecord analysis_record;
  Vec zero = NULL;
  Vec direct_solution = NULL;
  Vec pmg_solution = NULL;
  Vec analysis_solution = NULL;
  PetscScalar work_scalar = 0.0;
  PetscReal difference = NAN;
  PetscReal fixed_work = NAN;
  if ((dimension != 2U && dimension != 3U) || coupling_point_kernel == NULL ||
      coupling_point_context == NULL || requested_threads <= 0 ||
      trajectory_checksum == NULL) {
    return 1;
  }
  FIXTURE_CHECK(mechanics_fixture_create_with_coupling(
                    dimension, 1U, SS_MC_DAVIS_B, coupling_point_kernel,
                    coupling_point_context, requested_threads, &fixture) == 0);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &zero) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &direct_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &pmg_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDuplicate(fixture.load, &analysis_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecSet(zero, 0.0) == PETSC_SUCCESS);
  FIXTURE_CHECK(ss_nonlinear_fixed_solve(&fixture.nonlinear, &fixture.nonlinear_policy,
                                         1.0, zero, direct_solution,
                                         &direct_record) == SS_OK);
  if (direct_record.outcome != SS_NONLINEAR_CONVERGED) {
    fprintf(stderr,
            "coupled fixed dimension=%u outcome=%d iterations=%d "
            "initial=%.17g final=%.17g\n",
            dimension, (int)direct_record.outcome, (int)direct_record.iterations,
            (double)direct_record.initial_convergence,
            (double)direct_record.final_convergence);
  }
  FIXTURE_CHECK(direct_record.outcome == SS_NONLINEAR_CONVERGED);
  FIXTURE_CHECK(direct_record.history_count > 0U);
  FIXTURE_CHECK(VecDot(fixture.load, direct_solution, &work_scalar) == PETSC_SUCCESS);
  fixed_work = PetscRealPart(work_scalar);
  FIXTURE_CHECK(isfinite((double)fixed_work) && fixed_work > 0.0);

  FIXTURE_CHECK(mechanics_create_pmg(&fixture) == 0);
  fixture.nonlinear.pmg = fixture.pmg;
  fixture.nonlinear.linear_properties.pc_variable = 1;
  mechanics_pmg_policy(&fixture.nonlinear.linear_policies);
  FIXTURE_CHECK(ss_nonlinear_fixed_solve(&fixture.nonlinear, &fixture.nonlinear_policy,
                                         1.0, zero, pmg_solution,
                                         &pmg_record) == SS_OK);
  FIXTURE_CHECK(pmg_record.outcome == SS_NONLINEAR_CONVERGED);
  FIXTURE_CHECK(pmg_record.history_count > 0U &&
                pmg_record.history[0].linear_setup.pmg_selected != 0);
  FIXTURE_CHECK(VecAXPY(pmg_solution, -1.0, direct_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecNorm(pmg_solution, NORM_2, &difference) == PETSC_SUCCESS);
  FIXTURE_CHECK(isfinite((double)difference) && difference <= 1.0e-8);

  fixture.nonlinear.pmg = NULL;
  fixture.nonlinear.linear_properties.pc_variable = 0;
  mechanics_direct_policy(&fixture.nonlinear.linear_policies);
  mechanics_analysis_problem(&fixture, &analysis_problem);
  mechanics_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, fixed_work, &analysis_policy);
  analysis_policy.factor_maximum = 2.0;
  analysis_policy.work_maximum = 2.0 * fixed_work;
  analysis_policy.accepted_step_limit = 8;
  FIXTURE_CHECK(ss_analysis_run(&analysis_problem, &analysis_policy, zero,
                                analysis_solution, &analysis_record) == SS_OK);
  if (analysis_record.terminal_success == 0) {
    fprintf(stderr,
            "coupled SSR dimension=%u stop=%d points=%zu attempts=%zu "
            "factor=%.17g control=%.17g work=%.17g\n",
            dimension, (int)analysis_record.stop_reason, analysis_record.point_count,
            analysis_record.attempt_count, (double)analysis_record.final_factor,
            (double)analysis_record.final_control, (double)analysis_record.final_work);
  }
  FIXTURE_CHECK(analysis_record.terminal_success != 0);
  FIXTURE_CHECK(analysis_record.point_count >= 2U);
  FIXTURE_CHECK(analysis_record.commit_count == analysis_record.point_count);
  *trajectory_checksum =
      (double)analysis_record.point_count + (double)analysis_record.final_factor +
      (double)analysis_record.final_control + (double)analysis_record.final_work;
  FIXTURE_CHECK(isfinite(*trajectory_checksum));
  FIXTURE_CHECK(fabs(*trajectory_checksum - uncoupled_checksum[dimension]) > 1.0e-6);

  FIXTURE_CHECK(VecDestroy(&analysis_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDestroy(&pmg_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDestroy(&direct_solution) == PETSC_SUCCESS);
  FIXTURE_CHECK(VecDestroy(&zero) == PETSC_SUCCESS);
  FIXTURE_CHECK(mechanics_fixture_destroy(&fixture) == 0);
  return 0;
}

int mechanics_fixture_run_profiles(void) {
  FIXTURE_CHECK(mechanics_run_davis_modes() == 0);
  FIXTURE_CHECK(mechanics_run_dimension(2U) == 0);
  FIXTURE_CHECK(mechanics_run_dimension(3U) == 0);
  FIXTURE_CHECK(mechanics_run_high_order_profile(2U) == 0);
  FIXTURE_CHECK(mechanics_run_high_order_profile(3U) == 0);
  return 0;
}

int mechanics_fixture_run_region_body_force(void) {
  double self_weight_2d[3] = {NAN, NAN, NAN};
  double self_weight_3d[3] = {NAN, NAN, NAN};
  SsMeshLabelAssignment labels[sizeof(square_labels) / sizeof(square_labels[0])];
  const SsMcParameterInput material_input = {
      1.0e4,
      0.27,
      4.0,
      0.50,
      0.15,
      SS_MC_DAVIS_B,
      UINT64_C(1),
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  };
  SsMechanicsMaterialRegion regions[2] = {
      {
          .material_value = 1U,
          .has_body_force = 1,
          .body_force = {0.0, -10.0, 0.0},
      },
      {
          .material_value = 2U,
          .has_body_force = 1,
          .body_force = {0.0, -30.0, 0.0},
      },
  };
  SsMechanicsOperatorSpec mechanics_spec = {
      .regions = regions,
      .region_count = 2U,
      .body_force = {0.0, -999.0, 0.0},
  };
  const SsOperatorSpec operator_spec = {
      .physics = SS_ASSEMBLY_MECHANICS,
      .determinism = SS_ASSEMBLY_QUALIFICATION,
      .openmp_work_threshold = 1U,
      .requested_threads = 1,
      .mechanics = &mechanics_spec,
  };
  SsMeshInput mesh_input = mechanics_mesh_input(2U);
  const SsFieldSpec field_spec = {
      .order = 1U,
      .components = 2U,
  };
  SsMesh *mesh = NULL;
  SsFieldSpace *space = NULL;
  SsAssemblyPlan *plan = NULL;
  SsOperator *operator_handle = NULL;
  SsMcParameters *material = NULL;
  Vec zero = NULL;
  Vec residual = NULL;
  PetscScalar sum = NAN;
  SsMeshError mesh_error = {0};
  SsMcError material_error = {0};
  SsErrorRecord error = {0};
  SsOperatorState state = {
      .reduction_factor = 1.0,
      .external_load_factor = 1.0,
  };
  SsEntityId material_ids[2] = {square_cell_ids[0], square_cell_ids[1]};
  SsSolverMaterial source_materials[2] = {{.kind = SS_SOLVER_MATERIAL_MECHANICS,
                                           .entity_ids = &material_ids[0],
                                           .entity_count = 1U,
                                           .unsaturated_unit_weight = 10.0,
                                           .saturated_unit_weight = 20.0},
                                          {.kind = SS_SOLVER_MATERIAL_MECHANICS,
                                           .entity_ids = &material_ids[1],
                                           .entity_count = 1U,
                                           .unsaturated_unit_weight = 30.0,
                                           .saturated_unit_weight = 40.0}};
  SsSolverProblemModel source_model = {
      .dimension = 2U,
      .materials = source_materials,
      .material_count = 2U,
      .prescribed_saturation = {
          .axis = 0U, .count = 2U, .coordinates = {0.0, 1.0}, .levels = {1.0, 0.0}}};
  int result = 1;

  FIXTURE_CHECK(ss_solver_self_weight(2U, 10.0, self_weight_2d) == SS_OK);
  FIXTURE_CHECK(ss_solver_self_weight(3U, 20.0, self_weight_3d) == SS_OK);
  FIXTURE_CHECK(self_weight_2d[0] == 0.0 && self_weight_2d[1] == -10.0 &&
                self_weight_2d[2] == 0.0);
  FIXTURE_CHECK(self_weight_3d[0] == 0.0 && self_weight_3d[1] == -20.0 &&
                self_weight_3d[2] == 0.0);

  {
    SsSolverPrescribedSaturation profile = {
        .axis = 0U, .count = 2U, .coordinates = {0.0, 2.0}, .levels = {1.0, 3.0}};
    double physical[3] = {1.0, 2.0, 0.0};
    double weight = NAN;
    FIXTURE_CHECK(ss_solver_saturation_validate(&profile, 2U) == SS_OK);
    FIXTURE_CHECK(ss_solver_saturation_weight(&profile, 2U, physical, 10.0, 20.0,
                                              &weight) == SS_OK);
    FIXTURE_CHECK(weight == 20.0);
    physical[1] = 2.01;
    FIXTURE_CHECK(ss_solver_saturation_weight(&profile, 2U, physical, 10.0, 20.0,
                                              &weight) == SS_OK);
    FIXTURE_CHECK(weight == 10.0);
    physical[0] = -1.0;
    physical[1] = 1.0;
    FIXTURE_CHECK(ss_solver_saturation_weight(&profile, 2U, physical, 10.0, 20.0,
                                              &weight) == SS_OK);
    FIXTURE_CHECK(weight == 20.0);
    physical[0] = 9.0;
    physical[1] = 3.01;
    FIXTURE_CHECK(ss_solver_saturation_weight(&profile, 2U, physical, 10.0, 20.0,
                                              &weight) == SS_OK);
    FIXTURE_CHECK(weight == 10.0);
    profile.axis = 2U;
    physical[2] = 1.0;
    physical[1] = 2.0;
    FIXTURE_CHECK(ss_solver_saturation_validate(&profile, 2U) ==
                  SS_ERROR_INVALID_ARGUMENT);
    FIXTURE_CHECK(ss_solver_saturation_weight(&profile, 3U, physical, 10.0, 20.0,
                                              &weight) == SS_OK);
    FIXTURE_CHECK(weight == 20.0);
    profile.count = 1U;
    FIXTURE_CHECK(ss_solver_saturation_weight(&profile, 3U, physical, 10.0, 20.0,
                                              &weight) == SS_OK);
    FIXTURE_CHECK(weight == 10.0);
    profile.count = 2U;
    profile.coordinates[1] = 0.0;
    FIXTURE_CHECK(ss_solver_saturation_validate(&profile, 3U) ==
                  SS_ERROR_INVALID_ARGUMENT);
    profile.coordinates[1] = 2.0;
    profile.levels[1] = NAN;
    FIXTURE_CHECK(ss_solver_saturation_validate(&profile, 3U) ==
                  SS_ERROR_INVALID_ARGUMENT);
  }

  memcpy(labels, square_labels, sizeof(labels));
  labels[1].value = 2U;
  mesh_input.labels = labels;
  regions[0].parameters = material;
  regions[1].parameters = material;
  if (ss_mc_parameters_create(&material_input, &material, &material_error) != SS_OK) {
    goto cleanup;
  }
  regions[0].parameters = material;
  regions[1].parameters = material;
  if (ss_mesh_create(PETSC_COMM_SELF, &mesh_input, &mesh, &mesh_error) != SS_OK ||
      ss_mesh_field_space_create(mesh, &field_spec, &space) != SS_OK ||
      ss_assembly_plan_create(PETSC_COMM_SELF, space, &plan, &error) != SS_OK ||
      ss_operator_create(plan, &operator_spec, &operator_handle, &error) != SS_OK ||
      ss_assembly_plan_create_vector(plan, &zero) != SS_OK ||
      VecDuplicate(zero, &residual) != PETSC_SUCCESS ||
      VecSet(zero, 0.0) != PETSC_SUCCESS) {
    goto cleanup;
  }
  state.field = zero;
  if (ss_operator_residual(operator_handle, &state, residual, &error) != SS_OK ||
      VecSum(residual, &sum) != PETSC_SUCCESS) {
    goto cleanup;
  }
  /*
   * Each triangle has area 1/2. Internal stress is zero, so the unconstrained
   * residual sum is -integral(body_force_y) = 20.
   */
  if (!mechanics_close(PetscRealPart(sum), 20.0, 1.0e-12, 1.0e-12)) {
    goto cleanup;
  }
  /* y <= 1-x saturates the lower triangle; the upper remains dry.
   * The exact resultant is 0.5*20 + 0.5*30 = 25. This exercises the actual
   * application material lookup and callback through native assembly. */
  if (ss_operator_destroy(&operator_handle) != SS_OK)
    goto cleanup;
  mechanics_spec.coupling_point_kernel = ss_solver_prescribed_mechanics_force;
  mechanics_spec.coupling_point_context = &source_model;
  mechanics_spec.point_force_mode = SS_MECHANICS_POINT_FORCE_LOAD;
  if (ss_operator_create(plan, &operator_spec, &operator_handle, &error) != SS_OK ||
      ss_operator_residual(operator_handle, &state, residual, &error) != SS_OK ||
      VecSum(residual, &sum) != PETSC_SUCCESS ||
      !mechanics_close(PetscRealPart(sum), 25.0, 1.0e-12, 1.0e-12))
    goto cleanup;
  state.external_load_factor = 2.0;
  if (ss_operator_residual(operator_handle, &state, residual, &error) != SS_OK ||
      VecSum(residual, &sum) != PETSC_SUCCESS ||
      !mechanics_close(PetscRealPart(sum), 50.0, 1.0e-12, 1.0e-12))
    goto cleanup;
  state.external_load_factor = 0.25;
  if (ss_operator_residual(operator_handle, &state, residual, &error) != SS_OK ||
      VecSum(residual, &sum) != PETSC_SUCCESS ||
      !mechanics_close(PetscRealPart(sum), 6.25, 1.0e-12, 1.0e-12))
    goto cleanup;
  /* Fixed hydraulic coupling remains unscaled: 2*20 + 5 = 45. */
  if (ss_operator_destroy(&operator_handle) != SS_OK)
    goto cleanup;
  mechanics_spec.point_force_mode = SS_MECHANICS_POINT_FORCE_FIXED;
  state.external_load_factor = 2.0;
  if (ss_operator_create(plan, &operator_spec, &operator_handle, &error) != SS_OK ||
      ss_operator_residual(operator_handle, &state, residual, &error) != SS_OK ||
      VecSum(residual, &sum) != PETSC_SUCCESS ||
      !mechanics_close(PetscRealPart(sum), 45.0, 1.0e-12, 1.0e-12))
    goto cleanup;
  {
    double point[2] = {0.1, 0.1}, force[2];
    if (ss_solver_prescribed_mechanics_force(&source_model, 2U, UINT64_C(0), 0U, point,
                                             force) != SS_ERROR_SCHEMA)
      goto cleanup;
  }
  result = 0;

cleanup:
  if (VecDestroy(&residual) != PETSC_SUCCESS || VecDestroy(&zero) != PETSC_SUCCESS ||
      ss_operator_destroy(&operator_handle) != SS_OK ||
      ss_assembly_plan_destroy(&plan) != SS_OK ||
      ss_field_space_destroy(&space) != SS_OK || ss_mesh_destroy(&mesh) != SS_OK ||
      ss_mc_parameters_destroy(&material) != SS_OK) {
    result = 1;
  }
  return result;
}
