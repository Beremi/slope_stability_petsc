/* Energy-gradient oracle for the mechanics operator with natural loads.
 *
 * The direct-SSR control differentiates the operator functional, so its
 * directional derivative must equal the residual action R(u).v for every
 * conservative load, including facet tractions and normal pressures.  On an
 * elastic state the functional is quadratic, so the central difference is
 * exact up to rounding.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/assembly/assembly.h"
#include "slope_internal/material/material.h"
#include "slope_internal/mesh/mesh.h"

#include <petscsys.h>
#include <petscvec.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__,             \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static const SsEntityId square_vertex_ids[] = {
    UINT64_C(0x510000000001), UINT64_C(0x510000000002), UINT64_C(0x510000000003),
    UINT64_C(0x510000000004)};
static const double square_coordinates[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
static const uint64_t square_cells[] = {0U, 1U, 2U, 1U, 3U, 2U};
static const SsEntityId square_cell_ids[] = {UINT64_C(0x520000000001),
                                             UINT64_C(0x520000000002)};
static const uint64_t square_facets[] = {0U, 1U, 1U, 3U, 3U, 2U, 2U, 0U};
static const SsEntityId square_facet_ids[] = {
    UINT64_C(0x530000000001), UINT64_C(0x530000000002), UINT64_C(0x530000000003),
    UINT64_C(0x530000000004)};
static const SsMeshIntermediateEntity square_edges[] = {
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

static const SsEntityId tetra_vertex_ids[] = {
    UINT64_C(0x610000000001), UINT64_C(0x610000000002), UINT64_C(0x610000000003),
    UINT64_C(0x610000000004), UINT64_C(0x610000000005)};
static const double tetra_coordinates[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                                           0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
static const uint64_t tetra_cells[] = {0U, 1U, 2U, 3U, 1U, 2U, 3U, 4U};
static const SsEntityId tetra_cell_ids[] = {UINT64_C(0x620000000001),
                                            UINT64_C(0x620000000002)};
static const uint64_t tetra_facets[] = {0U, 1U, 2U, 0U, 1U, 3U, 0U, 2U, 3U,
                                        1U, 2U, 4U, 1U, 3U, 4U, 2U, 3U, 4U};
static const SsEntityId tetra_facet_ids[] = {
    UINT64_C(0x630000000001), UINT64_C(0x630000000002), UINT64_C(0x630000000003),
    UINT64_C(0x630000000004), UINT64_C(0x630000000005), UINT64_C(0x630000000006)};
#define TETRA_EDGE(id, a, b) {1U, UINT64_C(id), {UINT64_C(a), UINT64_C(b), 0U}}
#define TETRA_FACE(id, a, b, c)                                                        \
  {                                                                                    \
    2U, UINT64_C(id), { UINT64_C(a), UINT64_C(b), UINT64_C(c) }                        \
  }
static const SsMeshIntermediateEntity tetra_entities[] = {
    TETRA_EDGE(0x640000000001, 0x610000000001, 0x610000000002),
    TETRA_EDGE(0x640000000002, 0x610000000001, 0x610000000003),
    TETRA_EDGE(0x640000000003, 0x610000000001, 0x610000000004),
    TETRA_EDGE(0x640000000004, 0x610000000002, 0x610000000003),
    TETRA_EDGE(0x640000000005, 0x610000000002, 0x610000000004),
    TETRA_EDGE(0x640000000006, 0x610000000003, 0x610000000004),
    TETRA_EDGE(0x640000000007, 0x610000000002, 0x610000000005),
    TETRA_EDGE(0x640000000008, 0x610000000003, 0x610000000005),
    TETRA_EDGE(0x640000000009, 0x610000000004, 0x610000000005),
    TETRA_FACE(0x630000000001, 0x610000000001, 0x610000000002, 0x610000000003),
    TETRA_FACE(0x630000000002, 0x610000000001, 0x610000000002, 0x610000000004),
    TETRA_FACE(0x630000000003, 0x610000000001, 0x610000000003, 0x610000000004),
    TETRA_FACE(0x650000000001, 0x610000000002, 0x610000000003, 0x610000000004),
    TETRA_FACE(0x630000000004, 0x610000000002, 0x610000000003, 0x610000000005),
    TETRA_FACE(0x630000000005, 0x610000000002, 0x610000000004, 0x610000000005),
    TETRA_FACE(0x630000000006, 0x610000000003, 0x610000000004, 0x610000000005),
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

typedef struct {
  int pressure;
  int follower;
  double traction[3];
  double pressure_value;
  double spring;
} NaturalLoad;

/* A constant traction, a normal pressure p n, or an elastic support t0 - k u
 * (symmetric derivative, conservative).  The follower variant adds a skew
 * field-dependent part, for which no potential exists. */
static SsStatus natural_kernel(const void *context, const SsNaturalPointInput *input,
                               SsNaturalPointOutput *output) {
  const NaturalLoad *load = (const NaturalLoad *)context;
  const uint32_t count = input->components;
  uint32_t component;
  memset(output, 0, sizeof(*output));
  for (component = 0U; component < count; ++component) {
    output->load[component] =
        load->pressure != 0 ? load->pressure_value * input->outward_normal[component]
                            : load->traction[component];
    output->load[component] -= load->spring * input->value[component];
    output->load_derivative[component * count + component] = -load->spring;
  }
  if (load->follower != 0) {
    output->load[0] += input->value[1];
    output->load[1] -= input->value[0];
    output->load_derivative[1] = 1.0;
    output->load_derivative[count] = -1.0;
  }
  return SS_OK;
}

static SsMeshInput mesh_input(uint32_t dimension) {
  SsMeshInput input;
  memset(&input, 0, sizeof(input));
  input.dimension = dimension;
  input.coordinate_order = 1U;
  input.absolute_determinant_tolerance = 1.0e-12;
  input.relative_determinant_tolerance = 1.0e-12;
  if (dimension == 2U) {
    input.global_vertex_count = input.vertex_count = 4U;
    input.global_cell_count = input.cell_count = 2U;
    input.global_boundary_facet_count = input.boundary_facet_count = 4U;
    input.global_intermediate_entity_count = input.intermediate_entity_count =
        sizeof(square_edges) / sizeof(square_edges[0]);
    input.global_coordinate_node_count = input.coordinate_node_count = 4U;
    input.vertex_ids = square_vertex_ids;
    input.vertex_coordinates = square_coordinates;
    input.cell_ids = square_cell_ids;
    input.cells = square_cells;
    input.boundary_facet_ids = square_facet_ids;
    input.boundary_facets = square_facets;
    input.intermediate_entities = square_edges;
    input.label_count = sizeof(square_labels) / sizeof(square_labels[0]);
    input.labels = square_labels;
    input.coordinate_node_ids = square_vertex_ids;
    input.coordinate_coordinates = square_coordinates;
    input.cell_coordinate_nodes = square_cells;
  } else {
    input.global_vertex_count = input.vertex_count = 5U;
    input.global_cell_count = input.cell_count = 2U;
    input.global_boundary_facet_count = input.boundary_facet_count = 6U;
    input.global_intermediate_entity_count = input.intermediate_entity_count =
        sizeof(tetra_entities) / sizeof(tetra_entities[0]);
    input.global_coordinate_node_count = input.coordinate_node_count = 5U;
    input.vertex_ids = tetra_vertex_ids;
    input.vertex_coordinates = tetra_coordinates;
    input.cell_ids = tetra_cell_ids;
    input.cells = tetra_cells;
    input.boundary_facet_ids = tetra_facet_ids;
    input.boundary_facets = tetra_facets;
    input.intermediate_entities = tetra_entities;
    input.label_count = sizeof(tetra_labels) / sizeof(tetra_labels[0]);
    input.labels = tetra_labels;
    input.coordinate_node_ids = tetra_vertex_ids;
    input.coordinate_coordinates = tetra_coordinates;
    input.cell_coordinate_nodes = tetra_cells;
  }
  return input;
}

/* Deterministic pseudo-random values in [-1, 1]. */
static double next_value(uint64_t *seed) {
  *seed = *seed * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
  return (double)(*seed >> 11) / (double)(UINT64_C(1) << 52) - 1.0;
}

static int fill_vector(Vec vector, double scale, uint64_t seed) {
  PetscScalar *values = NULL;
  PetscInt size = 0;
  PetscInt index;
  CHECK(VecGetLocalSize(vector, &size) == PETSC_SUCCESS);
  CHECK(VecGetArray(vector, &values) == PETSC_SUCCESS);
  for (index = 0; index < size; ++index) {
    values[index] = scale * next_value(&seed);
  }
  CHECK(VecRestoreArray(vector, &values) == PETSC_SUCCESS);
  return 0;
}

static int functional_at(SsOperator *operator_handle, Vec base, Vec direction,
                         double step, double load_factor, double *value) {
  SsOperatorState state;
  SsErrorRecord error = {0};
  Vec shifted = NULL;
  CHECK(VecDuplicate(base, &shifted) == PETSC_SUCCESS);
  CHECK(VecWAXPY(shifted, step, direction, base) == PETSC_SUCCESS);
  memset(&state, 0, sizeof(state));
  state.field = shifted;
  state.reduction_factor = 1.0;
  state.external_load_factor = load_factor;
  CHECK(ss_operator_functional(operator_handle, &state, value, &error) == SS_OK);
  CHECK(VecDestroy(&shifted) == PETSC_SUCCESS);
  return 0;
}

static int check_energy_gradient(uint32_t dimension, uint32_t order,
                                 SsAssemblyDeterminism determinism, int matrix_free) {
  const SsMcParameterInput material_input = {
      1.0e4, 0.27,
      40.0,  0.50,
      0.15,  SS_MC_DAVIS_B,
      1U,    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"};
  const NaturalLoad traction = {0, 0, {3.0, -7.0, 2.0}, 0.0, 0.0};
  const NaturalLoad pressure = {1, 0, {0.0, 0.0, 0.0}, -11.0, 0.0};
  const NaturalLoad support = {0, 0, {0.5, 1.0, -0.25}, 0.0, 4.0e3};
  const double load_factor = 1.7;
  SsNaturalOperatorSpec natural[3];
  SsMeshInput input = mesh_input(dimension);
  SsFieldSpec field_spec;
  SsMeshError mesh_error = {0};
  SsErrorRecord error = {0};
  SsMcParameters *material = NULL;
  SsMesh *mesh = NULL;
  SsFieldSpace *space = NULL;
  SsAssemblyPlan *plan = NULL;
  SsOperator *operator_handle = NULL;
  SsMechanicsMaterialRegion region;
  SsMechanicsOperatorSpec mechanics;
  SsOperatorSpec spec;
  SsOperatorState state;
  Vec base = NULL;
  Vec direction = NULL;
  Vec residual = NULL;
  double plus = 0.0;
  double minus = 0.0;
  double fused = 0.0;
  double alone = 0.0;
  PetscScalar action = 0.0;
  memset(&field_spec, 0, sizeof(field_spec));
  field_spec.order = order;
  field_spec.components = dimension;
  field_spec.constraint_count = dimension == 2U ? 3U : 6U;
  field_spec.constraints = dimension == 2U ? square_constraints : tetra_constraints;
  natural[0] =
      (SsNaturalOperatorSpec){dimension == 2U ? 2U : 4U, natural_kernel, &traction, 1};
  natural[1] =
      (SsNaturalOperatorSpec){dimension == 2U ? 3U : 6U, natural_kernel, &pressure, 1};
  natural[2] =
      (SsNaturalOperatorSpec){dimension == 2U ? 4U : 5U, natural_kernel, &support, 1};
  CHECK(ss_mc_parameters_create(&material_input, &material, NULL) == SS_OK);
  CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &mesh_error) == SS_OK);
  CHECK(ss_mesh_field_space_create(mesh, &field_spec, &space) == SS_OK);
  CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, space, &plan, &error) == SS_OK);
  memset(&region, 0, sizeof(region));
  region.material_value = 1U;
  region.parameters = material;
  memset(&mechanics, 0, sizeof(mechanics));
  mechanics.regions = &region;
  mechanics.region_count = 1U;
  mechanics.body_force[dimension - 1U] = -15.0;
  memset(&spec, 0, sizeof(spec));
  spec.physics = SS_ASSEMBLY_MECHANICS;
  spec.determinism = determinism;
  spec.openmp_work_threshold = 1U;
  spec.requested_threads = 1;
  spec.mechanics = &mechanics;
  spec.natural_operations = natural;
  spec.natural_operation_count = 3U;
  spec.matrix_free = matrix_free;
  CHECK(ss_operator_create(plan, &spec, &operator_handle, &error) == SS_OK);
  CHECK(ss_assembly_plan_create_vector(plan, &base) == SS_OK);
  CHECK(VecDuplicate(base, &direction) == PETSC_SUCCESS);
  CHECK(VecDuplicate(base, &residual) == PETSC_SUCCESS);
  /* Displacements of order 1e-6 keep every point elastic (c = 40 kPa). */
  CHECK(fill_vector(base, 1.0e-6, 17U + order) == 0);
  CHECK(fill_vector(direction, 1.0e-6, 91U + dimension) == 0);
  memset(&state, 0, sizeof(state));
  state.field = base;
  state.reduction_factor = 1.0;
  state.external_load_factor = load_factor;
  CHECK(ss_operator_residual_functional(operator_handle, &state, residual, &fused,
                                        &error) == SS_OK);
  CHECK(ss_operator_functional(operator_handle, &state, &alone, &error) == SS_OK);
  CHECK(VecDot(residual, direction, &action) == PETSC_SUCCESS);
  CHECK(functional_at(operator_handle, base, direction, 1.0, load_factor, &plus) == 0);
  CHECK(functional_at(operator_handle, base, direction, -1.0, load_factor, &minus) ==
        0);
  {
    const double central = 0.5 * (plus - minus);
    const double scale = fmax(fabs((double)action), 1.0e-30);
    if (!(fabs(central - (double)action) <= 1.0e-8 * scale) ||
        !(fabs(fused - alone) <= 1.0e-12 * fmax(fabs(fused), 1.0e-30))) {
      fprintf(stderr,
              "d=%u P%u determinism=%d matrix_free=%d: dPi.v=%.17g R.v=%.17g "
              "(relative %.3e), fused=%.17g alone=%.17g\n",
              dimension, order, (int)determinism, matrix_free, central, (double)action,
              fabs(central - (double)action) / scale, fused, alone);
      return 1;
    }
  }
  CHECK(VecDestroy(&residual) == PETSC_SUCCESS);
  CHECK(VecDestroy(&direction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&base) == PETSC_SUCCESS);
  CHECK(ss_operator_destroy(&operator_handle) == SS_OK);
  CHECK(ss_assembly_plan_destroy(&plan) == SS_OK);
  CHECK(ss_field_space_destroy(&space) == SS_OK);
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  CHECK(ss_mc_parameters_destroy(&material) == SS_OK);
  return 0;
}

/* A displacement-dependent natural load has no potential; requesting the
 * functional must fail instead of returning an inconsistent energy. */
static int check_nonconservative_rejected(void) {
  const SsMcParameterInput material_input = {
      1.0e4, 0.27,
      40.0,  0.50,
      0.15,  SS_MC_DAVIS_B,
      1U,    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"};
  const NaturalLoad follower = {0, 1, {3.0, -7.0, 0.0}, 0.0, 0.0};
  SsNaturalOperatorSpec natural = {2U, natural_kernel, &follower, 1};
  SsMeshInput input = mesh_input(2U);
  SsFieldSpec field_spec;
  SsMeshError mesh_error = {0};
  SsErrorRecord error = {0};
  SsMcParameters *material = NULL;
  SsMesh *mesh = NULL;
  SsFieldSpace *space = NULL;
  SsAssemblyPlan *plan = NULL;
  SsOperator *operator_handle = NULL;
  SsMechanicsMaterialRegion region;
  SsMechanicsOperatorSpec mechanics;
  SsOperatorSpec spec;
  SsOperatorState state;
  Vec base = NULL;
  double value = 0.0;
  memset(&field_spec, 0, sizeof(field_spec));
  field_spec.order = 1U;
  field_spec.components = 2U;
  field_spec.constraint_count = 3U;
  field_spec.constraints = square_constraints;
  CHECK(ss_mc_parameters_create(&material_input, &material, NULL) == SS_OK);
  CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &mesh_error) == SS_OK);
  CHECK(ss_mesh_field_space_create(mesh, &field_spec, &space) == SS_OK);
  CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, space, &plan, &error) == SS_OK);
  memset(&region, 0, sizeof(region));
  region.material_value = 1U;
  region.parameters = material;
  memset(&mechanics, 0, sizeof(mechanics));
  mechanics.regions = &region;
  mechanics.region_count = 1U;
  memset(&spec, 0, sizeof(spec));
  spec.physics = SS_ASSEMBLY_MECHANICS;
  spec.determinism = SS_ASSEMBLY_QUALIFICATION;
  spec.openmp_work_threshold = 1U;
  spec.requested_threads = 1;
  spec.mechanics = &mechanics;
  spec.natural_operations = &natural;
  spec.natural_operation_count = 1U;
  CHECK(ss_operator_create(plan, &spec, &operator_handle, &error) == SS_OK);
  CHECK(ss_assembly_plan_create_vector(plan, &base) == SS_OK);
  CHECK(VecSet(base, 0.0) == PETSC_SUCCESS);
  memset(&state, 0, sizeof(state));
  state.field = base;
  state.reduction_factor = 1.0;
  state.external_load_factor = 1.0;
  CHECK(ss_operator_functional(operator_handle, &state, &value, &error) != SS_OK);
  CHECK(strcmp(error.code, "ASM_NATURAL_NONCONSERVATIVE") == 0);
  CHECK(VecDestroy(&base) == PETSC_SUCCESS);
  CHECK(ss_operator_destroy(&operator_handle) == SS_OK);
  CHECK(ss_assembly_plan_destroy(&plan) == SS_OK);
  CHECK(ss_field_space_destroy(&space) == SS_OK);
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  CHECK(ss_mc_parameters_destroy(&material) == SS_OK);
  return 0;
}

static int test_natural_energy(void) {
  uint32_t dimension;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    uint32_t order;
    for (order = 1U; order <= 2U; ++order) {
      CHECK(check_energy_gradient(dimension, order, SS_ASSEMBLY_QUALIFICATION, 0) == 0);
      CHECK(check_energy_gradient(dimension, order, SS_ASSEMBLY_PERFORMANCE, 0) == 0);
    }
  }
  CHECK(check_energy_gradient(3U, 2U, SS_ASSEMBLY_PERFORMANCE, 1) == 0);
  return check_nonconservative_rejected();
}

int main(int argc, char **argv) {
  int result = 2;
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 3;
  }
  if (argc == 2 && strcmp(argv[1], "ASM-ORACLE-NATURAL-ENERGY-001") == 0) {
    result = test_natural_energy();
  } else {
    fprintf(stderr, "usage: %s ASM-ORACLE-NATURAL-ENERGY-001\n", argv[0]);
  }
  if (PetscFinalize() != PETSC_SUCCESS) {
    return 3;
  }
  return result;
}
