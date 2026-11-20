/* Curved high-order mechanics operators on a distributed tetrahedral cube.
 *
 * The cube [0, 1]^3 is split into n^3 cells of six Kuhn tetrahedra whose G2 or
 * G4 coordinate nodes interpolate a smooth, orientation-preserving map, so all
 * interior cells are genuinely curved.  The oracle mode compares the
 * matrix-free tangent action with the assembled Jacobian and the two residual
 * paths.  The benchmark mode times operator setup, linearization, tangent
 * action and tangent assembly and reports resident and accounted bytes.
 *
 * Usage: curved_operator_test ASM-ORACLE-CURVED-MATFREE-001
 *        curved_operator_test bench N FIELD_ORDER GEOMETRY_ORDER THREADS REPEATS
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/assembly/assembly.h"
#include "slope_internal/build/config.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/material/material.h"
#include "slope_internal/mesh/mesh.h"

#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

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

#define VERTEX_ID(index) (UINT64_C(0x710000000000) + (uint64_t)(index))
#define NODE_ID(index) (UINT64_C(0x720000000000) + (uint64_t)(index))
#define CELL_ID(index) (UINT64_C(0x730000000000) + (uint64_t)(index))
#define EDGE_ID(a, b, nv) (UINT64_C(0x740000000000) + (uint64_t)(a) * (nv) + (b))
#define FACE_ID(a, b, c, nv)                                                           \
  (UINT64_C(0x750000000000) + ((uint64_t)(a) * (nv) + (b)) * (nv) + (c))

typedef struct {
  SsMeshInput input;
  SsEntityId *vertex_ids;
  double *vertex_coordinates;
  SsEntityId *cell_ids;
  uint64_t *cells;
  SsEntityId *facet_ids;
  uint64_t *facets;
  SsMeshIntermediateEntity *entities;
  SsMeshLabelAssignment *labels;
  SsEntityId *node_ids;
  double *node_coordinates;
  uint64_t *cell_nodes;
} CubeMesh;

static const int kuhn_permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                            {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

static void curved_map(const double *reference, double *physical) {
  const double pi = 3.14159265358979323846;
  physical[0] = reference[0] + 0.05 * sin(2.0 * pi * reference[1]);
  physical[1] = reference[1] + 0.05 * sin(2.0 * pi * reference[2]);
  physical[2] = reference[2] + 0.05 * sin(2.0 * pi * reference[0]);
}

static void owned_range(size_t count, int rank, int size, size_t *start,
                        size_t *length) {
  const size_t base = count / (size_t)size;
  const size_t extra = count % (size_t)size;
  *start = (size_t)rank * base + ((size_t)rank < extra ? (size_t)rank : extra);
  *length = base + ((size_t)rank < extra ? 1U : 0U);
}

/* Lattice offsets (0/1 per axis) of the four vertices of Kuhn tetrahedron t,
 * ordered for a positive determinant. */
static void kuhn_offsets(int t, int offsets[4][3]) {
  int vertex;
  memset(offsets, 0, 4U * 3U * sizeof(int));
  for (vertex = 1; vertex < 4; ++vertex) {
    memcpy(offsets[vertex], offsets[vertex - 1], 3U * sizeof(int));
    offsets[vertex][kuhn_permutations[t][vertex - 1]] = 1;
  }
  {
    const int a[3] = {offsets[1][0], offsets[1][1], offsets[1][2]};
    const int b[3] = {offsets[2][0], offsets[2][1], offsets[2][2]};
    const int c[3] = {offsets[3][0], offsets[3][1], offsets[3][2]};
    const int determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) -
                            a[1] * (b[0] * c[2] - b[2] * c[0]) +
                            a[2] * (b[0] * c[1] - b[1] * c[0]);
    if (determinant < 0) {
      int swap[3];
      memcpy(swap, offsets[1], sizeof(swap));
      memcpy(offsets[1], offsets[2], sizeof(swap));
      memcpy(offsets[2], swap, sizeof(swap));
    }
  }
}

static int compare_u64(const void *left, const void *right) {
  const uint64_t a = *(const uint64_t *)left;
  const uint64_t b = *(const uint64_t *)right;
  return a < b ? -1 : (a > b ? 1 : 0);
}

/* Grid coordinate of a vertex along one axis. */
static uint64_t vertex_axis(uint64_t vertex, size_t axis, size_t nv1) {
  return axis == 0U ? vertex % nv1
                    : (axis == 1U ? (vertex / nv1) % nv1 : vertex / (nv1 * nv1));
}

static int compare_entity(const void *left, const void *right) {
  return compare_u64(&((const SsMeshIntermediateEntity *)left)->stable_entity_id,
                     &((const SsMeshIntermediateEntity *)right)->stable_entity_id);
}

static void sort3(uint64_t *v) { ss_sort(v, 3U, sizeof(*v), compare_u64); }

static void cube_mesh_destroy(CubeMesh *mesh) {
  free(mesh->vertex_ids);
  free(mesh->vertex_coordinates);
  free(mesh->cell_ids);
  free(mesh->cells);
  free(mesh->facet_ids);
  free(mesh->facets);
  free(mesh->entities);
  free(mesh->labels);
  free(mesh->node_ids);
  free(mesh->node_coordinates);
  free(mesh->cell_nodes);
  memset(mesh, 0, sizeof(*mesh));
}

static int cube_mesh_build(MPI_Comm communicator, size_t n, uint32_t geometry_order,
                           CubeMesh *mesh) {
  const SsFeDescriptor *descriptor = NULL;
  const size_t nv1 = n + 1U;
  const uint64_t vertex_total = (uint64_t)(nv1 * nv1 * nv1);
  const size_t lattice1 = geometry_order * n + 1U;
  const size_t node_total = lattice1 * lattice1 * lattice1;
  const size_t cell_total = 6U * n * n * n;
  size_t cell_start = 0U;
  size_t cell_count = 0U;
  size_t vertex_start = 0U;
  size_t vertex_count = 0U;
  size_t node_start = 0U;
  size_t node_count = 0U;
  size_t entity_count = 0U;
  size_t facet_count = 0U;
  size_t label_count = 0U;
  uint64_t *keys = NULL;
  uint64_t unique_entities = 0U;
  uint64_t boundary_total = 0U;
  int rank = 0;
  int size = 1;
  memset(mesh, 0, sizeof(*mesh));
  CHECK(MPI_Comm_rank(communicator, &rank) == MPI_SUCCESS);
  CHECK(MPI_Comm_size(communicator, &size) == MPI_SUCCESS);
  CHECK(ss_fe_descriptor_find(3U, geometry_order, &descriptor) == SS_OK);
  owned_range(cell_total, rank, size, &cell_start, &cell_count);
  owned_range((size_t)vertex_total, rank, size, &vertex_start, &vertex_count);
  owned_range(node_total, rank, size, &node_start, &node_count);
  mesh->vertex_ids = calloc(vertex_count + 1U, sizeof(*mesh->vertex_ids));
  mesh->vertex_coordinates = calloc(3U * vertex_count + 1U, sizeof(double));
  mesh->cell_ids = calloc(cell_count + 1U, sizeof(*mesh->cell_ids));
  mesh->cells = calloc(4U * cell_count + 1U, sizeof(*mesh->cells));
  mesh->facet_ids = calloc(4U * cell_count + 1U, sizeof(*mesh->facet_ids));
  mesh->facets = calloc(12U * cell_count + 1U, sizeof(*mesh->facets));
  mesh->entities = calloc(10U * cell_count + 1U, sizeof(*mesh->entities));
  mesh->labels = calloc(5U * cell_count + 1U, sizeof(*mesh->labels));
  mesh->node_ids = calloc(node_count + 1U, sizeof(*mesh->node_ids));
  mesh->node_coordinates = calloc(3U * node_count + 1U, sizeof(double));
  mesh->cell_nodes =
      calloc(descriptor->node_count * cell_count + 1U, sizeof(*mesh->cell_nodes));
  keys = calloc(10U * cell_total, sizeof(*keys));
  CHECK(mesh->vertex_ids != NULL && mesh->vertex_coordinates != NULL &&
        mesh->cell_ids != NULL && mesh->cells != NULL && mesh->facet_ids != NULL &&
        mesh->facets != NULL && mesh->entities != NULL && mesh->labels != NULL &&
        mesh->node_ids != NULL && mesh->node_coordinates != NULL &&
        mesh->cell_nodes != NULL && keys != NULL);
  for (size_t v = 0U; v < vertex_count; ++v) {
    const size_t global = vertex_start + v;
    const double reference[3] = {(double)(global % nv1) / (double)n,
                                 (double)((global / nv1) % nv1) / (double)n,
                                 (double)(global / (nv1 * nv1)) / (double)n};
    mesh->vertex_ids[v] = VERTEX_ID(global);
    curved_map(reference, mesh->vertex_coordinates + 3U * v);
  }
  for (size_t node = 0U; node < node_count; ++node) {
    const size_t global = node_start + node;
    const size_t i = global % lattice1;
    const size_t j = (global / lattice1) % lattice1;
    const size_t k = global / (lattice1 * lattice1);
    const double reference[3] = {(double)i / (double)(lattice1 - 1U),
                                 (double)j / (double)(lattice1 - 1U),
                                 (double)k / (double)(lattice1 - 1U)};
    if (i % geometry_order == 0U && j % geometry_order == 0U &&
        k % geometry_order == 0U) {
      mesh->node_ids[node] = VERTEX_ID(
          (k / geometry_order * nv1 + j / geometry_order) * nv1 + i / geometry_order);
    } else {
      mesh->node_ids[node] = NODE_ID(global);
    }
    curved_map(reference, mesh->node_coordinates + 3U * node);
  }
  /* Global unique edge and face counts from every cell, on every rank. */
  for (size_t cell = 0U; cell < cell_total; ++cell) {
    const size_t cube = cell / 6U;
    const size_t origin[3] = {cube % n, (cube / n) % n, cube / (n * n)};
    int offsets[4][3];
    uint64_t vertex[4];
    size_t a;
    size_t slot = 10U * cell;
    kuhn_offsets((int)(cell % 6U), offsets);
    for (a = 0U; a < 4U; ++a) {
      vertex[a] = ((origin[2] + (size_t)offsets[a][2]) * nv1 + origin[1] +
                   (size_t)offsets[a][1]) *
                      nv1 +
                  origin[0] + (size_t)offsets[a][0];
    }
    for (a = 0U; a < 4U; ++a) {
      size_t b;
      for (b = a + 1U; b < 4U; ++b) {
        const uint64_t lo = vertex[a] < vertex[b] ? vertex[a] : vertex[b];
        const uint64_t hi = vertex[a] < vertex[b] ? vertex[b] : vertex[a];
        keys[slot++] = EDGE_ID(lo, hi, vertex_total);
      }
    }
    for (a = 0U; a < 4U; ++a) {
      uint64_t face[3];
      size_t b;
      size_t f = 0U;
      int boundary = 0;
      for (b = 0U; b < 4U; ++b) {
        if (b != a) {
          face[f++] = vertex[b];
        }
      }
      sort3(face);
      keys[slot++] = FACE_ID(face[0], face[1], face[2], vertex_total);
      for (b = 0U; b < 3U; ++b) {
        const uint64_t value = vertex_axis(face[0], b, nv1);
        if ((value == 0U || value == n) && vertex_axis(face[1], b, nv1) == value &&
            vertex_axis(face[2], b, nv1) == value) {
          boundary = 1;
        }
      }
      if (boundary != 0) {
        ++boundary_total;
        if (cell >= cell_start && cell < cell_start + cell_count) {
          const size_t local = facet_count++;
          mesh->facet_ids[local] = FACE_ID(face[0], face[1], face[2], vertex_total);
          mesh->facets[3U * local] = face[0];
          mesh->facets[3U * local + 1U] = face[1];
          mesh->facets[3U * local + 2U] = face[2];
          mesh->labels[label_count++] = (SsMeshLabelAssignment){
              SS_MESH_LABEL_SUPPORT, mesh->facet_ids[local], 1U};
        }
      }
    }
    if (cell >= cell_start && cell < cell_start + cell_count) {
      const size_t local = cell - cell_start;
      size_t node;
      mesh->cell_ids[local] = CELL_ID(cell);
      for (a = 0U; a < 4U; ++a) {
        mesh->cells[4U * local + a] = vertex[a];
      }
      for (a = 0U; a < 4U; ++a) {
        size_t b;
        for (b = a + 1U; b < 4U; ++b) {
          const uint64_t lo = vertex[a] < vertex[b] ? vertex[a] : vertex[b];
          const uint64_t hi = vertex[a] < vertex[b] ? vertex[b] : vertex[a];
          mesh->entities[entity_count++] = (SsMeshIntermediateEntity){
              1U, EDGE_ID(lo, hi, vertex_total), {VERTEX_ID(lo), VERTEX_ID(hi), 0U}};
        }
      }
      for (a = 0U; a < 4U; ++a) {
        uint64_t face[3];
        size_t b;
        size_t f = 0U;
        for (b = 0U; b < 4U; ++b) {
          if (b != a) {
            face[f++] = vertex[b];
          }
        }
        sort3(face);
        mesh->entities[entity_count++] = (SsMeshIntermediateEntity){
            2U,
            FACE_ID(face[0], face[1], face[2], vertex_total),
            {VERTEX_ID(face[0]), VERTEX_ID(face[1]), VERTEX_ID(face[2])}};
      }
      mesh->labels[label_count++] =
          (SsMeshLabelAssignment){SS_MESH_LABEL_MATERIAL, CELL_ID(cell), 1U};
      for (node = 0U; node < descriptor->node_count; ++node) {
        size_t lattice[3];
        size_t axis;
        for (axis = 0U; axis < 3U; ++axis) {
          size_t sum = geometry_order * origin[axis];
          for (a = 0U; a < 4U; ++a) {
            sum += (size_t)descriptor->barycentric_indices[node * 4U + a] *
                   (size_t)offsets[a][axis];
          }
          lattice[axis] = sum;
        }
        mesh->cell_nodes[descriptor->node_count * local + node] =
            (uint64_t)((lattice[2] * lattice1 + lattice[1]) * lattice1 + lattice[0]);
      }
    }
  }
  ss_sort(keys, 10U * cell_total, sizeof(*keys), compare_u64);
  for (size_t key = 0U; key < 10U * cell_total; ++key) {
    if (key == 0U || keys[key] != keys[key - 1U]) {
      ++unique_entities;
    }
  }
  free(keys);
  /* Each rank lists every edge and face of its cells exactly once. */
  ss_sort(mesh->entities, entity_count, sizeof(*mesh->entities), compare_entity);
  {
    size_t unique = 0U;
    for (size_t entity = 0U; entity < entity_count; ++entity) {
      if (unique == 0U || mesh->entities[entity].stable_entity_id !=
                              mesh->entities[unique - 1U].stable_entity_id) {
        mesh->entities[unique++] = mesh->entities[entity];
      }
    }
    entity_count = unique;
  }
  mesh->input = (SsMeshInput){
      .dimension = 3U,
      .global_vertex_count = vertex_total,
      .global_cell_count = cell_total,
      .global_boundary_facet_count = boundary_total,
      .global_intermediate_entity_count = unique_entities,
      .global_coordinate_node_count = node_total,
      .vertex_count = vertex_count,
      .cell_count = cell_count,
      .boundary_facet_count = facet_count,
      .vertex_ids = mesh->vertex_ids,
      .vertex_coordinates = mesh->vertex_coordinates,
      .cell_ids = mesh->cell_ids,
      .cells = mesh->cells,
      .boundary_facet_ids = mesh->facet_ids,
      .boundary_facets = mesh->facets,
      .intermediate_entity_count = entity_count,
      .intermediate_entities = mesh->entities,
      .label_count = label_count,
      .labels = mesh->labels,
      .coordinate_order = geometry_order,
      .coordinate_node_count = node_count,
      .coordinate_node_ids = mesh->node_ids,
      .coordinate_coordinates = mesh->node_coordinates,
      .cell_coordinate_nodes = mesh->cell_nodes,
      .absolute_determinant_tolerance = 1.0e-12,
      .relative_determinant_tolerance = 1.0e-12,
  };
  return 0;
}

typedef struct {
  SsMesh *mesh;
  SsFieldSpace *space;
  SsAssemblyPlan *plan;
  SsMcParameters *material;
  SsMechanicsMaterialRegion region;
  SsMechanicsOperatorSpec mechanics;
  SsOperator *assembled;
  SsOperator *matrix_free;
  Mat jacobian;
  Mat shell;
  Vec state;
} CurvedProblem;

static uint64_t next_seed(uint64_t seed) {
  return seed * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
}

static int fill_vector(Vec vector, double scale, uint64_t seed) {
  PetscScalar *values = NULL;
  PetscInt low = 0;
  PetscInt high = 0;
  PetscInt index;
  CHECK(VecGetOwnershipRange(vector, &low, &high) == PETSC_SUCCESS);
  CHECK(VecGetArray(vector, &values) == PETSC_SUCCESS);
  for (index = low; index < high; ++index) {
    /* Depends only on the global index: identical for every layout. */
    const uint64_t value = next_seed(seed ^ ((uint64_t)index * UINT64_C(2654435761)));
    values[index - low] =
        scale * ((double)(value >> 11) / (double)(UINT64_C(1) << 52) - 1.0);
  }
  CHECK(VecRestoreArray(vector, &values) == PETSC_SUCCESS);
  return 0;
}

static double resident_megabytes(void) {
  FILE *status = fopen("/proc/self/status", "r");
  char line[256];
  double value = 0.0;
  if (status == NULL) {
    return 0.0;
  }
  while (fgets(line, sizeof(line), status) != NULL) {
    if (strncmp(line, "VmRSS:", 6U) == 0) {
      value = strtod(line + 6, NULL) / 1024.0;
      break;
    }
  }
  (void)fclose(status);
  return value;
}

static int problem_create(MPI_Comm communicator, size_t n, uint32_t field_order,
                          uint32_t geometry_order, int threads, int with_assembled,
                          CurvedProblem *problem, double *resident_delta) {
  const SsMcParameterInput material_input = {
      2.0e4, 0.3,
      15.0,  0.6,
      0.0,   SS_MC_DAVIS_B,
      1U,    "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"};
  CubeMesh input;
  SsFieldSpec field_spec;
  SsMeshError mesh_error = {0};
  SsErrorRecord error = {0};
  SsOperatorSpec spec;
  double before;
  memset(problem, 0, sizeof(*problem));
  CHECK(cube_mesh_build(communicator, n, geometry_order, &input) == 0);
  CHECK(ss_mc_parameters_create(&material_input, &problem->material, NULL) == SS_OK);
  {
    const SsStatus mesh_status =
        ss_mesh_create(communicator, &input.input, &problem->mesh, &mesh_error);
    if (mesh_status != SS_OK) {
      fprintf(stderr, "mesh status %d code %s entity %llx\n", (int)mesh_status,
              mesh_error.code, (unsigned long long)mesh_error.stable_entity_id);
      return 1;
    }
  }
  cube_mesh_destroy(&input);
  memset(&field_spec, 0, sizeof(field_spec));
  field_spec.order = field_order;
  field_spec.components = 3U;
  CHECK(ss_mesh_field_space_create(problem->mesh, &field_spec, &problem->space) ==
        SS_OK);
  CHECK(ss_assembly_plan_create(communicator, problem->space, &problem->plan, &error) ==
        SS_OK);
  problem->region.material_value = 1U;
  problem->region.parameters = problem->material;
  problem->mechanics.regions = &problem->region;
  problem->mechanics.region_count = 1U;
  problem->mechanics.body_force[2] = -20.0;
  memset(&spec, 0, sizeof(spec));
  spec.physics = SS_ASSEMBLY_MECHANICS;
  spec.determinism = SS_ASSEMBLY_PERFORMANCE;
  spec.openmp_work_threshold = 1U;
  spec.requested_threads = threads;
  spec.mechanics = &problem->mechanics;
  spec.memory_cap_bytes = (size_t)UINT64_C(8) << 30U;
  CHECK(ss_assembly_plan_create_vector(problem->plan, &problem->state) == SS_OK);
  before = resident_megabytes();
  spec.matrix_free = 1;
  CHECK(ss_operator_create(problem->plan, &spec, &problem->matrix_free, &error) ==
        SS_OK);
  CHECK(ss_operator_matrix_free_create(problem->matrix_free, &problem->shell, &error) ==
        SS_OK);
  *resident_delta = resident_megabytes() - before;
  if (with_assembled != 0) {
    spec.matrix_free = 0;
    CHECK(ss_operator_create(problem->plan, &spec, &problem->assembled, &error) ==
          SS_OK);
    CHECK(ss_assembly_plan_create_matrix(problem->plan, &problem->jacobian) == SS_OK);
  }
  return 0;
}

static int problem_destroy(CurvedProblem *problem) {
  CHECK(MatDestroy(&problem->shell) == PETSC_SUCCESS);
  CHECK(MatDestroy(&problem->jacobian) == PETSC_SUCCESS);
  CHECK(VecDestroy(&problem->state) == PETSC_SUCCESS);
  CHECK(ss_operator_destroy(&problem->matrix_free) == SS_OK);
  if (problem->assembled != NULL) {
    CHECK(ss_operator_destroy(&problem->assembled) == SS_OK);
  }
  CHECK(ss_assembly_plan_destroy(&problem->plan) == SS_OK);
  CHECK(ss_field_space_destroy(&problem->space) == SS_OK);
  CHECK(ss_mesh_destroy(&problem->mesh) == SS_OK);
  CHECK(ss_mc_parameters_destroy(&problem->material) == SS_OK);
  return 0;
}

static double relative_difference(Vec actual, Vec expected) {
  Vec difference = NULL;
  PetscReal numerator = 0.0;
  PetscReal denominator = 0.0;
  if (VecDuplicate(actual, &difference) != PETSC_SUCCESS ||
      VecWAXPY(difference, -1.0, expected, actual) != PETSC_SUCCESS ||
      VecNorm(difference, NORM_2, &numerator) != PETSC_SUCCESS ||
      VecNorm(expected, NORM_2, &denominator) != PETSC_SUCCESS) {
    return INFINITY;
  }
  (void)VecDestroy(&difference);
  return (double)(numerator / (denominator > 0.0 ? denominator : 1.0));
}

static int check_curved_case(uint32_t field_order, uint32_t geometry_order,
                             int threads) {
  CurvedProblem problem;
  SsOperatorState state;
  SsErrorRecord error = {0};
  Vec direction = NULL;
  Vec shell_action = NULL;
  Vec assembled_action = NULL;
  Vec shell_residual = NULL;
  Vec assembled_residual = NULL;
  double resident = 0.0;
  double action_error;
  double residual_error;
  CHECK(problem_create(PETSC_COMM_WORLD, 2U, field_order, geometry_order, threads, 1,
                       &problem, &resident) == 0);
  CHECK(fill_vector(problem.state, 2.0e-4, 11U) == 0);
  memset(&state, 0, sizeof(state));
  state.field = problem.state;
  state.reduction_factor = 1.0;
  state.external_load_factor = 1.0;
  CHECK(VecDuplicate(problem.state, &direction) == PETSC_SUCCESS);
  CHECK(VecDuplicate(problem.state, &shell_action) == PETSC_SUCCESS);
  CHECK(VecDuplicate(problem.state, &assembled_action) == PETSC_SUCCESS);
  CHECK(VecDuplicate(problem.state, &shell_residual) == PETSC_SUCCESS);
  CHECK(VecDuplicate(problem.state, &assembled_residual) == PETSC_SUCCESS);
  CHECK(fill_vector(direction, 1.0, 29U) == 0);
  CHECK(ss_operator_matrix_free_linearize(problem.matrix_free, &state, 0.0, &error) ==
        SS_OK);
  CHECK(ss_operator_jacobian(problem.assembled, &state, problem.jacobian, &error) ==
        SS_OK);
  CHECK(MatMult(problem.shell, direction, shell_action) == PETSC_SUCCESS);
  CHECK(MatMult(problem.jacobian, direction, assembled_action) == PETSC_SUCCESS);
  CHECK(ss_operator_residual(problem.matrix_free, &state, shell_residual, &error) ==
        SS_OK);
  CHECK(ss_operator_residual(problem.assembled, &state, assembled_residual, &error) ==
        SS_OK);
  action_error = relative_difference(shell_action, assembled_action);
  residual_error = relative_difference(shell_residual, assembled_residual);
  CHECK(MatMultTranspose(problem.shell, direction, shell_action) == PETSC_SUCCESS);
  CHECK(MatMultTranspose(problem.jacobian, direction, assembled_action) ==
        PETSC_SUCCESS);
  if (!(action_error <= 1.0e-11) || !(residual_error <= 1.0e-12) ||
      !(relative_difference(shell_action, assembled_action) <= 1.0e-11)) {
    PetscPrintf(PETSC_COMM_WORLD,
                "P%u/G%u threads=%d: action %.3e residual %.3e transpose %.3e\n",
                field_order, geometry_order, threads, action_error, residual_error,
                relative_difference(shell_action, assembled_action));
    return 1;
  }
  CHECK(VecDestroy(&direction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&shell_action) == PETSC_SUCCESS);
  CHECK(VecDestroy(&assembled_action) == PETSC_SUCCESS);
  CHECK(VecDestroy(&shell_residual) == PETSC_SUCCESS);
  CHECK(VecDestroy(&assembled_residual) == PETSC_SUCCESS);
  return problem_destroy(&problem);
}

#ifndef SS_CURVED_BENCH_BASELINE
/* The registered higher-rule policy reaches the operator: the paired rule
 * has more points, and at zero displacement both rules reproduce the curved
 * P4/G4 body load, a polynomial of degree 4 + 9 that each integrates exactly. */
static int check_paired_rule(void) {
  CurvedProblem problem;
  SsOperatorState state;
  SsErrorRecord error = {0};
  SsOperatorSpec spec;
  SsOperator *paired = NULL;
  SsAssemblyDiagnostics default_diagnostics;
  SsAssemblyDiagnostics paired_diagnostics;
  Vec default_residual = NULL;
  Vec paired_residual = NULL;
  double resident = 0.0;
  double difference;
  CHECK(problem_create(PETSC_COMM_WORLD, 2U, 4U, 4U, 1, 0, &problem, &resident) == 0);
  memset(&spec, 0, sizeof(spec));
  spec.physics = SS_ASSEMBLY_MECHANICS;
  spec.determinism = SS_ASSEMBLY_PERFORMANCE;
  spec.openmp_work_threshold = 1U;
  spec.requested_threads = 1;
  spec.mechanics = &problem.mechanics;
  spec.matrix_free = 1;
  spec.volume_quadrature_mode = SS_FE_QUADRATURE_MODE_PAIRED_HIGHER;
  CHECK(ss_operator_create(problem.plan, &spec, &paired, &error) == SS_OK);
  spec.volume_quadrature_mode = (SsFeQuadratureMode)7;
  {
    SsOperator *rejected = NULL;
    CHECK(ss_operator_create(problem.plan, &spec, &rejected, &error) != SS_OK);
    CHECK(rejected == NULL);
  }
  CHECK(ss_operator_diagnostics(problem.matrix_free, &default_diagnostics) == SS_OK);
  CHECK(ss_operator_diagnostics(paired, &paired_diagnostics) == SS_OK);
  CHECK(paired_diagnostics.quadrature_points > default_diagnostics.quadrature_points);
  CHECK(strcmp(paired_diagnostics.volume_quadrature_id,
               default_diagnostics.volume_quadrature_id) != 0);
  CHECK(VecSet(problem.state, 0.0) == PETSC_SUCCESS);
  memset(&state, 0, sizeof(state));
  state.field = problem.state;
  state.reduction_factor = 1.0;
  state.external_load_factor = 1.0;
  CHECK(VecDuplicate(problem.state, &default_residual) == PETSC_SUCCESS);
  CHECK(VecDuplicate(problem.state, &paired_residual) == PETSC_SUCCESS);
  CHECK(ss_operator_residual(problem.matrix_free, &state, default_residual, &error) ==
        SS_OK);
  CHECK(ss_operator_residual(paired, &state, paired_residual, &error) == SS_OK);
  difference = relative_difference(paired_residual, default_residual);
  if (!(difference < 1.0e-12)) {
    PetscPrintf(PETSC_COMM_WORLD, "paired residual difference %.3e\n", difference);
    return 1;
  }
  CHECK(VecDestroy(&default_residual) == PETSC_SUCCESS);
  CHECK(VecDestroy(&paired_residual) == PETSC_SUCCESS);
  CHECK(ss_operator_destroy(&paired) == SS_OK);
  return problem_destroy(&problem);
}

#endif

static int run_oracle(void) {
#ifndef SS_CURVED_BENCH_BASELINE
  CHECK(check_paired_rule() == 0);
#endif
  CHECK(check_curved_case(2U, 2U, 1) == 0);
  CHECK(check_curved_case(4U, 2U, 1) == 0);
  CHECK(check_curved_case(4U, 4U, 1) == 0);
#if SLOPE_ENABLE_OPENMP
  CHECK(check_curved_case(4U, 2U, 2) == 0);
#endif
  return 0;
}

static int run_benchmark(size_t n, uint32_t field_order, uint32_t geometry_order,
                         int threads, int repeats) {
  CurvedProblem problem;
  SsOperatorState state;
  SsErrorRecord error = {0};
  SsAssemblyDiagnostics diagnostics;
  Vec input = NULL;
  Vec output = NULL;
  double resident = 0.0;
  double resident_sum = 0.0;
  double times[4] = {0.0, 0.0, 0.0, 0.0};
  double maxima[4];
  double start;
  int size = 1;
  int repeat;
  CHECK(MPI_Comm_size(PETSC_COMM_WORLD, &size) == MPI_SUCCESS);
  CHECK(MPI_Barrier(PETSC_COMM_WORLD) == MPI_SUCCESS);
  start = MPI_Wtime();
  CHECK(problem_create(PETSC_COMM_WORLD, n, field_order, geometry_order, threads, 1,
                       &problem, &resident) == 0);
  times[0] = MPI_Wtime() - start;
  CHECK(fill_vector(problem.state, 2.0e-4, 11U) == 0);
  memset(&state, 0, sizeof(state));
  state.field = problem.state;
  state.reduction_factor = 1.0;
  state.external_load_factor = 1.0;
  CHECK(VecDuplicate(problem.state, &input) == PETSC_SUCCESS);
  CHECK(VecDuplicate(problem.state, &output) == PETSC_SUCCESS);
  CHECK(fill_vector(input, 1.0, 29U) == 0);
  for (repeat = 0; repeat < repeats + 1; ++repeat) {
    double elapsed[3];
    CHECK(MPI_Barrier(PETSC_COMM_WORLD) == MPI_SUCCESS);
    start = MPI_Wtime();
    CHECK(ss_operator_matrix_free_linearize(problem.matrix_free, &state, 0.0, &error) ==
          SS_OK);
    elapsed[0] = MPI_Wtime() - start;
    CHECK(MPI_Barrier(PETSC_COMM_WORLD) == MPI_SUCCESS);
    start = MPI_Wtime();
    CHECK(MatMult(problem.shell, input, output) == PETSC_SUCCESS);
    elapsed[1] = MPI_Wtime() - start;
    CHECK(MPI_Barrier(PETSC_COMM_WORLD) == MPI_SUCCESS);
    start = MPI_Wtime();
    CHECK(ss_operator_jacobian(problem.assembled, &state, problem.jacobian, &error) ==
          SS_OK);
    elapsed[2] = MPI_Wtime() - start;
    if (repeat > 0) { /* the first repeat warms caches and first touch */
      times[1] += elapsed[0] / repeats;
      times[2] += elapsed[1] / repeats;
      times[3] += elapsed[2] / repeats;
    }
  }
  CHECK(MPI_Allreduce(times, maxima, 4, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD) ==
        MPI_SUCCESS);
  CHECK(MPI_Allreduce(&resident, &resident_sum, 1, MPI_DOUBLE, MPI_SUM,
                      PETSC_COMM_WORLD) == MPI_SUCCESS);
  CHECK(ss_operator_diagnostics(problem.matrix_free, &diagnostics) == SS_OK);
  {
#ifdef SS_CURVED_BENCH_BASELINE
    /* The baseline accounts no geometry bytes; the benchmark reports zero. */
    uint64_t local[4] = {diagnostics.matrix_free_bytes, diagnostics.admitted_peak_bytes,
                         0U, 0U};
#else
    uint64_t local[4] = {diagnostics.matrix_free_bytes, diagnostics.admitted_peak_bytes,
                         diagnostics.geometry_cache_bytes,
                         diagnostics.quadrature_points};
#endif
    uint64_t global[4] = {0U, 0U, 0U, 0U};
    CHECK(MPI_Allreduce(local, global, 4, MPI_UINT64_T, MPI_SUM, PETSC_COMM_WORLD) ==
          MPI_SUCCESS);
    PetscPrintf(PETSC_COMM_WORLD,
                "{\"n\": %zu, \"cells\": %zu, \"field_order\": %u, "
                "\"geometry_order\": %u, \"ranks\": %d, \"threads\": %d, "
                "\"repeats\": %d, \"setup_s\": %.6f, \"linearize_s\": %.6f, "
                "\"matmult_s\": %.6f, \"jacobian_s\": %.6f, "
                "\"matrix_free_operator_rss_mb\": %.1f, \"matrix_free_bytes\": %llu, "
                "\"admitted_peak_bytes\": %llu, \"geometry_cache_bytes\": %llu, "
                "\"quadrature_points\": %llu}\n",
                n, 6U * n * n * n, field_order, geometry_order, size, threads, repeats,
                maxima[0], maxima[1], maxima[2], maxima[3], resident_sum,
                (unsigned long long)global[0], (unsigned long long)global[1],
                (unsigned long long)global[2], (unsigned long long)global[3]);
  }
  CHECK(VecDestroy(&input) == PETSC_SUCCESS);
  CHECK(VecDestroy(&output) == PETSC_SUCCESS);
  return problem_destroy(&problem);
}

int main(int argc, char **argv) {
  int result = 2;
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 3;
  }
  if (argc == 2 && strcmp(argv[1], "ASM-ORACLE-CURVED-MATFREE-001") == 0) {
    result = run_oracle();
  } else if (argc == 7 && strcmp(argv[1], "bench") == 0) {
    result = run_benchmark(
        (size_t)strtoul(argv[2], NULL, 10), (uint32_t)strtoul(argv[3], NULL, 10),
        (uint32_t)strtoul(argv[4], NULL, 10), atoi(argv[5]), atoi(argv[6]));
  } else {
    PetscPrintf(PETSC_COMM_WORLD,
                "usage: %s ASM-ORACLE-CURVED-MATFREE-001 | bench N P G THREADS "
                "REPEATS\n",
                argv[0]);
  }
  if (PetscFinalize() != PETSC_SUCCESS) {
    return 3;
  }
  return result;
}
